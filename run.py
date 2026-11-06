#!/usr/bin/env python3
"""Reconstruct one dataset case or one image using an existing local model install."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from dataset import DEFAULT_REGEX, dump, inspect, select


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--case', help='Case containing conf/ and picture/')
    source.add_argument('--image', help='One case/picture/cam_N/NUMBER.jpg')
    p.add_argument('--cameras', help='Comma-separated IDs, e.g. 0,1,4')
    p.add_argument('--frame-id', help='Exact shared numeric identifier; defaults to smallest common number')
    p.add_argument('--frame-regex', default=DEFAULT_REGEX)
    p.add_argument('--out', type=Path, required=True, help='New output directory; existing nonempty directory is refused')
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('paths.json'))
    p.add_argument('--calibration-overrides', type=Path, help='JSON camera-ID -> complete verified canonical calibration')
    p.add_argument('--inspect-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--model', choices=['pi3x', 'omnivggt', 'any3r'])
    p.add_argument('--mode', choices=['raw', 'remap'], default='raw')
    p.add_argument('--size', type=int, default=448, help='Square padded canvas; positive multiple of 112')
    p.add_argument('--fov', type=float, default=100, help='Horizontal/vertical FOV of square virtual perspective views')
    p.add_argument('--directions', default='0,0;-50,0;50,0;0,-50;0,50', help='yaw,pitch degrees separated by semicolons')
    p.add_argument('--min-coverage', type=float, default=.25)
    p.add_argument('--drop-confidence-percentile', type=float, default=20)
    p.add_argument('--no-camera-priors', action='store_true', help='Disable Pi3X/OmniVGGT camera conditions for controlled comparison; Any3R still requires rays')
    return p


def infer(a, cfg, prepared):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for model inference')
    torch.set_num_threads(8)
    torch.manual_seed(0)
    spec = cfg[a.model]
    repo = Path(spec['repo']).expanduser().resolve()
    checkpoint = Path(spec['checkpoint']).expanduser().resolve()
    sys.path.insert(0, str(repo))
    n = len(prepared['views'])
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device='cuda')
    def array(x):
        return x.detach().float().cpu().numpy()
    images = tensor(prepared['images'] / 255.).permute(0, 3, 1, 2).contiguous()
    perspective = [v['perspective'] for v in prepared['views']]
    calibrated_poses = [p is not None for p in prepared['poses']]
    use_k = [x and not a.no_camera_priors for x in perspective]
    use_pose = [k and pose for k, pose in zip(use_k, calibrated_poses)]
    priors = {'perspective_views': perspective, 'intrinsics_used': [], 'poses_used': [], 'notes': []}
    started = time.perf_counter()
    if a.model == 'pi3x':
        from pi3.models.pi3x import Pi3X
        model = Pi3X()
        model.load_state_dict(load_file(str(checkpoint)), strict=True)
        K = prepared['K'].copy()
        # Pi3X get_pixel() uses j+.5/i+.5; canonical stored K uses OpenCV integer centers.
        K[:, :2, 2] += .5
        if sum(use_pose) < 2:
            use_pose = [False] * n
        pose_array = np.stack([np.eye(4) if x is None else x for x in prepared['poses']])
        inputs = dict(imgs=images[None], intrinsics=tensor(K)[None], poses=tensor(pose_array)[None],
                      mask_add_ray=torch.tensor([use_k], device='cuda'),
                      mask_add_pose=torch.tensor([use_pose], device='cuda'),
                      mask_add_depth=torch.zeros((1, n), dtype=torch.bool, device='cuda'))
        priors.update(intrinsics_used=use_k, poses_used=use_pose)
        priors['notes'].append('Raw fisheye/distorted views receive RGB only. Perspective views receive K with .5-center conversion. No fisheye K is passed as a perspective K.')
    elif a.model == 'omnivggt':
        agg = repo / 'omnivggt/models/aggregator.py'
        if 'OMNIVGGT_SKIP_BACKBONE_DOWNLOAD' not in agg.read_text():
            raise RuntimeError('Apply patch_omnivggt_offline.py to the OmniVGGT repo first; full weights will then be loaded strictly')
        os.environ['OMNIVGGT_SKIP_BACKBONE_DOWNLOAD'] = '1'
        from omnivggt.models.omnivggt import OmniVGGT
        from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
        model = OmniVGGT()
        model.load_state_dict(load_file(str(checkpoint)), strict=True)
        # Camera token contains only rotation/translation/fx/fy-derived FoV. Off-center K cannot be represented.
        centered = [np.allclose(k[:2, 2], [(a.size - 1) / 2] * 2, atol=1) for k in prepared['K']]
        use_pose = [p and c for p, c in zip(use_pose, centered)]
        ids = [i for i, x in enumerate(use_pose) if x]
        extrinsics = np.zeros((n, 3, 4), np.float32)
        for i in ids:
            extrinsics[i] = np.linalg.inv(prepared['poses'][i])[:3]
        inputs = dict(images=images, extrinsics=tensor(extrinsics)[None], intrinsics=tensor(prepared['K'])[None],
                      depth=torch.zeros((1, n, a.size, a.size, 1), device='cuda'),
                      mask=torch.zeros((1, n, a.size, a.size), device='cuda'),
                      depth_gt_index=[], camera_gt_index=ids)
        priors.update(intrinsics_used=use_pose, poses_used=use_pose)
        priors['notes'].append('K-only conditioning is unavailable. Only centered perspective views with explicit T_world_camera use camera conditions. Distortion and off-center principal point are not encoded. Raw fisheye is an uncalibrated baseline.')
    else:
        from any3r import Any3R
        if not checkpoint.is_dir() or not (checkpoint / 'config.json').is_file() or not (checkpoint / 'model.safetensors').is_file():
            raise ValueError('Any3R checkpoint must be a LOCAL directory containing config.json and model.safetensors')
        # Explicit local construction prevents an accidental Hub fallback.
        model = Any3R(**json.loads((checkpoint / 'config.json').read_text()))
        model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')), strict=True)
        inputs = dict(images=images.contiguous(memory_format=torch.channels_last), ray_maps=tensor(prepared['rays']))
        priors.update(intrinsics_used=[True] * n, poses_used=[False] * n)
        priors['notes'].append('Calibrated unit ray maps incorporate projection, K and distortion; poses are predicted. Any3R infer API does not accept known poses.')
    print('CAMERA_PRIORS ' + json.dumps(priors, ensure_ascii=False), flush=True)
    model = model.cuda().eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast('cuda', dtype=precision):
        if a.model == 'pi3x':
            pred = model(**inputs)
        elif a.model == 'omnivggt':
            pred = model.inference(**inputs)
        else:
            # Match upstream infer's channels-last layout; retain batch axis for common exports.
            pred = model(**inputs)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    if a.model == 'pi3x':
        points, poses = array(pred['points'][0]), array(pred['camera_poses'][0])
        confidence = array(pred['conf'][0, ..., 0].sigmoid())
    elif a.model == 'omnivggt':
        ex, K = pose_encoding_to_extri_intri(pred['pose_enc'], (a.size, a.size))
        w2c = np.tile(np.eye(4), (n, 1, 1))
        w2c[:, :3] = array(ex[0])
        points, poses = array(pred['world_points'][0]), np.linalg.inv(w2c)
        confidence = array(pred['world_points_conf'][0])
        np.save(a.out / 'predicted_intrinsics.npy', array(K[0]))
    else:
        points, poses = array(pred['world_points'][0]), array(pred['camera_poses'][0])
        confidence = array(pred['depth_conf'][0])
    return points, confidence, poses, {
        'model': a.model, 'mode': a.mode, 'repo': str(repo), 'checkpoint': str(checkpoint),
        'python': sys.version, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'precision': str(precision), 'model_load_seconds': load_seconds, 'inference_seconds': inference_seconds,
        'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3, 'priors': priors,
        'native_3dgs_head': False, 'timing_note': 'Single non-warmed forward pass; not a speed benchmark'}


def main():
    a = parser().parse_args()
    if a.size <= 0 or a.size % 112 or not 0 < a.fov < 180 or not 0 <= a.drop_confidence_percentile < 100 or not 0 < a.min_coverage <= 1:
        raise ValueError('Invalid size/FOV/filter settings; size must be a positive multiple of 112')
    a.out = a.out.expanduser().absolute()
    case, selected, selection = select(a.case, a.image, a.cameras, a.frame_id, a.frame_regex)
    if a.out.resolve().is_relative_to(case.resolve()):
        raise ValueError('Keep output outside the source case directory')
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Output directory is nonempty; choose a new --out to preserve prior runs')
    a.out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(a.calibration_overrides.read_text()) if a.calibration_overrides else None
    report = inspect(case, selected, selection, overrides)
    dump(a.out / 'inspection.json', report)
    print('SELECTION ' + json.dumps(selection, ensure_ascii=False), flush=True)
    for rec in report['cameras']:
        cal = rec.get('calibration', {})
        fish = ('YES' if cal['fisheye'] else 'NO') if cal else 'UNKNOWN'
        print(f'{rec["camera"]}: fisheye={fish}; {cal.get("reason", rec.get("error"))}; {rec["image"]}', flush=True)
        for warning in cal.get('warnings', []):
            print(f'  NOTE: {warning}', flush=True)
    if a.inspect_only:
        print(f'Inspection written to {a.out / "inspection.json"}; ready={report["ready"]}', flush=True)
        return
    if not report['ready']:
        raise ValueError('Unsupported calibration. See inspection.json; provide a verified calibration override or adapt dataset.normalize() before inference')
    if not a.prepare_only and not a.model:
        raise ValueError('--model is required for inference')
    cfg = json.loads(a.config.read_text())
    directions = [tuple(map(float, item.split(','))) for item in a.directions.split(';')]
    if any(len(d) != 2 for d in directions) or not directions:
        raise ValueError('--directions must contain yaw,pitch pairs')
    from geometry import prepare
    prepared = prepare(report, cfg['any3r']['repo'], a.out, a.size, a.mode, a.fov, directions, a.min_coverage)
    dump(a.out / 'views.json', prepared['views'])
    import numpy as np
    np.savez_compressed(a.out / 'camera_inputs.npz', intrinsics_opencv=prepared['K'], rays=prepared['rays'], valid=prepared['valid'])
    print(f'PREPARED {len(prepared["views"])} views. Remapping adds directions, not new camera centers or parallax.', flush=True)
    if len(prepared['views']) > 24:
        print('NOTE: Many views can require substantial GPU memory. Reduce --cameras, --directions or --size if necessary.', flush=True)
    if a.prepare_only:
        return
    points, confidence, poses, run = infer(a, cfg, prepared)
    from outputs import export
    metrics = export(a.out, prepared, points, confidence, poses, a.drop_confidence_percentile)
    run['selection'] = selection
    run['adapter_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    dump(a.out / 'metrics.json', metrics)
    dump(a.out / 'run.json', run)
    print(f'DONE {a.out}: {metrics["exported_points"]} points; open viewer.html or points.ply', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        raise
