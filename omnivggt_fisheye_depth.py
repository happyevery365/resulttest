#!/usr/bin/env python3
"""OmniVGGT backend for the shared fisheye-depth-first reconstruction pipeline."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from fisheye_depth import NAMES, depth_preview, dump, fuse, new_dir, prepare


def camera_inputs(data):
    images, K, R = data['images'], data['K'], data['R']
    if images.ndim != 4 or images.shape[0] != 4 or images.shape[-1] != 3:
        raise ValueError('Expected four RGB images [4,H,W,3]')
    h, w = images.shape[1:3]
    if h != w or h % 14 or K.shape != (4, 3, 3) or R.shape != (4, 3, 3):
        raise ValueError('Expected square patch-14 compatible images and four K/R matrices')
    if not np.isfinite(K).all() or not np.isfinite(R).all():
        raise ValueError('Non-finite camera parameters')
    if not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(R), 1):
        raise ValueError('Camera rotations must be proper rotations')
    expected = np.zeros_like(K)
    expected[:, 0, 0], expected[:, 1, 1] = K[:, 0, 0], K[:, 1, 1]
    expected[:, 0, 2], expected[:, 1, 2], expected[:, 2, 2] = (w-1)/2, (h-1)/2, 1
    if not np.allclose(K, expected) or np.any(K[:, [0, 1], [0, 1]] <= 0):
        raise ValueError('OmniVGGT camera tokens require centered, zero-skew perspective K')
    w2c = np.zeros((4, 3, 4), dtype='f4')
    w2c[:, :3, :3] = R.transpose(0, 2, 1)
    return K.astype('f4'), w2c


def confidence_weights(raw, valid):
    """One shared positive scale preserves cross-view confidence ratios, not probabilities."""
    good = valid & np.isfinite(raw) & (raw > 0)
    if not good.any():
        raise ValueError('No finite positive confidence on valid image pixels')
    scale = float(np.max(raw[good]))
    return np.where(good, raw / scale, 0).astype('f4'), scale


def infer(prepared, repo, checkpoint, out, device='cuda:0', camera_prior=True):
    import torch
    from safetensors.torch import load_file

    prepared, repo, checkpoint = map(lambda p: Path(p).resolve(), (prepared, repo, checkpoint))
    if not checkpoint.is_file() or not (repo / 'omnivggt/models/omnivggt.py').is_file():
        raise ValueError('Provide local OmniVGGT source and a .safetensors checkpoint file')
    if not torch.cuda.is_available() or not device.startswith('cuda'):
        raise RuntimeError('OmniVGGT inference requires working CUDA')
    # The local upstream patch only suppresses a redundant DINO initialization download.
    # Full model state is loaded strictly below, so no random backbone weights remain.
    if 'OMNIVGGT_SKIP_BACKBONE_DOWNLOAD' not in (repo / 'omnivggt/models/aggregator.py').read_text():
        raise RuntimeError('First apply patch_omnivggt_offline.py to the model repository')
    os.environ['OMNIVGGT_SKIP_BACKBONE_DOWNLOAD'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    sys.path.insert(0, str(repo))
    from omnivggt.models.omnivggt import OmniVGGT

    data = np.load(prepared / 'prepared.npz', allow_pickle=False)
    K, w2c = camera_inputs(data)
    h, w = data['images'].shape[1:3]
    out = new_dir(out)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    images = tensor(data['images'].astype('f4') / 255).permute(0, 3, 1, 2)[None].contiguous()
    inputs = dict(images=images, intrinsics=tensor(K)[None], extrinsics=tensor(w2c)[None],
                  depth=torch.zeros((1, 4, h, w, 1), device=device),
                  mask=torch.zeros((1, 4, h, w), device=device),
                  depth_gt_index=[], camera_gt_index=list(range(4)) if camera_prior else [])
    print('Loading local OmniVGGT checkpoint (strict, offline)...', flush=True)
    started = time.perf_counter()
    model = OmniVGGT()
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model = model.to(device).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'Joint OmniVGGT inference: four {w}x{h} views; camera prior={camera_prior}', flush=True)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=precision):
        pred = model.inference(**inputs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    z = pred['depth'][0, ..., 0].float().cpu().numpy()
    raw_conf = pred['depth_conf'][0].float().cpu().numpy()
    if z.shape != (4, h, w) or raw_conf.shape != z.shape:
        raise ValueError('Unexpected model output size; refusing implicit resize of depth')
    valid = data['valid'] & np.isfinite(z) & (z > 0)
    confidence, scale = confidence_weights(raw_conf, valid)
    np.savez_compressed(out / 'view_depths.npz', depth_z=z, confidence=confidence, raw_confidence=raw_conf)
    for i, name in enumerate(NAMES):
        depth_preview(out / f'{name}_depth_z.png', z[i], valid[i])
    np.savez(out / 'input_cameras.npz', intrinsics=K, world_to_camera=w2c)
    np.save(out / 'predicted_pose_encoding_diagnostic_only.npy', pred['pose_enc'][0].float().cpu().numpy())
    dump(out / 'inference.json', dict(
        model='OmniVGGT', repo=str(repo), checkpoint=str(checkpoint), python=sys.executable,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), precision=str(precision),
        load_seconds=load_seconds, inference_seconds=seconds,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
        input_views=4, input_size=[h, w], joint_forward_passes=1, additional_resize=False,
        intrinsics_used=camera_prior, poses_used=camera_prior, depth_prior_used=False,
        pose_prior='world-to-camera [R_source_virtual.T | 0]; upstream normalizes to first view',
        camera_encoding_note='Upstream encodes rotation, translation and fx/fy-derived FOV, not principal point or distortion. Shared remap uses centered OpenCV K=(S-1)/2; upstream decoded K assumes S/2 (0.5 pixel difference). Backprojection uses the known remap K, never decoded predicted K.',
        depth_definition='pred[depth][0,...,0]; optical-axis Z, confirmed by upstream depth unprojection',
        confidence_definition='upstream 1+exp(logit), divided by ONE maximum across all valid views; relative weights, NOT calibrated probabilities',
        confidence_divisor=scale,
        finite_positive_depth_fraction=float(valid.sum() / max(1, data['valid'].sum())),
        depth_scale='model predicted; not validated in meters; no physical baseline',
        prepared_sha256=hashlib.sha256((prepared / 'prepared.npz').read_bytes()).hexdigest(),
        world_points_used=False, predicted_poses_used_for_fusion=False))
    print(f'INFERRED {out}: {seconds:.2f}s', flush=True)
    return out / 'view_depths.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prepared', type=Path, help='Reuse Pi3X prepared directory, identical RGB/K/R')
    source.add_argument('--image', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--size', type=int, default=672)
    parser.add_argument('--fov', type=float, default=90)
    parser.add_argument('--yaw', type=float, default=45)
    parser.add_argument('--pitch', type=float, default=45)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-camera-prior', action='store_true', help='RGB-only ablation; disables BOTH K and pose conditioning')
    parser.add_argument('--min-confidence', type=float, default=0, help='Threshold on globally rescaled relative weights, not probabilities')
    parser.add_argument('--edge-ratio', type=float, default=1.1)
    args = parser.parse_args()
    if args.image and not args.calibration:
        parser.error('--image requires --calibration')
    if args.prepared and not (args.prepared / 'prepared.npz').is_file():
        parser.error('--prepared must contain prepared.npz')
    root = new_dir(args.out)
    if args.prepared:
        # Retain an explicit link for the web viewer without duplicating the large ray maps.
        (root / 'prepared').symlink_to(args.prepared.resolve(), target_is_directory=True)
        print('Reusing prepared inputs exactly; size/FOV/yaw/pitch arguments are not applied.', flush=True)
    else:
        prepare(args.image, args.calibration, root / 'prepared', args.size, args.fov, args.yaw, args.pitch)
    predictions = infer(root / 'prepared', args.repo, args.checkpoint, root / 'inference',
                        args.device, not args.no_camera_prior)
    fuse(root / 'prepared', predictions, root / 'fused', args.min_confidence, args.edge_ratio)
    # Shared geometry is identical, but the source of confidence weights differs from Pi3X.
    import json
    path = root / 'fused/metrics.json'
    metrics = json.loads(path.read_text())
    metrics.update(model='OmniVGGT', weight='globally rescaled raw depth confidence * virtual_ray_z^2 * border taper',
                   confidence_note='Relative model score; not a probability and not comparable to Pi3X confidence')
    dump(path, metrics)


if __name__ == '__main__':
    main()
