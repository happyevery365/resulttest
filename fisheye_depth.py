#!/usr/bin/env python3
"""One fisheye -> four pinhole views -> joint Pi3X -> fisheye range -> ONE cloud.

All mapping uses OpenCV integer pixel centers. Range is distance from the original
camera center, NOT optical-axis depth. Predicted world points/poses are never used
to fuse the views. No camera translations or multi-view parallax are fabricated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from outputs import save_ply, viewer

NAMES = ['left', 'center', 'right', 'up']
COLORS = [(255, 100, 90), (70, 220, 130), (70, 160, 255), (240, 190, 40)]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def new_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output must be new/empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def rotation(yaw, pitch):
    """Virtual camera -> source camera; x right, y down, z forward; pitch up."""
    y, p = np.deg2rad([yaw, pitch])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx


class KB4:
    """Monotone Kannala-Brandt / OPENCV_FISHEYE, including rays behind z=0.

    Spherical projection avoids the 90-degree branch ambiguity of pinhole-style
    normalized-coordinate undistortion. Bisection solves the radial polynomial.
    """
    def __init__(self, K, D, width, height, fov_deg):
        self.K = np.asarray(K, np.float64)
        self.D = np.asarray(D, np.float64)
        self.width, self.height = int(width), int(height)
        self.theta_max = np.deg2rad(float(fov_deg) / 2)
        if self.K.shape != (3, 3) or self.D.shape != (4,):
            raise ValueError('KB4 needs a 3x3 K and exactly four distortion coefficients')
        if not np.isfinite(self.K).all() or not np.isfinite(self.D).all():
            raise ValueError('Non-finite calibration')
        if min(self.K[0, 0], self.K[1, 1], self.width, self.height) <= 0 or not 0 < self.theta_max < np.pi:
            raise ValueError('Invalid focal length, image size or FOV')
        if not np.allclose(self.K[2], [0, 0, 1]) or abs(self.K[0, 1]) > 1e-9 or abs(self.K[1, 0]) > 1e-9:
            raise ValueError('This implementation requires zero-skew K')
        t = np.linspace(0, self.theta_max, 8193)
        derivative = np.ones_like(t)
        for i, k in enumerate(self.D):
            derivative += (2 * i + 3) * k * t ** (2 * i + 2)
        if np.min(derivative) <= 0:
            raise ValueError('Calibration folds within declared FOV; supply a verified monotone FOV')

    def radius(self, theta):
        out = np.ones_like(theta, dtype=np.float64)
        for i, k in enumerate(self.D):
            out += k * theta ** (2 * i + 2)
        return theta * out

    def unproject(self, u, v):
        x = (np.asarray(u) - self.K[0, 2]) / self.K[0, 0]
        y = (np.asarray(v) - self.K[1, 2]) / self.K[1, 1]
        rd = np.hypot(x, y)
        valid = np.isfinite(rd) & (rd <= self.radius(self.theta_max))
        lo, hi = np.zeros_like(rd), np.full_like(rd, self.theta_max)
        for _ in range(40):
            mid = (lo + hi) / 2
            below = self.radius(mid) < rd
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        theta = (lo + hi) / 2
        scale = np.sin(theta) / np.maximum(rd, 1e-15)
        rays = np.stack([x * scale, y * scale, np.cos(theta)], -1)
        rays = np.where((rd < 1e-15)[..., None], [0, 0, 1], rays)
        return np.where(valid[..., None], rays, [0, 0, 1]), valid

    def ray_map(self):
        v, u = np.indices((self.height, self.width), dtype=np.float64)
        return self.unproject(u, v)

    def project(self, rays):
        r = np.asarray(rays, np.float64)
        xy = np.linalg.norm(r[..., :2], axis=-1)
        theta = np.arctan2(xy, r[..., 2])
        rd = self.radius(theta)
        u = self.K[0, 0] * rd * r[..., 0] / np.maximum(xy, 1e-15) + self.K[0, 2]
        v = self.K[1, 1] * rd * r[..., 1] / np.maximum(xy, 1e-15) + self.K[1, 2]
        valid = (theta <= self.theta_max) & (np.linalg.norm(r, axis=-1) > 0)
        valid &= np.isfinite(r).all(-1) & (u >= 0) & (v >= 0) & (u <= self.width - 1) & (v <= self.height - 1)
        return u, v, valid


def load_camera(path, width, height):
    """Accept FIORD/Any3R COLMAP JSON or an explicit canonical OpenCV KB4 JSON."""
    raw = json.loads(Path(path).read_text())
    c = raw.get('camera', raw)
    if c.get('model') == 'OPENCV_FISHEYE' and 'params' in c:
        if len(c['params']) != 8:
            raise ValueError('OPENCV_FISHEYE must contain fx,fy,cx,cy,k1,k2,k3,k4')
        fx, fy, cx, cy, *d = c['params']
        # COLMAP has centers at .5; convert BEFORE OpenCV half-pixel resize.
        K = np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1]], float)
        convention = 'COLMAP .5 -> OpenCV integer centers'
    elif c.get('model') in ('kb4', 'OPENCV_FISHEYE') and c.get('pixel_convention') == 'opencv':
        K, d = np.array(c['K'], float), c['D']
        convention = 'OpenCV integer centers'
    else:
        raise ValueError('Need COLMAP OPENCV_FISHEYE params or model=kb4, K, D, pixel_convention=opencv')
    if 'fov_deg' not in c:
        raise ValueError('Specify calibrated valid fov_deg; the lens support cannot be guessed')
    sx, sy = width / int(c['width']), height / int(c['height'])
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + .5) * sx - .5
    K[1, 2] = (K[1, 2] + .5) * sy - .5
    cam = KB4(K, d, width, height, c['fov_deg'])
    meta = dict(model='kb4', K=K.tolist(), D=list(d), width=width, height=height,
                fov_deg=c['fov_deg'], pixel_convention='opencv', input_convention=convention,
                resize_note='Full-image resize assumed; cropped images require updated calibration.')
    return cam, meta


def pinhole_rays(K, size):
    v, u = np.indices((size, size), dtype=float)
    q = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def view_coordinates(source_rays, K, R, size):
    q = source_rays @ R  # R.T @ source_ray in column-vector notation
    z = q[..., 2]
    u = K[0, 0] * q[..., 0] / np.maximum(z, 1e-12) + K[0, 2]
    v = K[1, 1] * q[..., 1] / np.maximum(z, 1e-12) + K[1, 2]
    ok = (z > 1e-8) & (u >= 0) & (v >= 0) & (u <= size - 1) & (v <= size - 1)
    return u, v, z, ok


def prepare(image_path, calibration, out, size=448, fov=90., yaw=45., pitch=45.):
    if size <= 0 or size % 112 or not 0 < fov < 170 or not 0 < yaw < 180 or not 0 < pitch < 180:
        raise ValueError('size must be a positive multiple of 112; FOV in (0,170); angles in (0,180)')
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w = image.shape[:2]
    cam, camera_meta = load_camera(calibration, w, h)
    source_rays, source_valid = cam.ray_map()
    out = new_dir(out)
    f = size / (2 * np.tan(np.deg2rad(fov / 2)))
    K = np.array([[f, 0, (size - 1) / 2], [0, f, (size - 1) / 2], [0, 0, 1]])
    dirs = [(-yaw, 0), (0, 0), (yaw, 0), (0, pitch)]
    vrays = pinhole_rays(K, size)
    images, masks, rotations, coverage, meta = [], [], [], [], []
    overlay = image.copy()
    for name, color, (y, p) in zip(NAMES, COLORS, dirs):
        R = rotation(y, p)
        u, v, valid = cam.project(vrays @ R.T)
        rgb = cv2.remap(image, np.where(valid, u, -1).astype('f4'), np.where(valid, v, -1).astype('f4'), cv2.INTER_LINEAR)
        rgb[~valid] = 0
        if valid.mean() < .25:
            raise ValueError(f'{name} has less than 25% usable image; adjust angles/FOV')
        Image.fromarray(rgb).save(out / f'{name}.png')
        Image.fromarray(valid.astype('u1') * 255).save(out / f'{name}_valid.png')
        _, _, _, covered = view_coordinates(source_rays, K, R, size)
        covered &= source_valid
        contours, _ = cv2.findContours(covered.astype('u1'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        images.append(rgb); masks.append(valid); rotations.append(R); coverage.append(covered)
        meta.append(dict(name=name, yaw_deg=y, pitch_deg=p, K_opencv=K.tolist(),
                         R_source_virtual=R.tolist(), translation=[0, 0, 0], valid_fraction=float(valid.mean())))
    coverage = np.stack(coverage)
    Image.fromarray(image).save(out / 'source.png')
    canvas = Image.fromarray(overlay)
    draw = ImageDraw.Draw(canvas)
    for i, (name, color) in enumerate(zip(NAMES, COLORS)):
        draw.text((10, 10 + 20 * i), name, fill=color, stroke_width=1, stroke_fill='black')
    canvas.save(out / 'regions.png')
    montage = Image.new('RGB', (size * 2, size * 2 + 48), '#181818')
    d = ImageDraw.Draw(montage)
    for i, (name, rgb) in enumerate(zip(NAMES, images)):
        x, y = (i % 2) * size, (i // 2) * (size + 24)
        d.text((x + 8, y + 5), name, fill=COLORS[i])
        montage.paste(Image.fromarray(rgb), (x, y + 24))
    montage.save(out / 'four_views.png')
    count = coverage.sum(0).astype('u1')
    palette = np.array([[0, 0, 0], [70, 160, 255], [70, 220, 130], [240, 190, 40], [255, 100, 90]], 'u1')
    Image.fromarray(palette[count]).save(out / 'coverage.png')
    np.savez_compressed(out / 'prepared.npz', images=np.stack(images), valid=np.stack(masks),
                        K=np.repeat(K[None], 4, axis=0), R=np.stack(rotations), source_image=image,
                        source_rays=source_rays.astype('f4'), source_valid=source_valid, coverage=count)
    metadata = dict(source_image=str(Path(image_path).resolve()), calibration=str(Path(calibration).resolve()),
                    source_sha256=hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
                    calibration_sha256=hashlib.sha256(Path(calibration).read_bytes()).hexdigest(),
                    camera=camera_meta, views=meta, view_size=size, fov_deg=fov,
                    source_valid_pixels=int(source_valid.sum()), covered_pixels=int((count > 0).sum()),
                    covered_fraction_of_valid=float((count > 0).sum() / max(1, source_valid.sum())),
                    note='Four co-centered virtual views. No new parallax. Uncovered source pixels remain invalid.')
    dump(out / 'preparation.json', metadata)
    print(f'PREPARED {out}: four views; source coverage={metadata["covered_fraction_of_valid"]:.1%}', flush=True)
    return out


def depth_preview(path, depth, valid, limits=None):
    valid = valid & np.isfinite(depth) & (depth > 0)
    if limits is None:
        limits = np.percentile(depth[valid], [2, 98]) if valid.any() else (0, 1)
    lo, hi = limits
    scaled = np.where(valid, np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1), 0)
    rgb = cv2.cvtColor(cv2.applyColorMap((scaled * 255).astype('u1'), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    Image.fromarray(rgb).save(path)
    return [float(lo), float(hi)]


def infer(prepared_dir, repo, checkpoint, out, device='cuda', pose_prior=True):
    import torch
    from safetensors.torch import load_file
    if not torch.cuda.is_available() or not device.startswith('cuda'):
        raise RuntimeError('Pi3X inference needs working CUDA. prepare and fuse need CPU only.')
    repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
    if not (repo / 'pi3/models/pi3x.py').is_file() or not checkpoint.is_file():
        raise ValueError('Provide existing Pi3 repository and local Pi3X .safetensors checkpoint')
    src = Path(prepared_dir)
    data = np.load(src / 'prepared.npz', allow_pickle=False)
    out = new_dir(out)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    sys.path.insert(0, str(repo))
    from pi3.models.pi3x import Pi3X
    torch.set_num_threads(8)
    torch.manual_seed(0)
    print('Loading four views and allocating CUDA inputs...', flush=True)
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=torch.float32)
    images = tensor(data['images'] / 255.).permute(0, 3, 1, 2)[None].contiguous()
    K = data['K'].copy()
    K[:, :2, 2] += .5  # Pi3X's pixel grid is j+.5, i+.5
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, :3] = data['R']
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print('Loading local Pi3X checkpoint (no network downloads)...', flush=True)
    model = Pi3X(use_multimodal=True)
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model = model.to(device).eval()
    torch.cuda.synchronize()
    load_time = time.perf_counter() - start
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    start = time.perf_counter()
    print('Running ONE joint forward pass for left/center/right/up...', flush=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=precision):
        pred = model(imgs=images, intrinsics=tensor(K)[None], poses=tensor(poses)[None],
                     mask_add_ray=torch.ones((1, 4), dtype=torch.bool, device=device),
                     mask_add_pose=torch.full((1, 4), pose_prior, dtype=torch.bool, device=device),
                     mask_add_depth=torch.zeros((1, 4), dtype=torch.bool, device=device))
    torch.cuda.synchronize()
    forward_time = time.perf_counter() - start
    arr = lambda x: x.detach().float().cpu().numpy()
    # ONLY local z is used downstream. Do not take norm(local_points): Pi3X also
    # predicts xy; the known input K defines the rays for our depth reconstruction.
    local = arr(pred['local_points'][0])
    z = local[..., 2]
    conf = arr(pred['conf'][0, ..., 0].sigmoid())
    expected_rays = np.stack([pinhole_rays(k, z.shape[-1]) for k in data['K']])
    predicted_rays = local / np.maximum(np.linalg.norm(local, axis=-1, keepdims=True), 1e-12)
    angle = np.rad2deg(np.arccos(np.clip((expected_rays * predicted_rays).sum(-1), -1, 1)))
    good = data['valid'] & np.isfinite(z) & (z > 0)
    np.savez_compressed(out / 'view_depths.npz', depth_z=z, confidence=conf)
    np.save(out / 'predicted_poses_diagnostic_only.npy', arr(pred['camera_poses'][0]))
    for i, name in enumerate(NAMES):
        depth_preview(out / f'{name}_depth_z.png', z[i], good[i])
    manifest = dict(model='Pi3X', repo=str(repo), checkpoint=str(checkpoint),
                    python=sys.executable, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    precision=str(precision), load_seconds=load_time, inference_seconds=forward_time,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
                    input_views=4, joint_forward_passes=1, intrinsics_used=True, poses_used=pose_prior,
                    pose_prior='known virtual rotations with exactly zero translation',
                    depth_definition='pred[local_points][...,2]; optical-axis Z in each virtual camera',
                    depth_scale='model predicted; not validated in meters; no physical baseline',
                    finite_positive_depth_fraction=float(good.sum() / max(1, data['valid'].sum())),
                    predicted_ray_error_median_deg=float(np.median(angle[good])),
                    predicted_ray_error_p95_deg=float(np.percentile(angle[good], 95)),
                    prepared_sha256=hashlib.sha256((src / 'prepared.npz').read_bytes()).hexdigest(),
                    world_points_used=False, predicted_poses_used_for_fusion=False)
    dump(out / 'inference.json', manifest)
    print(f'INFERRED {out}: joint four-view forward {forward_time:.2f}s', flush=True)
    return out / 'view_depths.npz'


def sample_depth(z, confidence, valid, u, v, edge_ratio=1.1):
    """Bilinear on continuous valid surfaces; nearest at edges or invalid support.

    No interpolation through masked pixels, and no averaging across depth jumps.
    """
    h, w = z.shape
    x = np.clip(u, 0, w - 1)
    y = np.clip(v, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    vals = np.stack([z[y0, x0], z[y0, x1], z[y1, x0], z[y1, x1]])
    cs = np.stack([confidence[y0, x0], confidence[y0, x1], confidence[y1, x0], confidence[y1, x1]])
    good = np.stack([valid[y0, x0], valid[y0, x1], valid[y1, x0], valid[y1, x1]])
    good &= np.isfinite(vals) & (vals > 0) & np.isfinite(cs) & (cs > 0)
    good_all = good.all(0)
    clean = np.where(good, vals, 0)
    smooth = good_all & (clean.max(0) <= edge_ratio * np.maximum(clean.min(0), 1e-12))
    dx, dy = x - x0, y - y0
    ws = np.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
    ix, iy = np.floor(x + .5).astype(int), np.floor(y + .5).astype(int)
    zn, cn = z[iy, ix], confidence[iy, ix]
    nearest_ok = valid[iy, ix] & np.isfinite(zn) & (zn > 0) & np.isfinite(cn) & (cn > 0)
    zs = np.where(smooth, (ws * clean).sum(0), zn)
    c = np.where(smooth, (ws * np.where(good, cs, 0)).sum(0), cn)
    return zs, c, np.where(smooth, True, nearest_ok)


def weighted_median(values, weights):
    """One scalar per source pixel, never an average of foreground/background modes."""
    ok = np.isfinite(values) & (values > 0) & np.isfinite(weights) & (weights > 0)
    values, weights = np.where(ok, values, np.inf), np.where(ok, weights, 0)
    order = np.argsort(values, axis=0, kind='stable')
    sorted_v = np.take_along_axis(values, order, axis=0)
    sorted_w = np.take_along_axis(weights, order, axis=0)
    total = sorted_w.sum(0)
    # Strict > chooses the upper candidate on an exactly equal-weight tie.
    idx = np.argmax(np.cumsum(sorted_w, axis=0) > total[None] / 2, axis=0)
    depth = np.take_along_axis(sorted_v, idx[None], axis=0)[0]
    owner = np.take_along_axis(order, idx[None], axis=0)[0]
    return np.where(total > 0, depth, np.nan), np.where(total > 0, owner, -1), ok


def backproject_depths(data, depths, confidence, min_confidence=0., edge_ratio=1.1):
    images, vr_valid = data['images'], data['valid']
    if depths.shape != images.shape[:3] or confidence.shape != depths.shape:
        raise ValueError(f'Expected depth/confidence {images.shape[:3]}, got {depths.shape}/{confidence.shape}')
    if not 0 <= min_confidence < 1 or edge_ratio <= 1:
        raise ValueError('min-confidence in [0,1); edge-ratio > 1')
    valid_conf = confidence[np.isfinite(confidence)]
    if len(valid_conf) and (valid_conf.min() < 0 or valid_conf.max() > 1):
        raise ValueError('confidence must be probabilities in [0,1], not raw logits')
    size = depths.shape[-1]
    candidates, weights = [], []
    for i in range(4):
        u, v, rz, inside = view_coordinates(data['source_rays'], data['K'][i], data['R'][i], size)
        ok_pixels = vr_valid[i] & np.isfinite(confidence[i]) & (confidence[i] > min_confidence)
        z, c, sample_ok = sample_depth(depths[i], confidence[i], ok_pixels, u, v, edge_ratio)
        valid = data['source_valid'] & inside & sample_ok & np.isfinite(z) & (z > 0)
        # Z = radial_range * (R_source_virtual.T @ source_unit_ray).z
        radial = z / np.maximum(rz, 1e-12)
        border = np.minimum.reduce([u, v, size - 1 - u, size - 1 - v])
        taper = np.clip((border + 1) / (size * .1), .01, 1)
        weight = c * np.maximum(rz, 0) ** 2 * taper
        candidates.append(np.where(valid, radial, np.nan))
        weights.append(np.where(valid, weight, 0))
    candidates, weights = np.stack(candidates), np.stack(weights)
    radial, owner, available = weighted_median(candidates, weights)
    counts = available.sum(0).astype('u1')
    min_r = np.min(np.where(available, candidates, np.inf), axis=0)
    max_r = np.max(np.where(available, candidates, -np.inf), axis=0)
    spread = np.full_like(radial, np.nan)
    overlaps = counts >= 2
    spread[overlaps] = (max_r[overlaps] - min_r[overlaps]) / radial[overlaps]
    return radial, owner, counts, spread, candidates, weights


def fuse(prepared_dir, prediction_path, out, min_confidence=0., edge_ratio=1.1):
    src, prediction_path = Path(prepared_dir), Path(prediction_path)
    data = np.load(src / 'prepared.npz', allow_pickle=False)
    pred = np.load(prediction_path, allow_pickle=False)
    # Catch accidental mixing of predictions from different preparations.
    inference_manifest = prediction_path.parent / 'inference.json'
    if inference_manifest.exists():
        expected = json.loads(inference_manifest.read_text())['prepared_sha256']
        if hashlib.sha256((src / 'prepared.npz').read_bytes()).hexdigest() != expected:
            raise ValueError('Predictions belong to a different prepared.npz')
    radial, owner, counts, spread, candidates, weights = backproject_depths(
        data, pred['depth_z'], pred['confidence'], min_confidence, edge_ratio)
    good = np.isfinite(radial) & (radial > 0) & data['source_valid']
    if not good.any():
        raise ValueError('No valid reconstructed source pixels')
    out = new_dir(out)
    np.save(out / 'fisheye_range.npy', radial.astype('f4'))
    np.save(out / 'fisheye_valid.npy', good)
    # Persist the depth intermediate FIRST; the sole final cloud is generated
    # from this saved source-resolution depth, not the four virtual point maps.
    saved_range = np.load(out / 'fisheye_range.npy')
    v, u = np.nonzero(good)
    xyz = data['source_rays'][good] * saved_range[good, None]
    rgb = data['source_image'][good]
    save_ply(out / 'points.ply', xyz, rgb)
    np.save(out / 'points.npy', xyz)
    np.save(out / 'point_source_uv.npy', np.stack([u, v], -1).astype('i4'))
    viewer(out / 'viewer.html', xyz, rgb)
    limits = depth_preview(out / 'fisheye_range.png', radial, good)
    for i, name in enumerate(NAMES):
        depth_preview(out / f'{name}_range_on_fisheye.png', candidates[i], weights[i] > 0, limits)
    palette = np.array([(0, 0, 0)] + COLORS, 'u1')
    Image.fromarray(palette[owner + 1]).save(out / 'selected_view.png')
    Image.fromarray(good.astype('u1') * 255).save(out / 'fisheye_valid.png')
    np.savez_compressed(out / 'depth_fusion_diagnostics.npz', candidate_range=candidates.astype('f4'),
                        weights=weights.astype('f4'), contributor_count=counts,
                        selected_view=owner.astype('i1'), relative_spread=spread.astype('f4'))
    overlaps = good & (counts >= 2)
    report = dict(method='inverse warp optical-axis Z to original fisheye radial range; weighted median',
                  weight='sigmoid confidence * virtual_ray_z^2 * border taper',
                  interpolation='bilinear within smooth valid surfaces, nearest at discontinuities/invalid support',
                  interpolation_edge_ratio=edge_ratio, min_confidence=min_confidence,
                  source_valid_pixels=int(data['source_valid'].sum()), reconstructed_pixels=int(good.sum()),
                  reconstructed_fraction_of_source_valid=float(good.sum() / max(1, data['source_valid'].sum())),
                  overlap_pixels=int(overlaps.sum()),
                  overlap_relative_spread_p50_p95=np.percentile(spread[overlaps], [50, 95]).tolist() if overlaps.any() else None,
                  range_p05_p50_p95=np.percentile(radial[good], [5, 50, 95]).tolist(),
                  depth_preview_limits=limits, exported_points=len(xyz),
                  one_point_per_valid_source_pixel=True, point_cloud_fusion=False,
                  predicted_poses_used=False, coordinate_frame='original fisheye camera: right/down/forward',
                  depth_definition='radial distance; NaN where unobserved or rejected; not optical-axis Z',
                  scale='model scale; metric accuracy unverified', ground_truth_evaluated=False,
                  note='Overlap consistency is a diagnostic, not ground-truth accuracy. No per-view scale fitting or hole filling.',
                  prediction_file=str(prediction_path.resolve()),
                  prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest())
    dump(out / 'metrics.json', report)
    print(f'FUSED DEPTH {out}: {len(xyz)} unique original pixels -> {len(xyz)} points', flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run'):
        s = subs.add_parser(name)
        s.add_argument('--image', type=Path, required=True)
        s.add_argument('--calibration', type=Path, required=True)
        s.add_argument('--out', type=Path, required=True)
        s.add_argument('--size', type=int, default=448)
        s.add_argument('--fov', type=float, default=90)
        s.add_argument('--yaw', type=float, default=45)
        s.add_argument('--pitch', type=float, default=45)
    s = subs.add_parser('infer')
    s.add_argument('--prepared', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    for name in ('infer', 'run'):
        s = subs.choices[name]
        s.add_argument('--pi3x-repo', type=Path, required=True)
        s.add_argument('--checkpoint', type=Path, required=True)
        s.add_argument('--device', default='cuda:0')
        s.add_argument('--no-pose-prior', action='store_true')
    s = subs.add_parser('fuse')
    s.add_argument('--prepared', type=Path, required=True)
    s.add_argument('--predictions', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    for name in ('fuse', 'run'):
        s = subs.choices[name]
        s.add_argument('--min-confidence', type=float, default=0)
        s.add_argument('--edge-ratio', type=float, default=1.1)
    return p


def main():
    a = parser().parse_args()
    if a.command == 'prepare':
        prepare(a.image, a.calibration, a.out, a.size, a.fov, a.yaw, a.pitch)
    elif a.command == 'infer':
        infer(a.prepared, a.pi3x_repo, a.checkpoint, a.out, a.device, not a.no_pose_prior)
    elif a.command == 'fuse':
        fuse(a.prepared, a.predictions, a.out, a.min_confidence, a.edge_ratio)
    else:
        root = new_dir(a.out)
        prepared = prepare(a.image, a.calibration, root / 'prepared', a.size, a.fov, a.yaw, a.pitch)
        prediction = infer(prepared, a.pi3x_repo, a.checkpoint, root / 'inference', a.device, not a.no_pose_prior)
        fuse(prepared, prediction, root / 'fused', a.min_confidence, a.edge_ratio)


if __name__ == '__main__':
    main()
