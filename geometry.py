"""Camera-aware preprocessing. Source files are never changed."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def camera_library(repo):
    # Load the NumPy-only upstream geometry module without importing the neural network.
    path = Path(repo) / 'any3r/cameras.py'
    spec = importlib.util.spec_from_file_location('rig_any3r_cameras', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_camera(lib, cal):
    K = np.array(cal['K'])
    if cal['kind'] == 'equidistant':
        return lib.Equidistant(K[0, 0], K[1, 1], K[0, 2] + .5, K[1, 2] + .5,
                               width=cal['width'], height=cal['height'], fov_deg=cal['valid_fov_deg'])
    return lib.from_opencv(K, cal['D'], cal['width'], cal['height'],
                           fisheye=cal['kind'] == 'kb4', fov_deg=cal['valid_fov_deg'])


def rotation(yaw, pitch):
    y, p = np.deg2rad([yaw, pitch])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx  # virtual camera -> source camera; positive pitch looks upward


def remap_view(image, source, lib, size, fov, yaw, pitch):
    f = size / (2 * np.tan(np.deg2rad(fov / 2)))
    target = lib.Pinhole(f, f, size / 2, size / 2, width=size, height=size)
    rays, _ = target.ray_map()
    R = rotation(yaw, pitch)
    dirs = rays @ R.T
    u, v, good = source.project(dirs)
    # Upstream cameras use .5 centers, OpenCV remap uses integer centers.
    x, y = u - .5, v - .5
    good &= np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= image.shape[1] - 1) & (y <= image.shape[0] - 1)
    # Check that source projection is invertible on the selected branch.
    recovered, inverse_ok = source.pixel_to_ray(u, v)
    recovered /= np.maximum(np.linalg.norm(recovered, axis=-1, keepdims=True), 1e-12)
    good &= inverse_ok & (np.sum(recovered * dirs, axis=-1) > 1 - 1e-5)
    x = np.where(good, x, -1).astype(np.float32)
    y = np.where(good, y, -1).astype(np.float32)
    rgb = cv2.remap(image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rgb[~good] = 0
    return rgb, target, good, R


def prepare(report, repo, out, size, mode, fov, directions, min_coverage=.25):
    lib = camera_library(repo)
    prepared = []
    out = Path(out)
    (out / 'inputs').mkdir()
    for rec in report['cameras']:
        image = np.asarray(Image.open(rec['image']).convert('RGB'))
        cal = rec['calibration']
        camera = make_camera(lib, cal)
        if mode == 'remap' and cal['kind'] != 'pinhole':
            views = directions if cal['fisheye'] else [(0, 0)]
            for index, (yaw, pitch) in enumerate(views):
                rgb, cam, valid, R = remap_view(image, camera, lib, size, fov, yaw, pitch)
                coverage = float(valid.mean())
                if coverage < min_coverage:
                    print(f'SKIP {rec["camera"]} yaw={yaw} pitch={pitch}: valid coverage={coverage:.3f}', flush=True)
                    continue
                prepared.append((rec, rgb, cam, valid, R, f'{rec["camera"]}_view{index:02d}', True,
                                 {'yaw_deg': yaw, 'pitch_deg': pitch, 'fov_deg': fov}))
        else:
            _, valid = camera.ray_map()
            prepared.append((rec, image, camera, valid, np.eye(3), rec['camera'], cal['kind'] == 'pinhole', {}))
    if not prepared:
        raise ValueError('No valid views after remap')
    images, rays, masks, intrinsics, poses, metadata = [], [], [], [], [], []
    for rec, rgb, cam, valid, R, name, perspective, angles in prepared:
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # Require all bilinear contributors to be valid at remap boundaries.
        vm = cv2.resize(valid.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR) > .999
        cam = cam.resized(nw, nh)
        rr, lens_valid = cam.ray_map()
        vm &= lens_valid
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas = np.zeros((size, size, 3), np.uint8)
        ray = np.zeros((size, size, 3), np.float32)
        ray[..., 2] = 1
        mask = np.zeros((size, size), bool)
        canvas[top:top + nh, left:left + nw] = np.where(vm[..., None], resized, 0)
        ray[top:top + nh, left:left + nw] = np.where(vm[..., None], rr, [0, 0, 1])
        mask[top:top + nh, left:left + nw] = vm
        # This is a physical perspective K only if perspective=True.
        K = np.array([[cam.fx, 0, cam.cx - .5 + left], [0, cam.fy, cam.cy - .5 + top], [0, 0, 1]], np.float32)
        pose = rec['calibration']['T_world_camera']
        transform = np.eye(4)
        transform[:3, :3] = R
        pose = None if pose is None else np.array(pose) @ transform
        Image.fromarray(canvas).save(out / 'inputs' / (name + '.png'))
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / 'inputs' / (name + '_mask.png'))
        images.append(canvas); rays.append(ray); masks.append(mask); intrinsics.append(K); poses.append(pose)
        metadata.append({'name': name, 'source_image': rec['image'], 'source_camera': rec['camera'],
                         'perspective': perspective, 'K_opencv': K.tolist(), 'pad_top_left': [top, left],
                         'resized_width_height': [nw, nh], 'valid_fraction': float(mask.mean()),
                         'T_source_virtual': transform.tolist(), 'T_world_camera': None if pose is None else pose.tolist(),
                         **angles})
    return {'images': np.stack(images), 'rays': np.stack(rays), 'valid': np.stack(masks),
            'K': np.stack(intrinsics), 'poses': poses, 'views': metadata}
