"""Read-only case selection and deliberately conservative calibration parsing."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
DEFAULT_REGEX = r'(?P<id>\d+)$'


def select(case=None, image=None, cameras=None, frame_id=None, frame_regex=DEFAULT_REGEX):
    if image:
        image = Path(image).expanduser().absolute()
        if not image.is_file() or not re.fullmatch(r'cam_\d+', image.parent.name):
            raise ValueError('--image must be an existing picture/cam_N/image file')
        inferred = image.parent.parent.parent
        if image.parent.parent.name != 'picture' or (case and Path(case).resolve() != inferred.resolve()):
            raise ValueError('Image must belong to the selected case/picture/cam_N directory')
        if cameras or frame_id is not None:
            raise ValueError('--image cannot be combined with --cameras or --frame-id')
        return inferred, [(image.parent.name, image)], {'selection': 'single_image', 'synchronization_verified': False}
    case = Path(case).expanduser().absolute()
    folders = sorted((p for p in (case / 'picture').iterdir() if p.is_dir() and re.fullmatch(r'cam_\d+', p.name)),
                     key=lambda p: int(p.name[4:]))
    available = [p.name for p in folders]
    if cameras:
        requested = [c if c.startswith('cam_') else 'cam_' + c for c in cameras.split(',')]
        if len(set(requested)) != len(requested) or set(requested) - set(available):
            raise ValueError(f'Duplicate or missing requested cameras; available={available}')
        folders = [case / 'picture' / c for c in requested]
    if not folders:
        raise ValueError('No picture/cam_N directories found')
    pattern = re.compile(frame_regex)
    if 'id' not in pattern.groupindex:
        raise ValueError('--frame-regex must contain a named group (?P<id>...) of digits')
    indices, counts = [], {}
    for folder in folders:
        index, unmatched = {}, []
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        for f in files:
            m = pattern.search(f.stem)
            if not m or not m['id'].isdigit():
                unmatched.append(f.name)
                continue
            key = str(int(m['id']))
            if key in index:
                raise ValueError(f'Ambiguous frame {key} in {folder.name}: {index[key].name}, {f.name}; use --frame-regex')
            index[key] = f
        if unmatched:
            raise ValueError(f'{folder.name}: unmatched filenames {unmatched[:5]}; specify --frame-regex explicitly')
        indices.append(index)
        counts[folder.name] = len(index)
    common = set.intersection(*(set(x) for x in indices))
    if not common:
        raise ValueError(f'No common numeric identifier across selected cameras. Counts: {counts}. Use --cameras to select a subset.')
    chosen = str(int(frame_id)) if frame_id is not None else min(common, key=int)
    if chosen not in common:
        raise ValueError(f'Frame {chosen} is not common to every selected camera')
    return case, [(p.name, idx[chosen]) for p, idx in zip(folders, indices)], {
        'selection': 'common_numeric_identifier', 'frame_id': chosen, 'frame_regex': frame_regex,
        'common_count': len(common), 'common_first_20': sorted(common, key=int)[:20], 'image_counts': counts,
        'missing_expected_camera_folders': [f'cam_{i}' for i in range(13) if f'cam_{i}' not in available],
        'synchronization_verified': False,
        'synchronization_note': 'Equal filename numbers are not proof of equal exposure timestamps.'}


class CalibrationError(ValueError):
    pass


def read_yaml(path):
    if path.stat().st_size > 2 * 1024 * 1024:
        raise CalibrationError('Calibration exceeds 2 MiB')
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:opencv-matrix', lambda loader, node: loader.construct_mapping(node, deep=True))
    text = re.sub(r'^%YAML:1\.0\s*\n', '', path.read_text(encoding='utf-8-sig'))
    obj = yaml.load(text, Loader=Loader)
    if not isinstance(obj, dict):
        raise CalibrationError('Expected YAML mapping')
    return obj


def vector(value):
    if isinstance(value, dict):
        value = value.get('data', value)
    return np.asarray(value, dtype=np.float64)


def choose(obj, keys, default=None):
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def normalize(obj, cam_id, image_size):
    # Support a single-camera mapping, a Kalibr camN wrapper, or a camera wrapper.
    for wrapper in (cam_id, cam_id.replace('_', ''), 'camera'):
        if wrapper in obj and isinstance(obj[wrapper], dict):
            obj = obj[wrapper]
            break
    model = str(choose(obj, ['model', 'model_type', 'camera_model', 'projection_model'], '')).lower().replace('-', '_')
    dist_model = str(obj.get('distortion_model', '')).lower().replace('-', '_')
    known_camera_models = {'', 'pinhole', 'perspective', 'opencv', 'full_opencv', 'brown_conrady',
                           'kb4', 'kannala_brandt', 'kannala_brandt4', 'opencv_fisheye', 'equidistant',
                           'fisheye', 'mei', 'omni', 'double_sphere', 'ds', 'eucm', 'equidistant_ideal'}
    if model not in known_camera_models:
        raise CalibrationError(f'Unknown model {model!r}; numeric/vendor enums need an explicit mapping')
    if dist_model not in {'', 'none', 'plumb_bob', 'radtan', 'rational_polynomial', 'brown_conrady', 'equidistant', 'fisheye', 'kb4'}:
        raise CalibrationError(f'Unknown distortion_model {dist_model!r}')
    if model in {'mei', 'omni', 'double_sphere', 'ds', 'eucm'}:
        raise CalibrationError(f'{model}: parameter order/distortion composition requires a verified adapter; not interpreted as pinhole')
    fish_models = {'kb4', 'kannala_brandt', 'kannala_brandt4', 'opencv_fisheye', 'equidistant'}
    fish_dist = dist_model in {'equidistant', 'fisheye', 'kb4'}
    if model in fish_models and dist_model in {'plumb_bob', 'radtan', 'rational_polynomial', 'brown_conrady'}:
        raise CalibrationError('Conflicting fisheye projection and perspective distortion declarations')
    if model in fish_models or fish_dist:
        kind = 'kb4'
    elif model == 'equidistant_ideal':
        kind = 'equidistant'
    elif model in {'pinhole', 'perspective', 'opencv', 'full_opencv', 'brown_conrady'} or dist_model in {'plumb_bob', 'radtan', 'rational_polynomial', 'brown_conrady'}:
        kind = 'brown_conrady'
    else:
        raise CalibrationError(f'Projection ambiguous: model={model!r}, distortion_model={dist_model!r}; K/D length alone is not evidence')
    resolution = choose(obj, ['resolution', 'image_size'])
    if resolution is not None:
        width, height = map(int, resolution)
    else:
        width = choose(obj, ['image_width', 'width'])
        height = choose(obj, ['image_height', 'height'])
        if width is None or height is None:
            raise CalibrationError('Calibration width/height missing; cannot verify K resolution')
        width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise CalibrationError('Invalid image dimensions')
    K = choose(obj, ['K', 'camera_matrix', 'intrinsic_matrix'])
    if K is not None:
        K = vector(K).reshape(3, 3)
    else:
        intr = choose(obj, ['intrinsics', 'projection_parameters'])
        if isinstance(intr, dict):
            # camodocal Kannala-Brandt uses mu,mv,u0,v0 and k2,k3,k4,k5.
            if all(k in intr for k in ('mu', 'mv', 'u0', 'v0')):
                fx, fy, cx, cy = [intr[k] for k in ('mu', 'mv', 'u0', 'v0')]
            else:
                fx, fy, cx, cy = [intr[k] for k in ('fx', 'fy', 'cx', 'cy')]
        elif intr is not None:
            intr = vector(intr).ravel()
            if len(intr) != 4:
                raise CalibrationError('Only verified [fx,fy,cx,cy] intrinsic vectors are accepted')
            fx, fy, cx, cy = intr
        else:
            fx, fy, cx, cy = [obj[k] for k in ('fx', 'fy', 'cx', 'cy')]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    convention = obj.get('pixel_convention', 'opencv')
    if convention == 'colmap':
        K[:2, 2] -= 0.5
    elif convention != 'opencv':
        raise CalibrationError('pixel_convention must be opencv or colmap')
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9:
        raise CalibrationError('Invalid K or unsupported skew')
    D = choose(obj, ['D', 'distortion_coefficients', 'distortion_coeffs', 'distortion_parameters'])
    pp = obj.get('projection_parameters', {})
    if D is None and model == 'kannala_brandt' and isinstance(pp, dict) and all(k in pp for k in ('mu', 'mv', 'k2', 'k3', 'k4', 'k5')):
        D = [pp[k] for k in ('k2', 'k3', 'k4', 'k5')]
    if isinstance(D, dict) and 'data' not in D:
        order = ['k1', 'k2', 'k3', 'k4'] if kind == 'kb4' else ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
        if set(D) - set(order):
            raise CalibrationError('Unsupported named distortion coefficients')
        if kind == 'kb4' and not all(k in D for k in order):
            raise CalibrationError('KB4 requires four explicit coefficients')
        D = [D.get(k, 0) for k in order]
    if D is None:
        if kind == 'kb4' or (kind == 'brown_conrady' and dist_model not in {'', 'none'}):
            raise CalibrationError('Declared distortion model but missing coefficients')
        D = []
    D = vector(D).ravel()
    if not np.isfinite(D).all() or (kind == 'kb4' and len(D) != 4) or (kind == 'brown_conrady' and len(D) not in {0, 4, 5, 8}):
        raise CalibrationError('Invalid or unsupported distortion coefficient count')
    if kind == 'equidistant' and np.any(D):
        raise CalibrationError('equidistant_ideal cannot have nonzero distortion coefficients')
    if dist_model == 'none' and np.any(D):
        raise CalibrationError('distortion_model=none conflicts with nonzero coefficients')
    if kind == 'brown_conrady' and not np.any(D):
        kind = 'pinhole'
    if any(obj.get(key) is True for key in ('is_rectified', 'rectified', 'undistorted')) and kind != 'pinhole':
        raise CalibrationError('Image declared rectified/undistorted but calibration is distorted; provide the rectified projection K explicitly')
    if (width, height) != tuple(image_size):
        raise CalibrationError(f'Calibration {width}x{height} differs from image {image_size}; supply a verified scaled/cropped override')
    pose = obj.get('T_world_camera')  # Only this explicit convention; never guess vendor extrinsics.
    if pose is not None:
        pose = vector(pose).reshape(4, 4)
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4):
            raise CalibrationError('Invalid T_world_camera rigid transform')
        if obj.get('translation_unit') != 'm' or obj.get('camera_axes') != 'opencv':
            raise CalibrationError('T_world_camera requires translation_unit: m and camera_axes: opencv')
        pose = pose.tolist()
    fov = obj.get('valid_fov_deg')
    if fov is not None and not (0 < float(fov) <= 360):
        raise CalibrationError('Invalid valid_fov_deg')
    return {'kind': kind, 'fisheye': kind in {'kb4', 'equidistant'},
            'reason': f'model={model!r}; distortion_model={dist_model!r}',
            'width': width, 'height': height, 'K': K.tolist(), 'D': D.tolist(),
            'pixel_convention': 'opencv', 'source_pixel_convention': convention,
            'pixel_convention_assumption': 'Default OpenCV integer pixel centers unless explicitly overridden',
            'valid_fov_deg': fov, 'T_world_camera': pose,
            'warnings': ['No lens validity FOV provided; ray invertibility is checked but physical lens boundary is unknown'] if kind != 'pinhole' and fov is None else []}


def inspect(case, selected, selection, overrides=None):
    overrides = overrides or {}
    records = []
    for cam_id, image in selected:
        path = case / 'conf' / (cam_id + '.yaml')
        record = {'camera': cam_id, 'image': str(image), 'config': str(path)}
        try:
            with Image.open(image) as im:
                size = im.size
                record.update(image_size=list(size), image_mode=im.mode, exif_orientation=im.getexif().get(274, 1))
            obj = overrides[cam_id] if cam_id in overrides else read_yaml(path)
            record['top_level_keys'] = sorted(map(str, obj))
            record['override_used'] = cam_id in overrides
            if path.exists():
                record['config_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if record['exif_orientation'] != 1:
                raise CalibrationError('Nontrivial EXIF rotation: verify calibration coordinates before running')
            record['calibration'] = normalize(obj, cam_id, size)
            record['status'] = 'ok'
        except (ValueError, KeyError, TypeError, OSError, yaml.YAMLError) as e:
            record['status'] = 'unsupported'
            record['error'] = str(e)
        records.append(record)
    return {'case': str(case), 'selection': selection, 'cameras': records,
            'ready': all(r['status'] == 'ok' for r in records)}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
