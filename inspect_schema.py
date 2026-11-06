#!/usr/bin/env python3
"""Print YAML keys/types/shapes, never calibration values or image contents."""
import argparse
import json
import re
from pathlib import Path
import yaml


def read_yaml(path):
    """Read ordinary/OpenCV YAML without importing model or image dependencies."""
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Calibration exceeds 2 MiB')
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:opencv-matrix',
                           lambda loader, node: loader.construct_mapping(node, deep=True))
    text = re.sub(r'^%YAML:1\.0\s*\n', '', path.read_text(encoding='utf-8-sig'))
    obj = yaml.load(text, Loader=Loader)
    if not isinstance(obj, dict):
        raise ValueError('Expected YAML mapping')
    return obj


def shape(value, prefix='', depth=0):
    if depth > 15:
        return [{'field': prefix, 'type': 'depth_limit'}]
    if isinstance(value, dict):
        rows = [{'field': prefix or '<root>', 'type': 'mapping', 'key_count': len(value)}]
        for key, child in value.items():
            rows.extend(shape(child, f'{prefix}.{key}'.lstrip('.'), depth + 1))
        return rows
    if isinstance(value, list):
        row = {'field': prefix, 'type': 'list', 'length': len(value)}
        if value:
            row['first_element_type'] = type(value[0]).__name__
        return [row] + (shape(value[0], prefix + '[0]', depth + 1) if value and isinstance(value[0], (dict, list)) else [])
    return [{'field': prefix, 'type': type(value).__name__}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--case', type=Path)
    g.add_argument('--root', type=Path, help='Choose the first case under this dataset root')
    p.add_argument('--out', type=Path, default=Path('schema_report.json'))
    p.add_argument('--include-model-names', action='store_true', help='Optionally include only explicit projection/distortion name values; numeric parameters remain hidden')
    a = p.parse_args()
    case = a.case
    if case is None:
        case = next((x for x in sorted(a.root.iterdir()) if (x / 'conf').is_dir() and (x / 'picture').is_dir()), None)
    if case is None:
        raise ValueError('No case with conf and picture found')
    if a.out.exists():
        raise ValueError('Output exists; select a new --out')
    if a.out.resolve().is_relative_to(case.resolve()):
        raise ValueError('Output must be outside source case')
    report = {'note': 'Schema only. No calibration values. Case/image names and paths are omitted.', 'cameras': []}
    for f in sorted((case / 'conf').glob('cam_*.yaml')):
        row = {'camera': f.stem}
        try:
            obj = read_yaml(f)
            row['fields'] = shape(obj)
            if a.include_model_names:
                def names(obj, path='', depth=0):
                    result = {}
                    if depth <= 15 and isinstance(obj, dict):
                        for key, val in obj.items():
                            name = f'{path}.{key}'.lstrip('.')
                            if key in {'model', 'model_type', 'camera_model', 'projection_model', 'distortion_model'} and isinstance(val, (str, int)):
                                result[name] = val
                            elif isinstance(val, dict):
                                result.update(names(val, name, depth + 1))
                    return result
                row['model_names'] = names(obj)
        except Exception as e:
            # YAML parser messages may contain source lines, so never echo them here.
            row['error_type'] = type(e).__name__
        report['cameras'].append(row)
    if not report['cameras']:
        raise ValueError('No conf/cam_*.yaml files found')
    text = json.dumps(report, ensure_ascii=False, indent=2)
    with a.out.open('x', encoding='utf-8') as output:
        output.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
