#!/usr/bin/env python3
"""Sequential runner: each model uses its existing Python environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value')
    p.add_argument('--runs', default='pi3x:raw,pi3x:remap,omnivggt:raw,omnivggt:remap,any3r:raw,any3r:remap',
                   help='Comma-separated model:mode; defaults to all three models in both raw and remap modes')
    a, extra = p.parse_known_args()
    if any(x.split('=')[0] in {'--out', '--model', '--mode', '--config'} for x in extra):
        p.error('Output/model/mode/config are controlled by run_all.py')
    cfg = json.loads(a.config.read_text())
    runs = [tuple(x.split(':')) for x in a.runs.split(',')]
    if len(set(runs)) != len(runs) or any(len(x) != 2 or x[0] not in {'pi3x', 'omnivggt', 'any3r'} or x[1] not in {'raw', 'remap'} for x in runs):
        p.error('Invalid/duplicate --runs')
    for model, mode in runs:
        if not Path(cfg[model]['python']).is_file():
            p.error(f'Missing interpreter: {cfg[model]["python"]}')
    if a.out_root.exists() and any(a.out_root.iterdir()):
        p.error('--out-root must be new or empty')
    a.out_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
    results = []
    for model, mode in runs:
        tag = f'{model}_{mode}'
        command = [cfg[model]['python'], str(Path(__file__).with_name('run.py')),
                   '--config', str(a.config.resolve()), '--model', model, '--mode', mode,
                   '--out', str(a.out_root / tag), *extra]
        with (a.out_root / (tag + '.log')).open('w') as log:
            print(f'RUN {tag}', flush=True)
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
        results.append({'run': tag, 'returncode': code, 'output': str(a.out_root / tag)})
        (a.out_root / 'batch.json').write_text(json.dumps(results, indent=2))
    if any(r['returncode'] for r in results):
        sys.exit('Some runs failed; inspect batch.json and individual logs')


if __name__ == '__main__':
    main()
