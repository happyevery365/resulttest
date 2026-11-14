#!/usr/bin/env python3
"""Download only three small, calibrated FIORD examples (under 1 MiB total)."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = 'c6c57bdda7b2526bdf6be28cc3b71d5c56a1eb15'
BASE = f'https://raw.githubusercontent.com/Jingong-Chen/Any3R/{REVISION}/examples/fiord_meetingroom/'
FILES = ['SOURCE.md', 'cameras.json'] + [f'images/fisheye_{i:02d}.jpg' for i in (0, 4, 8)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        dst = a.out / name
        if dst.exists():
            raise FileExistsError(f'Refusing to overwrite {dst}; use a new directory')
        url = BASE + name
        remaining = 1024 * 1024 - sum(r['bytes'] for r in records)
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read(remaining + 1)
        if len(data) > remaining:
            raise ValueError(f'Total download exceeds the 1 MiB budget at {name}')
        if name.endswith('.jpg') and not data.startswith(b'\xff\xd8'):
            raise ValueError(f'Not a JPEG: {url}')
        if name.endswith('.json'):
            json.loads(data)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        records.append(dict(file=name, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        print(f'{name}: {len(data)} bytes', flush=True)
    manifest = dict(dataset='FIORD / MeetingRoom_In', license='CC BY 4.0',
                    original_dataset='https://zenodo.org/records/15181976',
                    distribution='Resampled examples distributed by Any3R; not full-resolution originals.',
                    revision=REVISION, files=records, total_bytes=sum(r['bytes'] for r in records))
    (a.out / 'download_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Total: {manifest["total_bytes"]} bytes. No weights or full dataset downloaded.')


if __name__ == '__main__':
    main()
