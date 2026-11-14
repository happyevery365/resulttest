#!/usr/bin/env python3
"""Extract ONE original FIORD fisheye + calibration via bounded HTTP ZIP ranges.

Never downloads the 4.6 GB archive: refuses a server ignoring Range, limits each
read and total bytes, and only extracts explicitly selected small members.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import urllib.request
import zipfile

URL = 'https://zenodo.org/records/15181976/files/MeetingRoom_In.zip?download=1'


class RangeFile(io.RawIOBase):
    def __init__(self, url, budget=32 * 1024**2):
        self.url, self.budget, self.received, self.pos = url, budget, 0, 0
        self.log = []
        self.tail, self.tail_start, self.size = self.fetch('bytes=-131072', 131072)

    def fetch(self, value, limit):
        if self.received + limit > self.budget:
            raise ValueError('Download budget exceeded; refusing full archive')
        req = urllib.request.Request(self.url, headers={'Range': value, 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', cr)
            if r.status != 206 or not match:
                raise ValueError('Server did not honor Range; refusing to download archive')
            start, end, size = map(int, match.groups())
            if end - start + 1 > limit:
                raise ValueError('Unexpected Content-Range size')
            content = r.read(limit + 1)
        if len(content) != end - start + 1 or len(content) > limit:
            raise ValueError('Truncated or oversized range')
        self.received += len(content)
        self.log.append(dict(range=cr, bytes=len(content)))
        return content, start, size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError('Negative seek')
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b''
        if n > 16 * 1024**2:
            raise ValueError('Oversized read refused')
        if self.pos >= self.tail_start:
            result = self.tail[self.pos - self.tail_start:self.pos - self.tail_start + n]
        else:
            result, start, size = self.fetch(f'bytes={self.pos}-{self.pos+n-1}', n)
            if start != self.pos or size != self.size:
                raise ValueError('Server returned inconsistent archive/range')
        self.pos += len(result)
        return result


def camera_records(blob):
    stream = io.BytesIO(blob)
    count, = struct.unpack('<Q', stream.read(8))
    cameras = {}
    for _ in range(count):
        camera_id, model, w, h = struct.unpack('<iiQQ', stream.read(24))
        if model != 5:
            raise ValueError(f'Expected COLMAP OPENCV_FISHEYE model id 5, got {model}')
        params = struct.unpack('<8d', stream.read(64))
        cameras[camera_id] = dict(model='OPENCV_FISHEYE', width=w, height=h, params=list(params))
    return cameras


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fov', type=float, default=200., help='Conservative total lens FOV; verify for other lenses')
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Use a new/empty output directory')
    a.out.mkdir(parents=True, exist_ok=True)
    remote = RangeFile(URL)
    records = []
    with zipfile.ZipFile(remote) as z:
        names = z.namelist()
        camera_name = next(n for n in names if n.endswith('/model/cameras.bin'))
        image_model = camera_name.replace('cameras.bin', 'images.bin')
        if z.getinfo(camera_name).file_size > 1024**2:
            raise ValueError('Unexpected camera file size')
        cameras_blob = z.read(camera_name)
        cameras = camera_records(cameras_blob)
        # Read only the FIRST image header, not all 89 MB of image features.
        # It supplies the authoritative image filename -> camera ID association.
        with z.open(image_model) as f:
            count, = struct.unpack('<Q', f.read(8))
            if count < 1:
                raise ValueError('Empty COLMAP image model')
            header = struct.unpack('<idddddddi', f.read(64))
            image_id, camera_id = header[0], header[-1]
            name_bytes = bytearray()
            for _ in range(1024):
                b = f.read(1)
                if b == b'\x00':
                    break
                if not b:
                    raise ValueError('Truncated COLMAP image name')
                name_bytes.extend(b)
            else:
                raise ValueError('COLMAP filename too long')
            image_relative = name_bytes.decode('utf-8')
        choices = [n for n in names if '/fisheyeimages/' in n and n.endswith('/' + image_relative)]
        if len(choices) != 1:
            raise ValueError(f'Cannot uniquely identify raw image for {image_relative}: {choices}')
        chosen = choices[0]
        info = z.getinfo(chosen)
        if info.file_size > 16 * 1024**2 or info.compress_size > 16 * 1024**2:
            raise ValueError('Selected image exceeds small-sample budget')
        print(f'Fetching only {chosen} ({info.file_size} bytes), camera_id={camera_id}', flush=True)
        content = z.read(chosen)  # zipfile verifies member CRC32
        filename = Path(chosen).name
        (a.out / filename).write_bytes(content)
        (a.out / 'cameras.bin').write_bytes(cameras_blob)
        cal = cameras[camera_id]
        cal['fov_deg'] = a.fov
        dump = lambda name, obj: (a.out / name).write_text(json.dumps(obj, indent=2) + '\n')
        dump('camera.json', dict(camera=cal, camera_id=camera_id,
             note='Original COLMAP intrinsics; FOV is a conservative processing support limit, not a measured lens specification.'))
        for name, data in [(chosen, content), (camera_name, cameras_blob)]:
            records.append(dict(archive_member=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        dump('download_manifest.json', dict(dataset='FIORD / MeetingRoom_In', url=URL,
             original_archive_bytes=remote.size, transferred_bytes=remote.received,
             archive_downloaded=False, image_id=image_id, image_name=image_relative,
             camera_id=camera_id, camera_association='First COLMAP images.bin header',
             files=records, requests=remote.log,
             license='CC BY 4.0 (FIORD attribution; see Any3R FIORD SOURCE.md)',
             fov_note='200 degrees default: conservative processing support, not an estimated intrinsic.'))
    (a.out / 'SOURCE.txt').write_text('FIORD: Gunes et al., SCIA 2025.\nhttps://zenodo.org/records/15181976\nOriginal fisheye image and COLMAP calibration; not resized.\n')
    print(f'DONE: {filename}; total transferred {remote.received} bytes. No full archive downloaded.', flush=True)


if __name__ == '__main__':
    main()
