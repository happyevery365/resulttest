#!/usr/bin/env python3
"""Fetch a few real pinhole/fisheye pairs from official KITTI-360 ZIPs via Range."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile
import zlib

from fetch_fiord_original import RangeFile

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/'
CALIB = BASE + '384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip'


class BufferedRange(RangeFile):
    def __init__(self, url, budget):
        self.cache, self.cache_start = b'', 0
        super().__init__(url, budget)

    def fetch(self, value, limit):
        for attempt in range(5):
            try:
                return super().fetch(value, limit)
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(1)

    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else min(n, self.size-self.pos)
        if n <= 0 or self.pos >= self.tail_start:
            return super().read(n)
        if n > 16*1024**2:
            raise ValueError('Oversized archive read refused')
        if not (self.cache_start <= self.pos and self.pos+n <= self.cache_start+len(self.cache)):
            count = min(max(n, 65536), self.size-self.pos)
            self.cache, self.cache_start, total = self.fetch(f'bytes={self.pos}-{self.pos+count-1}', count)
            if self.cache_start != self.pos or total != self.size:
                raise ValueError('Wrong HTTP range')
        offset = self.pos-self.cache_start
        result = self.cache[offset:offset+n]
        self.pos += len(result)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sequence', default='2013_05_28_drive_0000_sync')
    p.add_argument('--frames', type=int, nargs='+', default=[100,1000,2000])
    p.add_argument('--lidar', action='store_true', help='Also fetch matching sparse scans for diagnostic evaluation')
    p.add_argument('--resume', action='store_true', help='Reuse existing members only after archive CRC verification')
    p.add_argument('--direct', action='store_true', help='Connect directly without environment HTTP proxies')
    a = p.parse_args()
    if a.direct:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    if len(a.frames)>5 or len(set(a.frames))!=len(a.frames) or any(x<0 for x in a.frames):
        p.error('Select 1 to 5 unique nonnegative frame indices')
    if a.out.exists() and any(a.out.iterdir()) and not a.resume:
        p.error('Use a new/empty destination')
    a.out.mkdir(parents=True, exist_ok=True)
    records, archives = [], []
    budget, received = 80*1024**2, 0

    def save(relative, data, url, member):
        path = a.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records.append(dict(file=str(relative), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), url=url, archive_member=member))
        print(relative, len(data), flush=True)

    with urllib.request.urlopen(CALIB, timeout=40) as r:
        blob = r.read(100001)
    if len(blob)>100000:
        raise ValueError('Unexpected calibration size')
    received += len(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in ('calib_cam_to_pose.txt','calib_cam_to_velo.txt','perspective.txt','image_02.yaml'):
            save(Path('calibration')/name,z.read('calibration/'+name),CALIB,'calibration/'+name)

    jobs=[]
    for cam, sub in [('image_00','data_rect'),('image_02','data_rgb')]:
        jobs.append((BASE+f'data_2d_raw/{a.sequence}_{cam}.zip',
                     [(f'{a.sequence}/{cam}/{sub}/{i:010d}.png',Path('frames')/f'{i:010d}'/(cam+'.png')) for i in a.frames]))
    for typ,cam in [('perspective','image_00'),('fisheye','image_02')]:
        jobs.append((BASE+f'data_2d_raw/data_timestamps_{typ}.zip',
                     [(f'{a.sequence}/{cam}/timestamps.txt',Path('timestamps')/(cam+'.txt'))]))
    if a.lidar:
        jobs.append((BASE+f'data_3d_raw/{a.sequence}_velodyne.zip',
                     [(f'{a.sequence}/velodyne_points/data/{i:010d}.bin',Path('frames')/f'{i:010d}'/'velodyne.bin') for i in a.frames]))
    for url, wanted in jobs:
        remote=BufferedRange(url,budget=budget-received)
        with zipfile.ZipFile(remote) as z:
            names=z.namelist()
            for suffix, destination in wanted:
                matches=[n for n in names if n==suffix or n.endswith('/'+suffix)]
                if len(matches)!=1:
                    raise ValueError(f'Cannot uniquely find {suffix}: {matches}')
                name=matches[0]; info=z.getinfo(name)
                if max(info.file_size,info.compress_size)>8*1024**2:
                    raise ValueError('Selected member exceeds sample budget')
                existing=a.out/destination
                if a.resume and existing.is_file():
                    content=existing.read_bytes()
                    if len(content)!=info.file_size or zlib.crc32(content)!=info.CRC:
                        raise ValueError(f'Existing member fails archive CRC: {existing}')
                    save(destination,content,url,name)
                else:
                    save(destination,z.read(name),url,name)
        received+=remote.received
        archives.append(dict(url=url,archive_bytes=remote.size,transferred_bytes=remote.received,requests=remote.log))
    manifest=dict(dataset='KITTI-360',sequence=a.sequence,frames=a.frames,cameras=['image_00 rectified perspective','image_02 native MEI fisheye'],
                  transferred_bytes_this_invocation=received,resumed=a.resume,full_image_archives_downloaded=False,files=records,archives=archives,
                  source='https://www.cvlibs.net/datasets/kitti-360/',
                  license='CC BY-NC-SA 3.0; Liao, Xie and Geiger, KITTI-360, PAMI 2022; research examples only')
    (a.out/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE, transferred bytes:',received,flush=True)


if __name__=='__main__':
    main()
