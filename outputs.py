"""Point exports, offline viewer and explicitly non-ground-truth diagnostics."""
import json
from pathlib import Path
import numpy as np


def save_ply(path, xyz, rgb):
    rec = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, k in enumerate(('x', 'y', 'z')):
        rec[k] = xyz[:, i]
    for i, k in enumerate(('red', 'green', 'blue')):
        rec[k] = rgb[:, i]
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as f:
        f.write(header.encode()); f.write(rec.tobytes())


def viewer(path, xyz, rgb, limit=40000):
    rng = np.random.default_rng(0)
    ids = rng.choice(len(xyz), min(limit, len(xyz)), replace=False)
    p = xyz[ids].astype(float)
    center = np.median(p, axis=0)
    scale = max(float(np.percentile(np.linalg.norm(p - center, axis=1), 90)), 1e-8)
    data = np.concatenate([np.round((p - center) / scale, 5), rgb[ids]], axis=1).tolist()
    html = '''<!doctype html><meta charset="utf-8"><title>Point cloud</title>
<style>body{margin:0;background:#111;color:#eee;font:15px sans-serif}#help{position:absolute;padding:12px;background:#222b}canvas{display:block}</style>
<div id="help">点云预览 · 拖动旋转 · 滚轮缩放 · 双击复位<br>显示最多 40,000 点；完整点云见 points.ply。坐标仅为显示归一化。</div><canvas id="c"></canvas>
<script>const pts=__DATA__;const c=document.getElementById('c'),ctx=c.getContext('2d');let yaw=0,pitch=0,zoom=1,drag=false,lx=0,ly=0;
function draw(){c.width=innerWidth*devicePixelRatio;c.height=innerHeight*devicePixelRatio;c.style.width=innerWidth+'px';c.style.height=innerHeight+'px';ctx.fillStyle='#111';ctx.fillRect(0,0,c.width,c.height);let s=Math.min(c.width,c.height)*.38*zoom,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
let q=pts.map(p=>{let x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2];return [x,cp*p[1]-sp*z,sp*p[1]+cp*z,p]});q.sort((a,b)=>b[2]-a[2]);for(let [x,y,z,p] of q){ctx.fillStyle=`rgb(${p[3]},${p[4]},${p[5]})`;ctx.fillRect(c.width/2+x*s,c.height/2+y*s,2*devicePixelRatio,2*devicePixelRatio)}}
c.onpointerdown=e=>{drag=true;lx=e.clientX;ly=e.clientY;c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=false;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-lx)*.007;pitch+=(e.clientY-ly)*.007;lx=e.clientX;ly=e.clientY;draw()}};c.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.02,Math.min(100,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});c.ondblclick=()=>{yaw=pitch=0;zoom=1;draw()};onresize=draw;draw();</script>'''
    Path(path).write_text(html.replace('__DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf-8')


def export(out, prepared, points, confidence, poses, drop_percentile):
    valid = prepared['valid']
    finite = np.isfinite(points).all(-1) & np.isfinite(confidence)
    # No RGB brightness test: genuinely dark surfaces must not be deleted as padding.
    base = valid & finite
    if not base.any() or not np.isfinite(poses).all():
        raise ValueError('No finite valid points, or invalid predicted poses')
    thresholds = []
    keep = np.zeros_like(base)
    for i in range(len(points)):
        cutoff = float(np.percentile(confidence[i][base[i]], drop_percentile)) if base[i].any() else None
        thresholds.append(cutoff)
        if cutoff is not None:
            keep[i] = base[i] & (confidence[i] >= cutoff)
    if not keep.any():
        raise ValueError('Point filter removed all points')
    delta = points - poses[:, None, None, :3, 3]
    local = np.einsum('nhwi,nij->nhwj', delta, poses[:, :3, :3])
    radial = np.linalg.norm(local, axis=-1)
    unit = local / np.maximum(radial[..., None], 1e-12)
    angle = np.rad2deg(np.arccos(np.clip(np.sum(unit * prepared['rays'], axis=-1), -1, 1)))
    diag = []
    for i, view in enumerate(prepared['views']):
        mask = keep[i] & np.isfinite(radial[i]) & (radial[i] > 0)
        diag.append({'name': view['name'], 'valid_pixels': int(valid[i].sum()),
                     'finite_fraction_of_valid': float(base[i].sum() / max(1, valid[i].sum())),
                     'exported_points': int(keep[i].sum()), 'confidence_cutoff': thresholds[i],
                     'positive_radial_fraction': float((radial[i][base[i]] > 0).mean()) if base[i].any() else None,
                     'radial_depth_p05_p50_p95': np.percentile(radial[i][mask], [5, 50, 95]).tolist() if mask.any() else None,
                     'ray_angle_median_deg': float(np.median(angle[i][mask])) if mask.any() else None})
    xyz, rgb = points[keep], prepared['images'][keep]
    save_ply(out / 'points.ply', xyz, rgb)
    np.save(out / 'points.npy', xyz)
    np.savez_compressed(out / 'predictions.npz', points=points, confidence=confidence, keep=keep,
                        valid=valid, camera_to_world=poses, radial_depth=radial,
                        images=prepared['images'], names=np.array([v['name'] for v in prepared['views']]))
    viewer(out / 'viewer.html', xyz, rgb)
    return {'exported_points': len(xyz), 'views': diag,
            'filter': f'Geometric validity and finite points/confidence; discard bottom {drop_percentile}% confidence per view',
            'accuracy_metrics_available': False,
            'note': 'No ground truth. These are coverage/health/geometric diagnostics, not reconstruction accuracy. Ray agreement is structurally enforced by Any3R. Confidence values are not comparable across models. Depth scale is unverified.'}
