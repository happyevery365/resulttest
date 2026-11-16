#!/usr/bin/env python3
"""Build a self-contained local KITTI-360 mixed-camera review page (no CDN)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def interactive(path, xyz, rgb, source):
    rng=np.random.default_rng(0)
    ids=rng.choice(len(xyz),min(70000,len(xyz)),replace=False)
    records=np.column_stack([np.round(xyz[ids],4),rgb[ids],source[ids]]).tolist()
    html='''<!doctype html><meta charset="utf-8"><title>Mixed-camera cloud</title>
<style>body{margin:0;background:#121820;color:white;font:14px system-ui}#ui{position:absolute;background:#202938ee;padding:10px;line-height:1.8}canvas{display:block}input{width:55px}</style>
<div id="ui">拖动旋转 · 滚轮缩放 · 双击复位<br><label><input id="pin" type="checkbox" checked>针孔</label> <label><input id="fish" type="checkbox" checked>鱼眼</label> <label><input id="colors" type="checkbox">按相机着色</label><br>显示半径 <input id="radius" type="number" value="80" min="5" max="500"> m（尺度估计）<br>最多 70,000 点；半径裁剪仅用于预览，完整 PLY 未裁剪。<br>相机色：针孔蓝 / 鱼眼橙；坐标轴 R=X / G=Y / B=Z</div><canvas id="c"></canvas>
<script>const pts=__DATA__,c=document.getElementById('c'),ctx=c.getContext('2d');let yaw=.38,pitch=-.2,zoom=1,drag=false,lx=0,ly=0;
const get=id=>document.getElementById(id);
function transform(p){let x=p[0],y=p[1],z=p[2]-12,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),xx=cy*x+sy*z,zz=-sy*x+cy*z;return [xx,cp*y-sp*zz,sp*y+cp*zz]}
function draw(){let d=devicePixelRatio;c.width=innerWidth*d;c.height=innerHeight*d;c.style.width=innerWidth+'px';c.style.height=innerHeight+'px';ctx.fillStyle='#121820';ctx.fillRect(0,0,c.width,c.height);let s=Math.min(c.width,c.height)*zoom/45,r=Number(get('radius').value)||80;
let q=pts.filter(p=>(p[6]===0?get('pin').checked:get('fish').checked)&&Math.hypot(p[0],p[1],p[2])<r).map(p=>[...transform(p),p]);q.sort((a,b)=>b[2]-a[2]);for(let [x,y,z,p] of q){let color=get('colors').checked?(p[6]===0?[70,170,255]:[255,160,60]):p.slice(3,6);ctx.fillStyle=`rgb(${color})`;ctx.fillRect(c.width/2+x*s,c.height/2+y*s,1.5*d,1.5*d)}
let o=transform([0,0,0]);[[3,0,0,'#f66'],[0,3,0,'#6f6'],[0,0,3,'#66f']].forEach(p=>{let v=transform(p);ctx.strokeStyle=p[3];ctx.lineWidth=2*d;ctx.beginPath();ctx.moveTo(c.width/2+o[0]*s,c.height/2+o[1]*s);ctx.lineTo(c.width/2+v[0]*s,c.height/2+v[1]*s);ctx.stroke()})}
c.onpointerdown=e=>{drag=true;lx=e.clientX;ly=e.clientY;c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=false;c.onpointermove=e=>{if(drag){yaw+=(e.clientX-lx)*.007;pitch+=(e.clientY-ly)*.007;lx=e.clientX;ly=e.clientY;draw()}};c.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.05,Math.min(50,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});c.ondblclick=()=>{yaw=.38;pitch=-.2;zoom=1;draw()};['pin','fish','colors','radius'].forEach(id=>get(id).oninput=draw);onresize=draw;draw();</script>'''
    path.write_text(html.replace('__DATA__',json.dumps(records,separators=(',',':'))))


def preview(xyz,rgb,source,size=(640,460)):
    keep=np.flatnonzero(np.linalg.norm(xyz,axis=1)<80)
    rng=np.random.default_rng(0); keep=rng.choice(keep,min(70000,len(keep)),replace=False)
    p=xyz[keep].copy(); p[:,2]-=12
    yaw,pitch=.38,-.2
    R=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]])
    Rx=np.array([[1,0,0],[0,np.cos(pitch),-np.sin(pitch)],[0,np.sin(pitch),np.cos(pitch)]])
    p=p@R.T@Rx.T; xy=p[:,:2]*(min(size)/45)+np.array(size)/2
    out=np.full((size[1],size[0],3),22,'u1')
    for j in np.argsort(p[:,2])[::-1]:
        x,y=np.rint(xy[j]).astype(int)
        if 0<=x<size[0] and 0<=y<size[1]: cv2.circle(out,(x,y),1,tuple(int(c) for c in rgb[keep[j]]),-1)
    return Image.fromarray(out)


def build(root):
    root=Path(root); records=[]; sections=[]
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    except OSError:
        font=ImageFont.load_default()
    frames=[100,1000,2000]
    overview=Image.new('RGB',(1920,520*len(frames)),'#161c24'); draw=ImageDraw.Draw(overview)
    for row,frame in enumerate(frames):
        run=root/'runs'/f'kitti360_{frame:010d}'; base=run.relative_to(root).as_posix()
        prep=json.loads((run/'prepared/preparation.json').read_text())
        cards=[]; table=[]
        for j,model in enumerate(['pi3x','omnivggt']):
            out=run/model/'fused'; met=json.loads((out/'metrics.json').read_text()); inf=json.loads((run/model/'inference/inference.json').read_text())
            data=np.load(out/'points.npz'); xyz,rgb,source=data['xyz'],data['rgb'],data['source_camera']
            assert len(xyz)==met['total_points'] and np.isfinite(xyz).all()
            for s,name in enumerate(['pinhole','fisheye']):
                uv=np.load(out/(name+'_point_uv.npy')); depth=np.load(out/(name+'_range.npy')); valid=np.load(out/(name+'_valid.npy'))
                flat=uv[:,1].astype('i8')*depth.shape[1]+uv[:,0]
                assert np.all(np.diff(flat)>0) and len(uv)==valid.sum()==(source==s).sum()
                assert np.isnan(depth[~valid]).all()
            interactive(out/'mixed_viewer.html',xyz,rgb,source)
            cloud=preview(xyz,rgb,source); cloud.save(out/'preview.jpg',quality=92)
            overview.paste(cloud,((j+1)*640,row*520+45))
            draw.text(((j+1)*640+12,row*520+12),f'{model} | frame {frame} | display <80m',font=font,fill='white')
            prefix=f'{base}/{model}/fused'
            d=met['lidar']['cameras']
            record=dict(frame=frame,model=model,depth_multiplier=inf['scale_alignment']['multiplier'],
                        lidar=d,fish_overlap_p50_p95=met['cameras']['fisheye']['overlap_relative_spread_p50_p95'],
                        inference_seconds=inf['inference_seconds'],peak_GiB=inf['peak_GiB'],points=met['total_points'])
            records.append(record)
            table.append(f'<tr><td>{model}</td><td>{d["pinhole"]["abs_rel"]:.2%}</td><td>{d["fisheye"]["abs_rel"]:.2%}</td><td>{d["pinhole"]["predicted_to_lidar_ratio_median"]:.3f} / {d["fisheye"]["predicted_to_lidar_ratio_median"]:.3f}</td><td>{record["depth_multiplier"]:.3f}</td></tr>')
            cards.append(f'''<article><h3>{model}</h3><iframe loading="lazy" src="{prefix}/mixed_viewer.html" title="{model} 点云"></iframe><p><a href="{prefix}/mixed_viewer.html" target="_blank">大窗口交互</a> · <a href="{prefix}/points.ply">完整 PLY</a> · <a href="{prefix}/metrics.json">诊断记录</a></p><div class="depth"><a href="{prefix}/pinhole_range.png"><img src="{prefix}/pinhole_range.png" alt="针孔深度"></a><a href="{prefix}/fisheye_range.png"><img src="{prefix}/fisheye_range.png" alt="鱼眼深度"></a></div><p>左：原针孔深度；右：原鱼眼径向深度。各图独立颜色范围，黑色为无结果，定量值见 NPY。<a href="{prefix}/pinhole_range.npy">针孔 NPY</a> · <a href="{prefix}/fisheye_range.npy">鱼眼 NPY</a></p></article>''')
        fish=Image.open(run/'prepared/fisheye_source.png'); fish.thumbnail((300,300))
        pin=Image.open(run/'prepared/pinhole_source.png'); pin.thumbnail((630,170))
        overview.paste(fish,((640-fish.width)//2,row*520+40)); overview.paste(pin,((640-pin.width)//2,row*520+350))
        draw.text((12,row*520+12),f'Frame {frame}: real fisheye + pinhole',font=font,fill='white')
        sections.append(f'''<section><h2>样例 {frame}</h2><p>鱼眼−针孔时间差 {prep['fisheye_minus_pinhole_ms']:.3f} ms · <a href="{base}/prepared/inputs.png">实际 5 张模型输入</a> · <a href="{base}/prepared/preparation.json">内外参与预处理记录</a></p><div class="sources"><a href="{base}/prepared/pinhole_source.png"><img src="{base}/prepared/pinhole_source.png" alt="真实针孔原图"></a><a href="{base}/prepared/fisheye_source.png"><img src="{base}/prepared/fisheye_source.png" alt="真实鱼眼原图"></a></div><table><tr><th>模型</th><th>针孔 AbsRel ↓</th><th>鱼眼 AbsRel ↓</th><th>预测/激光中值比（针孔/鱼眼，理想=1）</th><th>统一深度倍率</th></tr>{''.join(table)}</table><div class="grid">{''.join(cards)}</div></section>''')
    overview.save(root/'mixed_overview.jpg',quality=94)
    (root/'mixed_summary.json').write_text(json.dumps(records,indent=2)+'\n')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实鱼眼＋针孔联合重建</title>
<style>body{font:16px/1.7 system-ui;max-width:1450px;margin:30px auto;padding:0 20px;color:#e6edf5;background:#121820}a{color:#91c5ff}h1{font-size:29px}h2{margin-top:40px}section{border-top:1px solid #465266;margin-top:36px}.grid,.sources{display:grid;grid-template-columns:1fr 1fr;gap:16px}.sources img{width:100%;height:300px;object-fit:contain;background:#080c12}article{padding:12px;background:#1b2432;border-radius:12px}iframe{width:100%;height:520px;border:0}img{max-width:100%}.depth{display:flex;gap:10px;align-items:center}.depth a{width:50%}.depth img{width:100%}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border:1px solid #465266;padding:8px;text-align:left}.notice{background:#352c20;border-left:4px solid #e5ae51;padding:15px}@media(max-width:850px){.grid,.sources{grid-template-columns:1fr}table{font-size:12px}}</style>
<h1>真实鱼眼＋针孔：Pi3X / OmniVGGT 联合重建</h1>
<p>KITTI-360 序列 0000 的 100 / 1000 / 2000 帧。真实 image_00 整流针孔 1408×376，真实 image_02 MEI 鱼眼 1400×1400，标定基线 0.959 m。来自同一路线的三个位置，前向与侧向相机局部视野重叠；不是用同一张全景图模拟两台相机。<a href="https://www.cvlibs.net/datasets/kitti-360/documentation.php">官方说明</a> · <a href="data/kitti360_mixed/download_manifest.json">下载记录</a> · <a href="mixed_summary.json">指标汇总</a></p>
<p>每组 5×672×672 一次联合预测：真实针孔保留长宽比并 remap 到居中方形画布（有空白边）；鱼眼按 MEI 标定展开左/中/右/上四个 90° 透视图。所有虚拟视图保留真实鱼眼相机位置。使用两模型支持的内参与外参条件，不输入激光或真实深度。</p>
<p>四视图的 Z 深度先转为原鱼眼射线距离，重叠处按置信度×光轴余弦平方×边缘权重取加权中值；真实针孔也恢复到原像素深度。分别保存两张深度图，再按各自光心与标定外参生成点云。没有直接拼接四份虚拟点云；两台真实相机可能重复观测同一表面，未做跨相机去重。</p>
<div class="notice">这次 Pi3X 的稀疏激光诊断更好，OmniVGGT 出现明显尺度偏差。两个模型会归一化位姿；此实验用“标定基线 / 预测基线”的中位数统一校正所有深度，不能保证真实米制准确性。最终旋转和平移仍用标定。激光仅用于误差诊断，没有用来调倍率。它是 2–80 m 的稀疏近邻像素投影比较，未做扫描运动补偿或严格遮挡过滤，不能当作官方基准。样例很少，不代表模型整体优劣。</div>
<p>每组约 159 万点；原针孔覆盖约 99.7%，鱼眼保守有效圆内覆盖约 69.3%。浏览器点云最多显示 7 万点，默认只显示参考原点 80 m 内，完整 PLY 不裁剪。可按相机切换蓝/橙颜色检查接缝。鱼眼四区域没有覆盖的像素保持 NaN。<a href="mixed_overview.jpg">三例预览大图</a> · <a href="pointcloud.html">之前的单鱼眼实验</a></p>
'''+''.join(sections)+'''<p>数据来源：Liao, Xie and Geiger, KITTI-360, PAMI 2022；CC BY-NC-SA 3.0。原始图片、权重和点云不进入公开代码仓库。<a href="https://github.com/happyevery365/resulttest">复现代码</a>。</p></html>'''
    (root/'mixed_comparison.html').write_text(html)
    print(json.dumps(records,indent=2)); print('REPORT AND PIXEL UNIQUENESS AUDIT PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    build(p.parse_args().root)
