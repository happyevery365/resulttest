#!/usr/bin/env python3
"""Real pinhole + MEI fisheye joint inference; depth-first reconstruction per real camera."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw

from fisheye_depth import (NAMES,depth_preview,dump,new_dir,pinhole_rays,rotation,
                           sample_depth,view_coordinates,weighted_median)
from mixed_camera_geometry import load_kitti_calibration,compose_virtual_pose,baseline_scale
from omnivggt_fisheye_depth import confidence_weights
from outputs import save_ply,viewer


def prepare(data_root,frame,out,size=672):
    if size<=0 or size%112:
        raise ValueError('Size must be a positive multiple of 112')
    root=Path(data_root); pin,fish,poses,T_ref_velo=load_kitti_calibration(root)
    photos=[np.array(Image.open(root/'frames'/f'{frame:010d}'/f'image_{i:02d}.png').convert('RGB')) for i in (0,2)]
    for image,cam in zip(photos,[pin,fish]):
        if image.shape[:2]!=(cam.height,cam.width):
            raise ValueError('Image dimensions differ from calibration; refusing implicit resize')
    out=new_dir(out); source_rays=[]; source_valid=[]
    for name,image,cam in zip(['pinhole','fisheye'],photos,[pin,fish]):
        Image.fromarray(image).save(out/(name+'_source.png'))
        rays,valid=cam.ray_map(); source_rays.append(rays.astype('f4')); source_valid.append(valid)
    # Reproject the real perspective photo to a centered square canvas, preserving
    # the full original FOV without stretching. Empty margins stay masked.
    xextent=max(pin.K[0,2]/pin.K[0,0],(pin.width-1-pin.K[0,2])/pin.K[0,0])
    yextent=max(pin.K[1,2]/pin.K[1,1],(pin.height-1-pin.K[1,2])/pin.K[1,1])
    pin_f=(size-1)/2/max(xextent,yextent)
    spec=[('pinhole',0,np.eye(3),pin_f)]+[(name,1,rotation(y,p),size/2) for name,(y,p) in zip(NAMES,[(-45,0),(0,0),(45,0),(0,45)])]
    images=[]; masks=[]; Ks=[]; Rs=[]; Ts=[]; ids=[]; details=[]
    montage=Image.new('RGB',(size*3,(size+24)*2),'#181818'); draw=ImageDraw.Draw(montage)
    for index,(name,source,R,f) in enumerate(spec):
        K=np.array([[f,0,(size-1)/2],[0,f,(size-1)/2],[0,0,1.]])
        rays=pinhole_rays(K,size); cam=[pin,fish][source]
        u,v,valid=cam.project(rays@R.T)
        # Verify the selected projection branch by ray roundtrip.
        back,ok=cam.unproject(u,v)
        valid &= ok & (np.sum(back*(rays@R.T),-1)>1-1e-6)
        image=cv2.remap(photos[source],np.where(valid,u,-1).astype('f4'),np.where(valid,v,-1).astype('f4'),cv2.INTER_LINEAR)
        image[~valid]=0
        if valid.mean()<.1: raise ValueError(f'{name}: insufficient valid RGB')
        Image.fromarray(image).save(out/(name+'.png'))
        Image.fromarray((valid*255).astype('u1')).save(out/(name+'_valid.png'))
        T=compose_virtual_pose(poses[source],R)
        images.append(image); masks.append(valid); Ks.append(K); Rs.append(R); Ts.append(T); ids.append(source)
        details.append(dict(name=name,source=['pinhole','fisheye'][source],K=K.tolist(),T_ref_virtual=T.tolist(),valid_fraction=float(valid.mean())))
        x=index%3*size; y=index//3*(size+24); draw.text((x+8,y+4),name,fill='white'); montage.paste(Image.fromarray(image),(x,y+24))
    montage.save(out/'inputs.png')
    np.savez_compressed(out/'prepared.npz',images=np.stack(images),valid=np.stack(masks),K=np.stack(Ks),R=np.stack(Rs),
                        poses=np.stack(Ts),source_ids=np.array(ids),source_poses=np.stack(poses),
                        pinhole_rays=source_rays[0],fisheye_rays=source_rays[1],pinhole_valid=source_valid[0],fisheye_valid=source_valid[1],
                        pinhole_image=photos[0],fisheye_image=photos[1],T_ref_velo=T_ref_velo)
    timestamps={}
    for cam in ('image_00','image_02'):
        path=root/'timestamps'/(cam+'.txt')
        if path.is_file(): timestamps[cam]=path.read_text().splitlines()[frame]
    skew=None
    if len(timestamps)==2:
        skew=float((np.datetime64(timestamps['image_02'],'ns')-np.datetime64(timestamps['image_00'],'ns'))/np.timedelta64(1,'ms'))
    if skew is not None and abs(skew)>50:
        raise ValueError(f'Camera timestamps differ by {skew} ms; require an explicit time compensation design')
    dump(out/'preparation.json',dict(dataset='KITTI-360',frame=frame,data_root=str(root.resolve()),size=size,views=details,
        baseline_m=float(np.linalg.norm(poses[1][:3,3])),reference_frame='rectified image_00 camera, x right y down z forward',
        timestamps=timestamps,fisheye_minus_pinhole_ms=skew,
        fish_model='MEI, full k1/k2/p1/p2 and xi; plus-root inverse with roundtrip verification',
        fisheye_support='conservative inscribed image circle plus valid invertible model branch; not a measured lens FOV',
        pinhole_input='real rectified perspective photo remapped to centered square canvas preserving full FOV; masked empty margins',
        source_hashes=[hashlib.sha256((root/'frames'/f'{frame:010d}'/f'image_{i:02d}.png').read_bytes()).hexdigest() for i in (0,2)]))
    print(f'PREPARED frame={frame}, 1 real pinhole + 4 fisheye virtual views; baseline={np.linalg.norm(poses[1][:3,3]):.4f}m, skew={skew}ms',flush=True)


def infer(prepared,model_name,repo,checkpoint,out,device='cuda:0'):
    import torch
    from safetensors.torch import load_file
    repo=Path(repo).resolve(); checkpoint=Path(checkpoint).resolve()
    if not checkpoint.is_file(): raise ValueError('Local checkpoint file required')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['OMNIVGGT_SKIP_BACKBONE_DOWNLOAD']='1'
    sys.path.insert(0,str(repo)); torch.set_num_threads(8); torch.manual_seed(0); torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(); data=np.load(Path(prepared)/'prepared.npz',allow_pickle=False)
    out=new_dir(out); N,H,W=data['images'].shape[:3]; K=data['K'].copy(); T=data['poses']
    tensor=lambda a:torch.as_tensor(a,device=device,dtype=torch.float32)
    images=tensor(data['images'].astype('f4')/255).permute(0,3,1,2)[None].contiguous()
    print(f'Loading {model_name}, local strict checkpoint...',flush=True); start=time.perf_counter()
    if model_name=='pi3x':
        from pi3.models.pi3x import Pi3X
        model=Pi3X(use_multimodal=True); K[:,:2,2]+=.5
        inputs=dict(imgs=images,intrinsics=tensor(K)[None],poses=tensor(T)[None],
                    mask_add_ray=torch.ones((1,N),dtype=torch.bool,device=device),mask_add_pose=torch.ones((1,N),dtype=torch.bool,device=device),
                    mask_add_depth=torch.zeros((1,N),dtype=torch.bool,device=device))
    else:
        if 'OMNIVGGT_SKIP_BACKBONE_DOWNLOAD' not in (repo/'omnivggt/models/aggregator.py').read_text():
            raise RuntimeError('Apply the existing offline initialization patch first')
        from omnivggt.models.omnivggt import OmniVGGT
        from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
        model=OmniVGGT()
        inputs=dict(images=images,intrinsics=tensor(K)[None],extrinsics=tensor(np.linalg.inv(T)[:,:3])[None],
                    depth=torch.zeros((1,N,H,W,1),device=device),mask=torch.zeros((1,N,H,W),device=device),
                    depth_gt_index=[],camera_gt_index=list(range(N)))
    model.load_state_dict(load_file(str(checkpoint)),strict=True); model=model.to(device).eval()
    torch.cuda.synchronize(); load_seconds=time.perf_counter()-start
    precision=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f'ONE joint forward: {N} views, {H}x{W}, measured nonzero camera baseline',flush=True); start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=precision):
        pred=model(**inputs) if model_name=='pi3x' else model.inference(**inputs)
    torch.cuda.synchronize(); seconds=time.perf_counter()-start
    arr=lambda x:x.detach().float().cpu().numpy()
    if model_name=='pi3x':
        z=arr(pred['local_points'][0,...,2]); raw=arr(pred['conf'][0,...,0].sigmoid()); poses=arr(pred['camera_poses'][0]); divisor=1.
        conf=np.where(data['valid']&np.isfinite(raw)&(raw>0),raw,0)
    else:
        z=arr(pred['depth'][0,...,0]); raw=arr(pred['depth_conf'][0]); conf,divisor=confidence_weights(raw,data['valid']&np.isfinite(z)&(z>0))
        ex,_=pose_encoding_to_extri_intri(pred['pose_enc'],(H,W)); E=np.tile(np.eye(4),(N,1,1)); E[:,:3]=arr(ex[0]); poses=np.linalg.inv(E)
    if z.shape!=(N,H,W): raise ValueError('Output size changed; refusing implicit depth resize')
    scale,scale_info=baseline_scale(poses,T,data['source_ids'])
    np.savez_compressed(out/'predictions.npz',depth_z=z,confidence=conf,raw_confidence=raw,predicted_c2w=poses,depth_scale=scale)
    for i,name in enumerate(['pinhole']+NAMES): depth_preview(out/(name+'_depth.png'),z[i],data['valid'][i]&np.isfinite(z[i])&(z[i]>0))
    dump(out/'inference.json',dict(model=model_name,repo=str(repo),checkpoint=str(checkpoint),python=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        precision=str(precision),load_seconds=load_seconds,inference_seconds=seconds,peak_GiB=torch.cuda.max_memory_allocated()/1024**3,
        input_views=N,input_size=[H,W],joint_forward_passes=1,intrinsics_used=True,poses_used=True,known_pose_unit='meters',
        depth_definition='local optical-axis Z',scale_alignment=scale_info,confidence_divisor=divisor,
        pose_use='Predicted pose baseline LENGTHS used only to estimate one global depth scale. All rotations/translations for final placement are calibration, never predicted poses.',
        geometry_note='Models normalize pose translations internally; supplying meter-valued poses alone does not guarantee metric output. Estimated baseline scale is not GT depth accuracy.',
        omni_principal_point_note='Omni tokens omit principal point and decode S/2; remap/backprojection uses known (S-1)/2. This 0.5px convention difference is not encoded.',
        prepared_sha256=hashlib.sha256((Path(prepared)/'prepared.npz').read_bytes()).hexdigest()))
    print(f'INFERRED {model_name}: {seconds:.2f}s, shared baseline depth multiplier={scale:.5g}',flush=True)


def sparse_lidar_diagnostic(data_root,frame,data,ranges,scale):
    path=Path(data_root)/'frames'/f'{frame:010d}'/'velodyne.bin'
    if not path.is_file(): return dict(available=False)
    pin,fish,poses,T=load_kitti_calibration(data_root)
    points=np.fromfile(path,dtype='<f4').reshape(-1,4)[:,:3]
    points=points@T[:3,:3].T+T[:3,3]
    report={}
    for s,(cam,name) in enumerate(zip([pin,fish],['pinhole','fisheye'])):
        local=(points-poses[s][:3,3])@poses[s][:3,:3]
        u,v,valid=cam.project(local); distance=np.linalg.norm(local,axis=-1)
        valid &= np.isfinite(distance)&(distance>2)&(distance<80)
        u=np.rint(u[valid]).astype(int); v=np.rint(v[valid]).astype(int); d=distance[valid]
        gt=np.full((cam.height,cam.width),np.inf); np.minimum.at(gt,(v,u),d)
        prediction=ranges[s]; ok=np.isfinite(gt)&np.isfinite(prediction)&(prediction>0)
        truth=gt[ok]; estimate=prediction[ok]
        if not len(truth): report[name]=dict(samples=0); continue
        rel=np.abs(estimate-truth)/truth
        report[name]=dict(samples=len(truth),abs_rel=float(rel.mean()),median_abs_rel=float(np.median(rel)),
                          rmse=float(np.sqrt(np.mean((estimate-truth)**2))),delta_1_25=float((np.maximum(estimate/truth,truth/estimate)<1.25).mean()),
                          predicted_to_lidar_ratio_median=float(np.median(estimate/truth)))
    return dict(available=True,range_filter_m=[2,80],cameras=report,scale_source='predicted-vs-calibrated camera baseline, NOT LiDAR',
                note='Sparse scan diagnostic only: nearest-pixel projection/z-buffer, no motion deskew or precise occlusion filtering; not official benchmark or dense GT. Dynamic objects, timing and cross-sensor visibility can affect errors.')


def fuse(prepared,prediction_dir,out):
    prepared=Path(prepared); prediction_dir=Path(prediction_dir)
    meta=json.loads((prepared/'preparation.json').read_text()); inf=json.loads((prediction_dir/'inference.json').read_text())
    if hashlib.sha256((prepared/'prepared.npz').read_bytes()).hexdigest()!=inf['prepared_sha256']: raise ValueError('Mismatched preparation')
    data=np.load(prepared/'prepared.npz'); pred=np.load(prediction_dir/'predictions.npz'); out=new_dir(out)
    scale=float(pred['depth_scale']); ranges=[]; xyzs=[]; rgbs=[]; metrics={}; identities=[]
    size=data['images'].shape[1]
    for source,name in enumerate(['pinhole','fisheye']):
        rays=data[name+'_rays']; source_valid=data[name+'_valid']; candidates=[]; weights=[]
        ids=np.flatnonzero(data['source_ids']==source)
        for i in ids:
            u,v,rz,inside=view_coordinates(rays,data['K'][i],data['R'][i],size)
            z,c,ok=sample_depth(pred['depth_z'][i],pred['confidence'][i],data['valid'][i],u,v,1.1)
            valid=source_valid&inside&ok&np.isfinite(z)&(z>0)
            d=z/np.maximum(rz,1e-12)*scale
            border=np.minimum.reduce([u,v,size-1-u,size-1-v]); taper=np.clip((border+1)/(.1*size),.01,1)
            candidates.append(np.where(valid,d,np.nan)); weights.append(np.where(valid,c*np.maximum(rz,0)**2*taper,0))
        candidates=np.stack(candidates); weights=np.stack(weights)
        radial,owner,available=weighted_median(candidates,weights); valid=np.isfinite(radial)&(radial>0)&source_valid
        if not valid.any():
            raise ValueError(f'{name}: no valid positive reconstructed depths')
        # Physical cameras have different ray origins: NEVER combine their depths by rotation alone.
        np.save(out/(name+'_range.npy'),radial.astype('f4')); saved=np.load(out/(name+'_range.npy'))
        np.save(out/(name+'_valid.npy'),valid); depth_preview(out/(name+'_range.png'),saved,valid)
        v,u=np.nonzero(valid); local=rays[valid]*saved[valid,None]; T=data['source_poses'][source]
        world=local@T[:3,:3].T+T[:3,3]; rgb=data[name+'_image'][valid]
        save_ply(out/(name+'_points.ply'),world,rgb); np.save(out/(name+'_point_uv.npy'),np.stack([u,v],-1).astype('i4'))
        ranges.append(saved); xyzs.append(world); rgbs.append(rgb); identities.append(np.full(len(world),source,'u1'))
        both=available.sum(0)>=2
        spread=np.full_like(radial,np.nan)
        spread[both]=(np.max(np.where(available,candidates,-np.inf),0)[both]-np.min(np.where(available,candidates,np.inf),0)[both])/radial[both]
        metrics[name]=dict(points=int(valid.sum()),source_valid_pixels=int(source_valid.sum()),coverage=float(valid.sum()/max(1,source_valid.sum())),
            overlap_relative_spread_p50_p95=np.percentile(spread[both],[50,95]).tolist() if both.any() else None,
            range_p05_p50_p95=np.percentile(saved[valid],[5,50,95]).tolist(),one_point_per_valid_source_pixel=True)
        np.savez_compressed(out/(name+'_fusion.npz'),candidate_range=candidates.astype('f4'),weights=weights.astype('f4'),selected_view=owner.astype('i1'))
        # Numerical audit: mapping is unique and transformed points retain the saved per-camera radius.
        assert len(world)==len(np.unique(v*valid.shape[1]+u))==int(valid.sum())
        back=(world-T[:3,3])@T[:3,:3]
        np.testing.assert_allclose(np.linalg.norm(back,axis=-1),saved[valid],rtol=3e-5,atol=1e-4)
    xyz=np.concatenate(xyzs); rgb=np.concatenate(rgbs)
    save_ply(out/'points.ply',xyz,rgb); viewer(out/'viewer.html',xyz,rgb)
    np.savez_compressed(out/'points.npz',xyz=xyz,rgb=rgb,source_camera=np.concatenate(identities))
    lidar=sparse_lidar_diagnostic(meta['data_root'],meta['frame'],data,ranges,scale)
    dump(out/'metrics.json',dict(model=inf['model'],frame=meta['frame'],cameras=metrics,total_points=len(xyz),scale_alignment=inf['scale_alignment'],
        lidar=lidar,virtual_point_cloud_fusion=False,physical_camera_clouds='Concatenated ONLY AFTER separate original-camera depth maps; no cross-camera duplicate removal',
        note='Points are placed with measured rig extrinsics and one estimated depth multiplier. Metric accuracy depends on the estimated scale; see sparse LiDAR diagnostic.'))
    print(f'FUSED {out}: {len(xyz)} points from TWO original-camera depth maps',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); subs=p.add_subparsers(dest='command',required=True)
    s=subs.add_parser('prepare'); s.add_argument('--data',type=Path,required=True); s.add_argument('--frame',type=int,required=True); s.add_argument('--size',type=int,default=672)
    s=subs.add_parser('infer'); s.add_argument('--prepared',type=Path,required=True); s.add_argument('--model',choices=['pi3x','omnivggt'],required=True)
    s.add_argument('--repo',type=Path,required=True); s.add_argument('--checkpoint',type=Path,required=True); s.add_argument('--device',default='cuda:0')
    s=subs.add_parser('fuse'); s.add_argument('--prepared',type=Path,required=True); s.add_argument('--predictions',type=Path,required=True)
    for s in subs.choices.values(): s.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare': prepare(a.data,a.frame,a.out,a.size)
    elif a.command=='infer': infer(a.prepared,a.model,a.repo,a.checkpoint,a.out,a.device)
    else: fuse(a.prepared,a.predictions,a.out)


if __name__=='__main__': main()
