"""Real KITTI-360 MEI + rectified pinhole cameras, with explicit rig transforms."""
from pathlib import Path
import numpy as np
import yaml


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-15)


class Pinhole:
    def __init__(self, K, width, height):
        self.K, self.width, self.height = np.asarray(K, float), int(width), int(height)

    def unproject(self, u, v):
        q = np.stack([(u-self.K[0,2])/self.K[0,0], (v-self.K[1,2])/self.K[1,1], np.ones_like(u)],-1)
        return unit(q), np.ones_like(u,dtype=bool)

    def project(self, rays):
        z=rays[...,2]
        u=self.K[0,0]*rays[...,0]/np.maximum(z,1e-15)+self.K[0,2]
        v=self.K[1,1]*rays[...,1]/np.maximum(z,1e-15)+self.K[1,2]
        ok=(z>0)&(u>=0)&(v>=0)&(u<=self.width-1)&(v<=self.height-1)
        return u,v,ok

    def ray_map(self):
        v,u=np.indices((self.height,self.width),dtype=float)
        return self.unproject(u,v)


class Mei(Pinhole):
    """Unified sphere + full Brown distortion, including calibrated tangential terms."""
    def __init__(self,K,D,xi,width,height):
        super().__init__(K,width,height)
        self.D=np.asarray(D,float)
        self.xi=float(xi)
        if self.D.shape!=(4,) or self.xi<0 or not np.isfinite(self.D).all():
            raise ValueError('MEI requires finite k1,k2,p1,p2 and nonnegative xi')

    def distort(self,x,y):
        k1,k2,p1,p2=self.D; r2=x*x+y*y; L=1+k1*r2+k2*r2*r2
        return x*L+2*p1*x*y+p2*(r2+2*x*x), y*L+p1*(r2+2*y*y)+2*p2*x*y

    def image_mask(self,u,v):
        # Conservative inscribed image circle, explicitly not a measured lens FOV.
        return ((u-(self.width-1)/2)**2+(v-(self.height-1)/2)**2 <= (min(self.width,self.height)/2)**2)

    def unproject(self,u,v):
        xd=(u-self.K[0,2])/self.K[0,0]; yd=(v-self.K[1,2])/self.K[1,1]
        x,y=xd.copy(),yd.copy(); k1,k2,p1,p2=self.D
        for _ in range(25):
            ax,ay=self.distort(x,y); ex,ey=ax-xd,ay-yd
            r2=x*x+y*y; L=1+k1*r2+k2*r2*r2; dx=2*x*(k1+2*k2*r2); dy=2*y*(k1+2*k2*r2)
            a=L+x*dx+2*p1*y+6*p2*x; b=x*dy+2*p1*x+2*p2*y
            c=y*dx+2*p1*x+2*p2*y; d=L+y*dy+6*p1*y+2*p2*x
            det=a*d-b*c; det=np.where(np.abs(det)>1e-12,det,np.nan)
            x-=np.clip((d*ex-b*ey)/det,-.2,.2); y-=np.clip((-c*ex+a*ey)/det,-.2,.2)
        r2=x*x+y*y; disc=1+(1-self.xi**2)*r2
        lam=(self.xi+np.sqrt(np.maximum(disc,0)))/(1+r2)
        rays=unit(np.stack([lam*x,lam*y,lam-self.xi],-1))
        ax,ay=self.distort(x,y)
        valid=(disc>1e-10)&np.isfinite(rays).all(-1)&(np.hypot(ax-xd,ay-yd)<1e-8)&self.image_mask(u,v)
        return np.where(valid[...,None],rays,[0,0,1]),valid

    def project(self,rays):
        r=unit(rays); den=r[...,2]+self.xi
        x,y=self.distort(r[...,0]/np.maximum(den,1e-15),r[...,1]/np.maximum(den,1e-15))
        u=self.K[0,0]*x+self.K[0,2]; v=self.K[1,1]*y+self.K[1,2]
        # The plus-root inverse is injective on z > -1/xi for xi>1.
        branch = r[...,2] > (-1/self.xi if self.xi>1 else -self.xi)
        ok=branch&(den>0)&np.isfinite(u)&np.isfinite(v)&(u>=0)&(v>=0)&(u<=self.width-1)&(v<=self.height-1)&self.image_mask(u,v)
        return u,v,ok


def parse_matrix_file(path):
    result={}
    for line in Path(path).read_text().splitlines():
        key,sep,value=line.partition(':')
        if not sep: continue
        try: result[key]=np.array([float(x) for x in value.split()])
        except ValueError: pass
    return result


def homogeneous(values):
    T=np.eye(4); T[:3]=np.asarray(values).reshape(3,4)
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(T[:3,:3])<.999:
        raise ValueError('Invalid calibrated rigid transform')
    return T


def load_kitti_calibration(root):
    root=Path(root); cal=root/'calibration'
    p=parse_matrix_file(cal/'perspective.txt'); poses=parse_matrix_file(cal/'calib_cam_to_pose.txt')
    w,h=p['S_rect_00'].astype(int); K=p['P_rect_00'].reshape(3,4)[:,:3]
    pin=Pinhole(K,w,h)
    rect=np.eye(4); rect[:3,:3]=p['R_rect_00'].reshape(3,3)
    T_rig_raw=homogeneous(poses['image_00'])
    T_rig_pin=T_rig_raw@np.linalg.inv(rect)
    text=(cal/'image_02.yaml').read_text(); text='\n'.join(s for s in text.splitlines() if not s.startswith('%'))
    f=yaml.safe_load(text); pp=f['projection_parameters']; dd=f['distortion_parameters']
    fish=Mei([[pp['gamma1'],0,pp['u0']],[0,pp['gamma2'],pp['v0']],[0,0,1]],
             [dd[k] for k in ('k1','k2','p1','p2')],f['mirror_parameters']['xi'],f['image_width'],f['image_height'])
    T_rig_fish=homogeneous(poses['image_02'])
    # Reference frame is the real rectified perspective camera at this timestamp.
    T_ref_fish=np.linalg.inv(T_rig_pin)@T_rig_fish
    T_velo_raw=homogeneous(np.loadtxt(cal/'calib_cam_to_velo.txt'))
    T_ref_velo=rect@np.linalg.inv(T_velo_raw)
    return pin,fish,[np.eye(4),T_ref_fish],T_ref_velo


def compose_virtual_pose(T_ref_source,R_source_virtual):
    local=np.eye(4); local[:3,:3]=R_source_virtual
    return T_ref_source@local


def baseline_scale(predicted_c2w,known_c2w,source_ids):
    """Estimate ONE depth scale from distinct real-camera baseline lengths (not LiDAR)."""
    ratios=[]; predicted_lengths=[]; known_lengths=[]
    for i in range(len(source_ids)):
        for j in range(i):
            if source_ids[i]==source_ids[j]: continue
            known=float(np.linalg.norm(known_c2w[i,:3,3]-known_c2w[j,:3,3]))
            predicted=float(np.linalg.norm(predicted_c2w[i,:3,3]-predicted_c2w[j,:3,3]))
            if known>1e-5 and predicted>1e-6 and np.isfinite(predicted):
                ratios.append(known/predicted); predicted_lengths.append(predicted); known_lengths.append(known)
    if not ratios:
        raise ValueError('Predicted real-camera baseline collapsed; cannot estimate shared depth scale')
    scale=float(np.median(ratios))
    return scale,dict(method='median known/predicted baseline over distinct physical cameras; one common multiplier for all depth maps',
                      multiplier=scale,predicted_baselines=predicted_lengths,known_baselines_m=known_lengths,
                      ratio_p05_p95=np.percentile(ratios,[5,95]).tolist(),ground_truth_depth_used=False)
