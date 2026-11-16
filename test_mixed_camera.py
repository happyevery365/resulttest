"""Synthetic tests for heterogeneous cameras, actual translations and depth scale."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from mixed_camera_geometry import Mei,Pinhole,compose_virtual_pose,baseline_scale
from fisheye_depth import rotation,pinhole_rays


class MixedTests(unittest.TestCase):
    def test_mei_roundtrip_with_tangential_distortion(self):
        cam=Mei([[1300,0,699.5],[0,1301,699.5],[0,0,1]],[.016,1.65,.0004,-.0005],2.21,1400,1400)
        v,u=np.meshgrid(np.linspace(100,1250,41),np.linspace(100,1250,41))
        rays,ok=cam.unproject(u,v); uu,vv,valid=cam.project(rays); good=ok&valid
        self.assertGreater(good.sum(),1000)
        np.testing.assert_allclose(uu[good],u[good],atol=1e-7); np.testing.assert_allclose(vv[good],v[good],atol=1e-7)

    def test_mei_rejects_noninjective_branch(self):
        cam=Mei([[100,0,500],[0,100,500],[0,0,1]],[0,0,0,0],2.,1000,1000)
        theta=np.deg2rad(140); ray=np.array([[np.sin(theta),0,np.cos(theta)]])
        self.assertFalse(cam.project(ray)[2][0])

    def test_real_translation_preserved_in_virtual_cameras(self):
        source=np.eye(4); source[:3,:3]=rotation(70,10); source[:3,3]=[.3,.1,1.2]
        virtual=compose_virtual_pose(source,rotation(-45,0))
        np.testing.assert_allclose(virtual[:3,3],source[:3,3])
        point=np.array([.1,-.2,3.,1.]); local=np.eye(4); local[:3,:3]=rotation(-45,0)
        np.testing.assert_allclose(virtual@point,source@local@point)
        self.assertGreater(np.linalg.norm(virtual[:3,3]),1)

    def test_depth_scale_uses_only_distinct_camera_baselines(self):
        known=np.tile(np.eye(4),(5,1,1)); known[1:,:3,3]=[1,0,0]
        predicted=known.copy(); predicted[:,:3,3]*=4
        predicted[:,:3,3]+=np.array([2,3,4])  # global translation is irrelevant
        factor,info=baseline_scale(predicted,known,np.array([0,1,1,1,1]))
        self.assertEqual(factor,.25); self.assertEqual(len(info['known_baselines_m']),4)
        np.testing.assert_allclose(factor*predicted[1:,:3,3]-factor*predicted[0,:3,3],known[1:,:3,3])
        predicted[:,:3,3]=0
        with self.assertRaises(ValueError): baseline_scale(predicted,known,np.array([0,1,1,1,1]))

    def test_plane_depth_returns_to_two_separate_camera_origins(self):
        K=np.array([[20,0,10],[0,20,10],[0,0,1.]])
        cam=Pinhole(K,21,21); rays,valid=cam.ray_map()
        for tx in (0,1):
            # Plane in the reference frame z=4; both cameras face forward.
            radial=4/rays[...,2]; local=radial[...,None]*rays
            world=local+np.array([tx,0,0]); np.testing.assert_allclose(world[...,2],4)
            np.testing.assert_allclose(world[10,10],[tx,0,4])

    def test_end_to_end_depth_fusion_with_rotated_translated_camera(self):
        from mixed_camera_depth import fuse
        size=32; K=np.array([[32,0,15.5],[0,32,15.5],[0,0,1.]])
        rays=pinhole_rays(K,size)
        sources=np.tile(np.eye(4),(2,1,1))
        sources[1,:3,:3]=rotation(20,-5); sources[1,:3,3]=[.8,.1,.2]
        ids=np.array([0,1,1,1,1]); Rs=np.stack([np.eye(3),rotation(-15,0),np.eye(3),rotation(15,0),rotation(0,15)])
        poses=np.stack([compose_virtual_pose(sources[s],R) for s,R in zip(ids,Rs)])
        # A known world plane Z=4, represented in each virtual camera's optical Z.
        q=rays/rays[...,2:]
        depths=np.stack([(4-T[2,3])/(q@T[:3,:3].T)[...,2]/2 for T in poses])
        rgb=np.full((size,size,3),120,'u1'); valid=np.ones((size,size),bool)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); prep=root/'prepared'; pred=root/'prediction'
            prep.mkdir(); pred.mkdir()
            np.savez(prep/'prepared.npz',images=np.stack([rgb]*5),valid=np.stack([valid]*5),
                     K=np.stack([K]*5),R=Rs,poses=poses,source_ids=ids,source_poses=sources,
                     pinhole_rays=rays,fisheye_rays=rays,pinhole_valid=valid,fisheye_valid=valid,
                     pinhole_image=rgb,fisheye_image=rgb)
            (prep/'preparation.json').write_text(json.dumps(dict(data_root=str(root),frame=0)))
            (pred/'inference.json').write_text(json.dumps(dict(model='synthetic',scale_alignment={},
                prepared_sha256=hashlib.sha256((prep/'prepared.npz').read_bytes()).hexdigest())))
            np.savez(pred/'predictions.npz',depth_z=depths,confidence=np.ones_like(depths),depth_scale=2.)
            fuse(prep,pred,root/'fused')
            cloud=np.load(root/'fused/points.npz')
            self.assertEqual(set(cloud['source_camera']),{0,1})
            np.testing.assert_allclose(cloud['xyz'][:,2],4,atol=.003)
            for name in ['pinhole','fisheye']:
                uv=np.load(root/'fused'/(name+'_point_uv.npy'))
                self.assertEqual(len(uv),len(np.unique(uv,axis=0)))
                self.assertGreater(len(uv),size*size*.8)


if __name__=='__main__': unittest.main()
