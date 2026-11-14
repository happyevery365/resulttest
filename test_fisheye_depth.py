"""Geometry/scale/occlusion tests independent of the neural network and GPU."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from fisheye_depth import (KB4, backproject_depths, fuse, load_camera, pinhole_rays,
                           prepare, rotation, sample_depth, weighted_median)


def synthetic_views(size=112):
    cam = KB4([[28, 0, 47.5], [0, 28, 47.5], [0, 0, 1]], [0, 0, 0, 0], 96, 96, 190)
    rays, valid = cam.ray_map()
    K = np.array([[size / 2, 0, (size - 1) / 2], [0, size / 2, (size - 1) / 2], [0, 0, 1]])
    R = np.stack([rotation(-45, 0), rotation(0, 0), rotation(45, 0), rotation(0, 45)])
    vrays = pinhole_rays(K, size)
    masks = np.stack([cam.project(vrays @ r.T)[2] for r in R])
    data = dict(images=np.full((4, size, size, 3), 128, 'u1'), valid=masks,
                K=np.repeat(K[None], 4, 0), R=R,
                source_rays=rays, source_valid=valid,
                source_image=np.full((96, 96, 3), 128, 'u1'))
    return data, vrays


class GeometryTests(unittest.TestCase):
    def test_kb4_roundtrip_beyond_180(self):
        cam = KB4([[80, 0, 150], [0, 81, 150], [0, 0, 1]], [.01, -.001, 0, 0], 301, 301, 210)
        t = np.deg2rad(np.array([0, 30, 60, 89, 95, 104]))
        rays = np.stack([np.sin(t), np.zeros_like(t), np.cos(t)], -1)
        u, v, ok = cam.project(rays)
        recovered, inverse_ok = cam.unproject(u, v)
        self.assertTrue((ok & inverse_ok).all())
        np.testing.assert_allclose(recovered, rays, atol=1e-10)
        self.assertLess(recovered[-1, 2], 0)

    def test_rotation_signs(self):
        forward = np.array([0., 0., 1.])
        self.assertLess((rotation(-45, 0) @ forward)[0], 0)
        self.assertGreater((rotation(45, 0) @ forward)[0], 0)
        self.assertLess((rotation(0, 45) @ forward)[1], 0)

    def test_opencv_fisheye_forward_agrees_in_front_hemisphere(self):
        import cv2
        K = np.array([[120., 0, 200], [0, 130, 210], [0, 0, 1]])
        D = np.array([.01, -.001, .0001, -.00001])
        cam = KB4(K, D, 420, 420, 190)
        pts = np.array([[.2, -.3, 1.], [-.7, .5, 1.], [0, 0, 1.]])
        uv, _ = cv2.fisheye.projectPoints(pts[None], np.zeros(3), np.zeros(3), K, D)
        u, v, _ = cam.project(pts)
        np.testing.assert_allclose(np.stack([u, v], -1), uv[0], atol=1e-9)

    def test_reject_folded_calibration(self):
        with self.assertRaisesRegex(ValueError, 'folds'):
            KB4(np.diag([100., 100, 1]), [-1, 0, 0, 0], 200, 200, 190)

    def test_colmap_resize_pixel_centers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'camera.json'
            path.write_text(json.dumps(dict(camera=dict(model='OPENCV_FISHEYE', width=400,
                height=400, params=[100, 100, 200, 200, 0, 0, 0, 0], fov_deg=190))))
            cam, _ = load_camera(path, 200, 200)
            self.assertEqual(cam.K[0, 0], 50)
            self.assertEqual(cam.K[0, 2], 99.5)

    def test_sphere_range_not_axis_z(self):
        data, vrays = synthetic_views()
        z = np.repeat((5 * vrays[..., 2])[None], 4, 0)
        radial, _, counts, _, _, _ = backproject_depths(data, z, np.ones_like(z))
        good = counts > 0
        # Bilinear interpolation should recover constant radius, including seams.
        np.testing.assert_allclose(radial[good], 5, atol=.003)
        self.assertTrue((counts >= 2).any())
        # Uncovered lower part must NOT be invented or set to finite zero depth.
        self.assertTrue((data['source_valid'] & ~good).any())
        self.assertTrue(np.isnan(radial[~good]).all())

    def test_known_plane_reconstruction(self):
        data, vrays = synthetic_views()
        z = []
        for R in data['R']:
            source_dirs = vrays @ R.T
            ranges = 4 / np.maximum(source_dirs[..., 2], 1e-9)
            z.append(np.where(source_dirs[..., 2] > .3, ranges * vrays[..., 2], np.nan))
        z = np.stack(z)
        radial, _, counts, _, _, _ = backproject_depths(data, z, np.ones_like(z))
        good = (counts > 0) & (data['source_rays'][..., 2] > .5)
        points = data['source_rays'][good] * radial[good, None]
        np.testing.assert_allclose(points[:, 2], 4, atol=.005)

    def test_depth_edges_do_not_average_foreground_background(self):
        z = np.array([[1., 100.], [1., 100.]])
        sampled, _, ok = sample_depth(z, np.ones_like(z), np.ones_like(z, bool), np.array([.49, .51]), np.array([.5, .5]))
        np.testing.assert_array_equal(sampled, [1, 100])
        self.assertTrue(ok.all())

    def test_invalid_depth_does_not_poison_neighbor(self):
        z = np.array([[2., np.nan], [2., np.nan]])
        sampled, _, ok = sample_depth(z, np.ones_like(z), np.isfinite(z), np.array([.2, .8]), np.array([.5, .5]))
        self.assertEqual(sampled[0], 2)
        np.testing.assert_array_equal(ok, [True, False])

    def test_weighted_median_rejects_low_weight_outlier(self):
        values = np.array([[[5.]], [[5.1]], [[4.9]], [[100.]]])
        weights = np.array([[[1.]], [[1.]], [[1.]], [[.01]]])
        result, _, _ = weighted_median(values, weights)
        self.assertEqual(result.item(), 5)

    def test_all_invalid_no_depth_and_confidence_filter(self):
        data, vrays = synthetic_views()
        z = np.repeat((5 * vrays[..., 2])[None], 4, 0)
        result = backproject_depths(data, z, np.full_like(z, .1), min_confidence=.2)
        self.assertTrue(np.isnan(result[0]).all())
        self.assertTrue((result[2] == 0).all())
        result = backproject_depths(data, np.full_like(z, np.nan), np.ones_like(z))
        self.assertTrue(np.isnan(result[0]).all())

    def test_end_to_end_one_point_per_original_pixel(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.fromarray(np.full((96, 96, 3), 128, 'u1')).save(root / 'image.png')
            (root / 'cal.json').write_text(json.dumps(dict(model='kb4', width=96, height=96,
                K=[[28, 0, 47.5], [0, 28, 47.5], [0, 0, 1]], D=[0, 0, 0, 0], fov_deg=190,
                pixel_convention='opencv')))
            prepare(root / 'image.png', root / 'cal.json', root / 'prepared', size=112)
            data = np.load(root / 'prepared/prepared.npz')
            z = np.stack([5 * pinhole_rays(K, 112)[..., 2] for K in data['K']])
            np.savez(root / 'depth.npz', depth_z=z, confidence=np.ones_like(z))
            report = fuse(root / 'prepared', root / 'depth.npz', root / 'fused')
            xy = np.load(root / 'fused/point_source_uv.npy')
            points = np.load(root / 'fused/points.npy')
            ranges = np.load(root / 'fused/fisheye_range.npy')
            self.assertEqual(len(xy), len(np.unique(xy, axis=0)))
            self.assertEqual(len(points), report['reconstructed_pixels'])
            self.assertEqual(len(points), np.isfinite(ranges).sum())
            np.testing.assert_allclose(np.linalg.norm(points, axis=-1), 5, atol=.003)
            self.assertEqual(len(list(root.rglob('*.ply'))), 1)


if __name__ == '__main__':
    unittest.main()
