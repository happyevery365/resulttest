import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dataset import CalibrationError, inspect, normalize, read_yaml, select
from geometry import camera_library, make_camera, prepare, remap_view, rotation
from evaluate_cloud import measure
from outputs import export
from inspect_schema import shape


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.case = self.root / 'case'
        (self.case / 'conf').mkdir(parents=True)
        for camera in ('cam_0', 'cam_1'):
            folder = self.case / 'picture' / camera
            folder.mkdir(parents=True)
            for number in ('000000010', '000000020'):
                Image.new('RGB', (32, 32), (0, 0, 0)).save(folder / (number + '.jpg'))
            (self.case / 'conf' / (camera + '.yaml')).write_text('model: pinhole\nwidth: 32\nheight: 32\nintrinsics: [16, 16, 15.5, 15.5]\n')

    def tearDown(self):
        self.temp.cleanup()

    def calibration(self, kind='pinhole'):
        return {'model': kind, 'width': 32, 'height': 32, 'intrinsics': [16, 16, 15.5, 15.5], **({'D': [0, 0, 0, 0]} if kind == 'kb4' else {})}

    def test_shared_identifier_and_single_image(self):
        case, selected, report = select(self.case)
        self.assertEqual(report['frame_id'], '10')
        self.assertEqual(len(selected), 2)
        self.assertFalse(report['synchronization_verified'])
        _, one, _ = select(image=selected[0][1])
        self.assertEqual(one, [selected[0]])
        _, chosen, _ = select(self.case, frame_id='000020', cameras='1')
        self.assertEqual(chosen[0][0], 'cam_1')
        self.assertTrue(chosen[0][1].name.startswith('000000020'))

    def test_duplicate_and_missing_frames_fail(self):
        folder = self.case / 'picture/cam_0'
        Image.new('RGB', (32, 32)).save(folder / '10.png')
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            select(self.case)
        (folder / '10.png').unlink()
        with self.assertRaisesRegex(ValueError, 'not common'):
            select(self.case, frame_id='999')

    def test_full_large_number_without_float_roundoff(self):
        for camera in ('cam_0', 'cam_1'):
            Image.new('RGB', (32, 32)).save(self.case / 'picture' / camera / '1730300000123456789.jpg')
        _, _, info = select(self.case, frame_id='1730300000123456789')
        self.assertEqual(info['frame_id'], '1730300000123456789')

    def test_unknown_projection_not_guessed_from_four_coefficients(self):
        c = self.calibration(); c.pop('model'); c['D'] = [0, 0, 0, 0]
        with self.assertRaises(CalibrationError):
            normalize(c, 'cam_0', (32, 32))
        c['model'] = 'fisheye'
        with self.assertRaises(CalibrationError):
            normalize(c, 'cam_0', (32, 32))

    def test_ros_and_kalibr_and_camodocal(self):
        c = self.calibration(); c['distortion_model'] = 'equidistant'; c['distortion_coeffs'] = [0, 0, 0, 0]
        self.assertTrue(normalize({'cam0': c}, 'cam_0', (32, 32))['fisheye'])
        c = {'model_type': 'KANNALA_BRANDT', 'image_width': 32, 'image_height': 32,
             'projection_parameters': {'mu': 16, 'mv': 16, 'u0': 15.5, 'v0': 15.5, 'k2': .1, 'k3': .2, 'k4': .3, 'k5': .4}}
        self.assertEqual(normalize(c, 'cam_0', (32, 32))['D'], [.1, .2, .3, .4])

    def test_opencv_yaml_and_resolution(self):
        path = self.root / 'opencv.yaml'
        path.write_text('%YAML:1.0\n---\nmodel: pinhole\nwidth: 32\nheight: 32\nK: !!opencv-matrix\n  rows: 3\n  cols: 3\n  dt: d\n  data: [16,0,15.5,0,16,15.5,0,0,1]\n')
        cal = normalize(read_yaml(path), 'cam_0', (32, 32))
        self.assertEqual(cal['kind'], 'pinhole')
        with self.assertRaisesRegex(CalibrationError, 'differs'):
            normalize(read_yaml(path), 'cam_0', (16, 16))

    def test_pose_convention_is_required(self):
        c = self.calibration(); c['T_world_camera'] = np.eye(4).tolist()
        with self.assertRaises(CalibrationError):
            normalize(c, 'cam_0', (32, 32))
        c.update(translation_unit='m', camera_axes='opencv')
        self.assertIsNotNone(normalize(c, 'cam_0', (32, 32))['T_world_camera'])

    def test_rectified_flag_cannot_reuse_distorted_calibration(self):
        c = self.calibration('kb4'); c['is_rectified'] = True
        with self.assertRaisesRegex(CalibrationError, 'rectified'):
            normalize(c, 'cam_0', (32, 32))

    def test_schema_hides_values(self):
        report = json.dumps(shape({'camera': {'fx': 123456789, 'password': 'not-for-report', 'K': list(range(9))}}))
        self.assertNotIn('123456789', report)
        self.assertNotIn('not-for-report', report)
        self.assertIn('"length": 9', report)

    def test_cloud_metrics_known_offset(self):
        a = np.array([[0., 0, 0], [10, 0, 0]])
        self.assertAlmostEqual(measure(a, a, .1)['f1'], 1)
        m = measure(a, a + [0, 1, 0], .1)
        self.assertAlmostEqual(m['symmetric_mean_distance'], 1)
        self.assertEqual(m['f1'], 0)

    @unittest.skipUnless(os.environ.get('ANY3R_REPO'), 'Set ANY3R_REPO to enable camera geometry tests')
    def test_identity_remap_and_fisheye_back_hemisphere(self):
        lib = camera_library(os.environ['ANY3R_REPO'])
        cal = normalize(self.calibration(), 'cam_0', (32, 32))
        cam = make_camera(lib, cal)
        image = np.arange(32 * 32 * 3).reshape(32, 32, 3).astype('uint8')
        rgb, target, valid, R = remap_view(image, cam, lib, 32, 90, 0, 0)
        self.assertTrue(np.allclose(R, np.eye(3)))
        self.assertGreater(valid.mean(), .9)
        self.assertLessEqual(int(np.abs(rgb[valid].astype(int) - image[valid].astype(int)).max()), 1)
        fish = lib.KB4(10, 10, 16, 16, width=32, height=32)
        ray = np.array([[np.sin(np.deg2rad(100)), 0, np.cos(np.deg2rad(100))]])
        u, v, ok = fish.project(ray)
        back, ok2 = fish.pixel_to_ray(u, v)
        np.testing.assert_allclose(back, ray, atol=1e-7)
        self.assertTrue(ok.all() and ok2.all())
        np.testing.assert_allclose(rotation(90, 0) @ [0, 0, 1], [1, 0, 0], atol=1e-7)

    @unittest.skipUnless(os.environ.get('ANY3R_REPO'), 'Set ANY3R_REPO to enable camera geometry tests')
    def test_preparation_and_dark_surface_exports(self):
        case, selected, selection = select(self.case)
        report = inspect(case, selected, selection)
        out = self.root / 'out'; out.mkdir()
        prepared = prepare(report, os.environ['ANY3R_REPO'], out, 112, 'raw', 100, [(0, 0)])
        self.assertEqual(prepared['images'].shape, (2, 112, 112, 3))
        # A black image is still valid geometry; exclude only geometry padding/masks.
        points = prepared['rays'] * 2
        poses = np.tile(np.eye(4), (2, 1, 1))
        metrics = export(out, prepared, points, np.ones((2, 112, 112)), poses, 20)
        self.assertEqual(metrics['exported_points'], 2 * 112 * 112)
        self.assertTrue((out / 'viewer.html').exists())
        self.assertFalse(metrics['accuracy_metrics_available'])


if __name__ == '__main__':
    unittest.main()
