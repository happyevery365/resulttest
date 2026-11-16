"""Adapter-specific pose and confidence tests; shared geometry is tested separately."""
import unittest
import numpy as np
from omnivggt_fisheye_depth import camera_inputs, confidence_weights
from fisheye_depth import weighted_median
from test_fisheye_depth import synthetic_views


class OmniAdapterTests(unittest.TestCase):
    def test_camera_frame_roundtrip_and_zero_baseline(self):
        data, rays = synthetic_views()
        _, w2c = camera_inputs(data)
        self.assertTrue(np.all(w2c[:, :, 3] == 0))
        # A known point in each virtual frame must return to that frame after
        # its source-frame transformation. Passing R instead of R.T fails.
        p_virtual = np.array([.2, -.3, 2.])
        for R, E in zip(data['R'], w2c):
            np.testing.assert_allclose(E[:, :3] @ (R @ p_virtual) + E[:, 3], p_virtual, atol=1e-6)

    def test_off_center_intrinsics_rejected(self):
        data, _ = synthetic_views()
        data['K'][0, 0, 2] += 5
        with self.assertRaises(ValueError):
            camera_inputs(data)

    def test_global_confidence_scale_preserves_fusion(self):
        raw = np.array([[2., 20.], [5., 3.], [9., 8.], [4., 1.]])
        values = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
        normalized, scale = confidence_weights(raw, np.ones_like(raw, bool))
        self.assertEqual(scale, 20.)
        np.testing.assert_allclose(normalized, raw / 20)
        np.testing.assert_array_equal(weighted_median(values, raw)[0], weighted_median(values, normalized)[0])
        # Per-view normalization would destroy these relative confidences.
        np.testing.assert_allclose(normalized[2, 0] / normalized[0, 0], 4.5)

    def test_invalid_confidence_not_used(self):
        raw = np.array([np.inf, np.nan, -2., 4., 100.])
        valid = np.array([True, True, True, True, False])
        weights, scale = confidence_weights(raw, valid)
        np.testing.assert_array_equal(weights, [0, 0, 0, 1, 0])
        self.assertEqual(scale, 4)
        with self.assertRaises(ValueError):
            confidence_weights(raw, np.zeros_like(valid))


if __name__ == '__main__':
    unittest.main()
