#!/usr/bin/env python3
"""Ground-truth point-cloud metrics with an explicit coordinate/scale contract."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from dataset import dump


def load(path):
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected .npy [N,3] Cartesian points')
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(1)]
    if not len(points):
        raise ValueError('Empty finite point cloud')
    return points


def measure(pred, gt, threshold):
    d_pred = cKDTree(gt).query(pred, workers=-1)[0]
    d_gt = cKDTree(pred).query(gt, workers=-1)[0]
    precision, recall = float((d_pred <= threshold).mean()), float((d_gt <= threshold).mean())
    return {'mean_pred_to_gt': float(d_pred.mean()), 'mean_gt_to_pred': float(d_gt.mean()),
            'symmetric_mean_distance': float((d_pred.mean() + d_gt.mean()) / 2),
            'symmetric_mean_squared_distance': float(((d_pred**2).mean() + (d_gt**2).mean()) / 2),
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            'threshold': threshold}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred', required=True)
    p.add_argument('--gt', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--same-frame-and-scale', action='store_true')
    group.add_argument('--pred-to-gt', help='.npy 4x4 externally determined similarity transform; no automatic fitting')
    p.add_argument('--units', required=True, help='Common coordinate unit, e.g. m; both clouds must use it after transform')
    p.add_argument('--threshold', required=True, type=float)
    p.add_argument('--max-points', default=200000, type=int)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        p.error('Output exists; choose a new --out')
    if a.threshold <= 0 or not np.isfinite(a.threshold) or a.max_points <= 0:
        p.error('threshold and max-points must be positive and finite')
    pred, gt = load(a.pred), load(a.gt)
    T = np.eye(4)
    if a.pred_to_gt:
        T = np.load(a.pred_to_gt, allow_pickle=False)
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError('Invalid homogeneous transform')
        scales = np.linalg.svd(T[:3, :3], compute_uv=False)
        if scales.min() <= 0 or np.linalg.det(T[:3, :3]) <= 0 or not np.allclose(scales, scales[0], rtol=1e-4):
            raise ValueError('Transform must be a proper rigid/similarity transform, not anisotropic scaling or reflection')
        pred = pred @ T[:3, :3].T + T[:3, 3]
    original = [len(pred), len(gt)]
    rng = np.random.default_rng(0)
    pred = pred[rng.choice(len(pred), min(a.max_points, len(pred)), replace=False)]
    gt = gt[rng.choice(len(gt), min(a.max_points, len(gt)), replace=False)]
    report = measure(pred, gt, a.threshold)
    report.update(units=a.units, squared_distance_units=a.units + '^2', original_counts=original,
                  sampled_counts=[len(pred), len(gt)], pred_to_gt=T.tolist(),
                  alignment='User-declared common frame/scale or externally supplied transform; no automatic ICP/Sim3',
                  scope='Point-to-point distances on deterministic sampled clouds. Use comparable visible regions; occlusion and density affect these metrics.')
    dump(a.out, report)
    print(report)


if __name__ == '__main__':
    main()
