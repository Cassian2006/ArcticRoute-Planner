import unittest
import numpy as np
import xarray as xr
import torch
import os
import json

# Make sure modules can be imported
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ArcticRoute.core.fusion_adv.dataset import build_patches, make_weak_labels
from ArcticRoute.core.fusion_adv.calibrate import fit_calibrator, apply_calibration, save_calibrator

class TestPhaseK(unittest.TestCase):

    def test_dataset_shape_handling(self):
        """Test build_patches for shape cropping and upscaling."""
        # 1. Test cropping to smallest common shape
        ch1 = xr.DataArray(np.random.rand(100, 120), dims=("y", "x"))
        ch2 = xr.DataArray(np.random.rand(110, 100), dims=("y", "x"))
        labels = xr.DataArray(np.random.randint(0, 2, (110, 120)), dims=("y", "x"))
        channels = {"c1": ch1, "c2": ch2}
        
        # Common shape should be min(100,110) x min(120,100) -> 100x100
        dataset = build_patches(channels, labels, tile=64, stride=64)
        # From a 100x100 image, we get one 64x64 patch
        self.assertEqual(len(dataset), 1)
        x, y, m = dataset[0]
        self.assertEqual(x.shape, (2, 64, 64)) # 2 channels, 64x64 tile

        # 2. Test upscaling if common shape is smaller than tile
        ch1_small = xr.DataArray(np.random.rand(20, 30), dims=("y", "x"))
        ch2_small = xr.DataArray(np.random.rand(25, 25), dims=("y", "x"))
        labels_small = xr.DataArray(np.random.randint(0, 2, (25, 30)), dims=("y", "x"))
        channels_small = {"c1": ch1_small, "c2": ch2_small}
        
        # Common shape 20x25, must be upscaled to at least 32x32 for the tile
        dataset_upscaled = build_patches(channels_small, labels_small, tile=32, stride=32)
        self.assertEqual(len(dataset_upscaled), 1)
        x_up, _, _ = dataset_upscaled[0]
        self.assertEqual(x_up.shape, (2, 32, 32))

    def test_label_masking(self):
        """Test that NaN in weak labels are correctly masked out."""
        ais = xr.DataArray([
            [0.1, 0.9, 0.2],
            [0.8, 0.3, 0.95]
        ], dims=("y", "x"))
        # With tau_q=0.8, threshold is ~0.9 -> (0,1) and (1,2) are positive
        labels = make_weak_labels(ais, None, None, cfg={"tau_pos_q": 0.8})
        
        # Manually set a NaN
        labels.values[0, 0] = np.nan
        
        channels = {"c1": ais}
        dataset = build_patches(channels, labels, tile=3, stride=3)
        self.assertEqual(len(dataset), 1)
        _, y, m = dataset[0]
        
        # y should have 0 where label was NaN, and m should be 0 to mask it
        self.assertEqual(y[0, 0, 0], 0.0)
        self.assertEqual(m[0, 0, 0], 0.0)
        
        # Check a valid positive label
        self.assertEqual(y[0, 0, 1], 1.0)
        self.assertEqual(m[0, 0, 1], 1.0)
        
        # Check total mask sum (5 valid pixels out of 6)
        self.assertEqual(m.sum(), 5)

    def test_calibration_monotonicity(self):
        """Test that isotonic calibration is monotonic."""
        probs = np.linspace(0, 1, 100)
        # Create labels that are generally increasing with probs, but noisy
        labels = (probs + np.random.normal(0, 0.1, 100) > 0.5).astype(float)
        mask = np.ones_like(probs)
        
        model = fit_calibrator(probs, labels, mask, method='isotonic')
        self.assertEqual(model['method'], 'isotonic')
        
        # Check that the fitted y-values are monotonic
        y_calib = np.array(model['y'])
        self.assertTrue(np.all(np.diff(y_calib) >= -1e-9), "Isotonic calibration y-values are not monotonic")

        # Test apply_calibration with a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            calib_path = tmp.name
        try:
            save_calibrator(model, calib_path)
            test_probs = np.array([0.1, 0.5, 0.9])
            calibrated_probs = apply_calibration(test_probs, calib_path)
            self.assertTrue(np.all(np.diff(calibrated_probs) >= -1e-9), "Applied isotonic calibration is not monotonic")
        finally:
            if os.path.exists(calib_path):
                os.remove(calib_path)

if __name__ == '__main__':
    unittest.main()



