from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("skimage.filters")

pytestmark = pytest.mark.p0

from ArcticRoute.core.predictors.cv_sat import _run_otsu, compute_otsu_mask


def test_run_otsu_returns_nan_for_empty_input():
    img = np.array([], dtype=np.float32)
    mask, threshold = _run_otsu(img)
    assert mask.shape == img.shape
    assert mask.dtype == bool
    assert math.isnan(threshold)


def test_run_otsu_returns_nan_for_constant_input():
    img = np.full((16, 16), 0.5, dtype=np.float32)
    mask, threshold = _run_otsu(img)
    assert mask.shape == img.shape
    assert mask.dtype == bool
    assert not mask.any()
    assert math.isnan(threshold)


def test_run_otsu_returns_nan_for_nan_heavy_input():
    img = np.linspace(0, 1, 12 * 12, dtype=np.float32).reshape(12, 12)
    img.ravel()[60:] = np.nan  # >30% NaN
    mask, threshold = _run_otsu(img)
    assert mask.shape == img.shape
    assert mask.dtype == bool
    assert not mask.any()
    assert math.isnan(threshold)


def test_run_otsu_returns_mask_for_valid_gradient():
    img = np.linspace(-1, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    mask, threshold = _run_otsu(img)
    assert mask.shape == img.shape
    assert mask.dtype == bool
    assert mask.any()
    assert (~mask).any()
    assert np.isfinite(threshold)

    _, _, info = compute_otsu_mask(img, method="gradient", min_samples=1, min_fraction=0.0)
    assert info["reason"] == "ok"


def test_compute_otsu_reports_reasons():
    constant = np.full((8, 8), 0.1, dtype=np.float32)
    _, threshold_constant, info_constant = compute_otsu_mask(
        constant,
        method="constant",
        min_samples=1,
        min_fraction=0.0,
    )
    assert math.isnan(threshold_constant)
    assert info_constant["reason"] == "flat_field"

    nan_heavy = np.linspace(0, 1, 20 * 20, dtype=np.float32).reshape(20, 20)
    nan_heavy.ravel()[200:] = np.nan
    _, threshold_nan, info_nan = compute_otsu_mask(
        nan_heavy,
        method="nan-heavy",
        min_samples=1,
        min_fraction=0.0,
    )
    assert math.isnan(threshold_nan)
    assert info_nan["reason"] == "too_many_nan"
