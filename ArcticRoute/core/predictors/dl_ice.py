from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from ArcticRoute.exceptions import ArcticRouteError
from logging_config import get_logger

from ..interfaces import Predictor

logger = get_logger(__name__)


class DLIcePredictor(Predictor):
    """Placeholder deep-learning ice predictor that mirrors the environmental grid."""

    def __init__(
        self,
        env_path: Path,
        var_name: str,
        *,
        model_path: Optional[Path] = None,
        batch_size: int = 4,
    ) -> None:
        self.env_path = Path(env_path)
        self.var_name = var_name
        self.model_path = Path(model_path) if model_path else None
        self.batch_size = int(batch_size)
        self._dataset: Optional[xr.Dataset] = None

    def prepare(self, base_time_index: int) -> xr.Dataset:
        logger.info(
            "CV placeholder mode: generating ice_prob layer (model_path=%s, batch_size=%s)",
            self.model_path or "N/A",
            self.batch_size,
        )
        try:
            with xr.open_dataset(self.env_path) as env_ds:
                if self.var_name not in env_ds:
                    raise KeyError(f"{self.var_name} missing in {self.env_path}")
                template = env_ds[self.var_name]
                ice_prob = xr.DataArray(
                    np.full(template.shape, 0.5, dtype="float32"),
                    dims=template.dims,
                    coords=template.coords,
                    name="ice_prob",
                    attrs={
                        "placeholder": True,
                        "model_path": str(self.model_path) if self.model_path else "",
                        "batch_size": self.batch_size,
                    },
                )
                dataset = xr.Dataset({"ice_prob": ice_prob})
        except FileNotFoundError as err:
            logger.exception("Environment dataset missing: %s", self.env_path)
            raise ArcticRouteError("ARC-DL-404", "Environment dataset missing", detail=str(err)) from err
        except KeyError as err:
            logger.exception("Environment dataset invalid (variable=%s)", self.var_name)
            raise ArcticRouteError("ARC-DL-002", "Environment dataset invalid", detail=str(err)) from err
        except Exception as err:
            logger.exception("DL ice predictor failed")
            raise ArcticRouteError("ARC-DL-000", "DL ice predictor failed", detail=str(err)) from err
        self._dataset = dataset
        return dataset

    def predict(self, var: Optional[str], t0: int, horizon: Optional[int]) -> Optional[xr.DataArray]:
        if self._dataset is None:
            raise RuntimeError("call prepare() before predict()")
        key = var or "ice_prob"
        if key not in self._dataset:
            return None
        data = self._dataset[key]
        if "time" in data.dims:
            if horizon is None:
                return data.isel(time=t0)
            end = max(t0, 0) + max(horizon, 1)
            return data.isel(time=slice(t0, end))
        return data
