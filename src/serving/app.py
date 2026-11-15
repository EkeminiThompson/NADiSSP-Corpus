"""FastAPI inference service around the exported ONNX model.

    NADIP_MODEL=models/nadip.onnx NADIP_META=data/processed/corpus_meta.json \
        uvicorn --factory src.serving.app:create_app --port 8000

Endpoints
  GET  /health    liveness + model/caches state (used by the Docker healthcheck)
  POST /predict   raw sensor window of one asset -> RUL (cycles), fault probability, three-tier alert.
                  The window is preprocessed exactly as in training: forward-fill, source-fitted IQR clip, z-score.
                  With "offline": true (simulated link loss) or if inference fails, the last cached prediction for
                  the asset is returned with source="edge_cache" while it is younger than the cache limit (72 h).
  POST /feedback  post-maintenance outcome appended to a JSONL buffer for the buffered batch update of Fig. 3.7.
                  Retraining itself is offline (src/evaluation/deployment.py: EWC update); this service never
                  retrains in place.

The alert thresholds are deployment settings, not results: red = fault probability >= threshold or RUL <= critical_rul;
amber = fault probability >= threshold / 2 or RUL <= early_degradation_rul; otherwise green.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..data.preprocess import Normalizer
from ..data.schema import CHANNELS, DATASETS, OFFSETS, N_CHANNELS, to_unified


class PredictRequest(BaseModel):
    asset_id: str
    dataset: str = Field(description="asset-class analogue: one of cmapss, ai4i, w3, espset")
    window: List[List[Optional[float]]] = Field(description="T x C_dataset raw sensor values (null = missing), schema channel order")
    offline: bool = False


class FeedbackRequest(BaseModel):
    asset_id: str
    failed: bool
    rul_cycles: Optional[float] = None
    window: Optional[List[List[Optional[float]]]] = None
    dataset: Optional[str] = None


def _ffill(x: np.ndarray) -> np.ndarray:
    import pandas as pd
    return pd.DataFrame(x).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)


def create_app(model_path: Optional[str] = None, meta_path: Optional[str] = None, threshold: Optional[float] = None) -> FastAPI:
    import onnxruntime as ort
    model_path = model_path or os.environ.get("NADIP_MODEL", "models/nadip.onnx")
    meta_path = meta_path or os.environ.get("NADIP_META", "data/processed/corpus_meta.json")
    thr = float(threshold if threshold is not None else os.environ.get("NADIP_THRESHOLD", 0.5))
    cache_hours = float(os.environ.get("NADIP_CACHE_HOURS", 72))
    buffer_path = Path(os.environ.get("NADIP_FEEDBACK_BUFFER", "feedback_buffer.jsonl"))
    critical_rul = float(os.environ.get("NADIP_CRITICAL_RUL", 30))
    early_rul = float(os.environ.get("NADIP_EARLY_RUL", 70))

    meta = json.load(open(meta_path))
    norms = {k: Normalizer.from_dict(v) for k, v in meta["normalizers"].items()}
    window = int(meta["window"])
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    cache: "OrderedDict[str, dict]" = OrderedDict()

    app = FastAPI(title="NADIP inference service", version="1.0")

    def tier(rul: float, p: float) -> str:
        if p >= thr or rul <= critical_rul:
            return "red"
        if p >= thr / 2 or rul <= early_rul:
            return "amber"
        return "green"

    def from_cache(asset: str) -> Optional[dict]:
        e = cache.get(asset)
        if e is None:
            return None
        age_h = (time.time() - e["ts"]) / 3600.0
        if age_h > cache_hours:
            return None
        return {**e["pred"], "source": "edge_cache", "age_hours": age_h}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": Path(model_path).name, "window": window, "n_channels": N_CHANNELS,
                "cache_entries": len(cache), "cache_limit_hours": cache_hours, "datasets": sorted(norms)}

    @app.post("/predict")
    def predict(req: PredictRequest):
        if req.dataset not in norms:
            raise HTTPException(422, f"unknown or unfitted dataset '{req.dataset}' (known: {sorted(norms)})")
        if req.offline:
            hit = from_cache(req.asset_id)
            if hit is None:
                raise HTTPException(503, "offline and no cached prediction younger than the cache limit")
            return hit
        try:
            raw = np.array([[np.nan if v is None else v for v in row] for row in req.window], dtype=np.float32)
            n_ch = len(CHANNELS[req.dataset])
            if raw.ndim != 2 or raw.shape[1] != n_ch or raw.shape[0] != window:
                raise HTTPException(422, f"window must be {window} x {n_ch} for dataset '{req.dataset}', got {list(raw.shape)}")
            norm = norms[req.dataset]
            z = norm.transform(norm.clip(_ffill(raw)))
            x = to_unified(req.dataset, z[None]).astype(np.float32)
            rul, p = sess.run(None, {"window": x})
            pred = {"rul_cycles": float(rul[0]), "fault_probability": float(p[0]), "alert": tier(float(rul[0]), float(p[0])),
                    "source": "cloud", "missing_fraction": float(np.isnan(raw).mean())}
        except HTTPException:
            raise
        except Exception as e:                                   # inference failure -> edge-cache fallback
            hit = from_cache(req.asset_id)
            if hit is None:
                raise HTTPException(503, f"inference failed and nothing cached: {e}")
            return hit
        cache[req.asset_id] = {"ts": time.time(), "pred": pred}
        cache.move_to_end(req.asset_id)
        while len(cache) > 10_000:
            cache.popitem(last=False)
        return pred

    @app.post("/feedback")
    def feedback(fb: FeedbackRequest):
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(buffer_path, "a") as f:
            f.write(json.dumps({**fb.model_dump(), "ts": time.time()}) + "\n")
        n = sum(1 for _ in open(buffer_path))
        return {"buffered": n}

    return app
