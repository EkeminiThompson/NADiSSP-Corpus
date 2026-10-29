"""Deployment simulation (Objective 4; Tables 3.20, 4.10-4.12).

What is *measured* here and what is not
* Packet loss     : measured -- random time-step loss on held-out test windows, imputed, scored.
* Self-learning   : measured on a SIMULATED outage (parametric sensor drift grows with outage length);
                    the online update is a buffered fine-tune with an EWC penalty. Epochs-to-recovery are
                    epochs of that procedure on this data, not field observations.
* Latency         : measured on THIS machine only (ONNX Runtime, CPU). Jetson / Raspberry Pi / A100 figures of
                    Table 4.10 are not measured here; run `scripts/benchmark_latency.py` on those devices.
"""

from __future__ import annotations

import copy
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data.schema import CHANNELS, DATASETS, OFFSETS, role_indices
from ..models.nadip import NADIP, masked_bce, masked_mse
from ..training.data import Bundle
from ..training.runner import DEVICE, predict
from .metrics import auprc, best_f1_threshold, f1_at, rul_rmse


# --------------------------------------------------------------------------- packet loss
def drop_and_impute(X: np.ndarray, rate: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Each time step is lost with probability `rate` (a lost packet drops all channels).
    mode 'ffill'      : forward-fill from the last received step (leading gap: first received step)
    mode 'edge_cache' : replace lost steps by cached training statistics (0 in z-score units)."""
    n, T, C = X.shape
    lost = rng.random((n, T)) < rate
    out = X.copy()
    if mode == "edge_cache":
        out[lost] = 0.0
        return out
    idx = np.where(~lost, np.arange(T)[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    first = np.where(~lost, np.arange(T)[None, :], T).min(1)
    idx = np.where(idx < 0, np.minimum(first, T - 1)[:, None], idx)
    return np.take_along_axis(out, idx[:, :, None], axis=1)


def packet_loss_curve(trained, bundle: Bundle, rows: np.ndarray, thr: float, rates: List[float], seed: int = 42,
                      edge_cache_above: float = 0.30) -> List[Dict]:
    X = bundle.X[torch.from_numpy(rows)].numpy()
    y, has, rul = bundle.fail[rows].numpy(), bundle.has_rul[rows].numpy(), bundle.rul[rows].numpy()
    res, base = [], None
    for p in rates:
        rng = np.random.default_rng(seed)
        mode = "none" if p == 0 else ("edge_cache" if p > edge_cache_above else "ffill")
        Xp = X if p == 0 else drop_and_impute(X, p, "edge_cache" if mode == "edge_cache" else "ffill", rng)
        out = trained.predict(torch.from_numpy(Xp), bundle.dataset[rows])
        a = auprc(y, out["fail_prob"])
        base = a if p == 0 else base
        res.append({"packet_loss": p, "imputation": mode, "auprc": a, "f1": f1_at(y, out["fail_prob"], thr),
                    "rul_rmse": rul_rmse(rul, out["rul"], has),
                    "auprc_drop_pct": float(100 * (base - a) / base) if base and not np.isnan(a) else float("nan")})
    return res


# -------------------------------------------------------------------- outage + EWC recovery
def apply_sensor_drift(X: np.ndarray, dataset_ids: np.ndarray, strength: float, seed: int) -> np.ndarray:
    """Parametric drift of pressure / temperature / vibration channels (z-score units): a per-channel signed
    offset growing along the window, scaled by `strength` (std units). Used only for the outage simulation."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    T = X.shape[1]
    ramp = (0.5 + 0.5 * np.linspace(0, 1, T))[None, :, None]
    for di, name in enumerate(DATASETS):
        rows = np.flatnonzero(dataset_ids == di)
        cols = OFFSETS[name] + role_indices(name, ["pressure", "temperature", "vibration"])
        if len(rows) == 0 or len(cols) == 0:
            continue
        sign = rng.choice([-1.0, 1.0], size=len(cols))
        out[np.ix_(rows, np.arange(T), cols)] += strength * sign[None, None, :] * ramp
    return out


def ewc_fisher(model: NADIP, bundle: Bundle, rows: np.ndarray, n: int = 256, seed: int = 0) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    rows = rng.choice(rows, size=min(n, len(rows)), replace=False)
    fisher = {k: torch.zeros_like(p) for k, p in model.named_parameters() if p.requires_grad}
    model.eval()
    for r in rows:
        model.zero_grad()
        t = torch.tensor([r])
        out = model(bundle.X[t].to(DEVICE))
        loss = masked_mse(out["rul"] / model.rul_scale, bundle.rul[t].to(DEVICE) / model.rul_scale,
                          bundle.has_rul[t].to(DEVICE)) + masked_bce(out["fail_logit"], bundle.fail[t].to(DEVICE),
                                                                      torch.ones(1, dtype=torch.bool, device=DEVICE))
        loss.backward()
        for k, p in model.named_parameters():
            if p.grad is not None and k in fisher:
                fisher[k] += p.grad.detach() ** 2 / len(rows)
    model.zero_grad()
    return fisher


def simulate_outage(trained, bundle: Bundle, hours: float, cfg: Dict, seed: int = 42, tol: float = 0.02) -> Dict:
    """Outage of `hours`: sensor drift accumulates; on reconnection, fine-tune on the buffered (drifted) stream with an
    EWC penalty anchoring the weights that mattered before the outage. The buffer holds only what the 72-hour edge
    cache can hold -- beyond that only the most recent 10% survives, so recovery relies on EWC (Table 4.12 'EWC only')."""
    dep = cfg["deployment"]
    if trained.model is None:
        return {"note": "needs a NADIP transformer"}
    rows_te = bundle.idx("test", 1)
    rows_buf = bundle.idx("val", 1)
    rows_src = bundle.idx("train", 0)
    if min(len(rows_te), len(rows_buf), len(rows_src)) == 0:
        return {"note": "needs target val/test rows and source train rows"}
    strength = dep["drift_max_std"] * min(1.0, np.log1p(hours) / np.log1p(dep["outage_windows"]["quarter"]))
    y = bundle.fail[rows_te].numpy()
    use_w = trained.variant.use_weibull
    clean = auprc(y, predict(trained.model, bundle.X[rows_te], use_w)["fail_prob"])
    Xte = torch.from_numpy(apply_sensor_drift(bundle.X[rows_te].numpy(), bundle.dataset[rows_te], strength, seed))
    Xbuf = torch.from_numpy(apply_sensor_drift(bundle.X[rows_buf].numpy(), bundle.dataset[rows_buf], strength, seed + 1))
    at_reconnect = auprc(y, predict(trained.model, Xte, use_w)["fail_prob"])

    rng = np.random.default_rng(seed)
    cache_hours = dep.get("cache_limit_hours", 72)
    keep = len(rows_buf) if hours <= cache_hours else max(4, int(0.1 * len(rows_buf)))
    sel = rng.permutation(len(rows_buf))[:keep]
    model = copy.deepcopy(trained.model)
    fisher = ewc_fisher(model, bundle, rows_src, seed=seed)
    anchor = {k: p.detach().clone() for k, p in model.named_parameters() if k in fisher}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=dep.get("recovery_lr", 1e-4))
    bs = cfg["training"]["batch_size"]
    curve, recovered_at = [], None
    for ep in range(1, dep["recovery_max_epochs"] + 1):
        model.train()
        for i in range(0, len(sel), bs):
            b = sel[i:i + bs]
            if len(b) < 2:
                continue
            r = torch.from_numpy(rows_buf[b])
            out = model(Xbuf[b].to(DEVICE))
            loss = masked_mse(out["rul"] / model.rul_scale, bundle.rul[r].to(DEVICE) / model.rul_scale, bundle.has_rul[r].to(DEVICE)) \
                + masked_bce(out["fail_logit"], bundle.fail[r].to(DEVICE), torch.ones(len(b), dtype=torch.bool, device=DEVICE))
            pen = sum((fisher[k] * (p - anchor[k]) ** 2).sum() for k, p in model.named_parameters() if k in fisher)
            loss = loss + 0.5 * dep["ewc_lambda"] * pen
            opt.zero_grad(); loss.backward(); opt.step()
        a = auprc(y, predict(model, Xte, use_w)["fail_prob"])
        curve.append({"epoch": ep, "auprc": a})
        if recovered_at is None and not np.isnan(a) and not np.isnan(clean) and a >= clean - tol:
            recovered_at = ep
    return {"outage_hours": hours, "drift_strength_std": float(strength), "buffer_rows": int(len(sel)),
            "auprc_before_outage": clean, "auprc_at_reconnection": at_reconnect, "recovery_curve": curve,
            "epochs_to_recovery": recovered_at, "auprc_after_recovery": curve[-1]["auprc"] if curve else None,
            "recovery_tolerance": tol, "evaluated_with": "transformer only (XGBoost stage not updated online)"}


# ---------------------------------------------------------------------------------- ONNX
class _Export(nn.Module):
    def __init__(self, model: NADIP, use_weibull: bool):
        super().__init__()
        self.m, self.use_weibull = model, use_weibull

    def forward(self, x):
        out = self.m(x)
        rul = self.m.weibull.rul_median(out["z"]) if self.use_weibull else out["rul"]
        return rul.clamp(0, self.m.rul_scale), torch.sigmoid(out["fail_logit"])


def export_onnx(trained, path: Path, window: int, n_channels: int, opset: int = 17) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    m = _Export(copy.deepcopy(trained.model).cpu().eval(), trained.variant.use_weibull).eval()
    x = torch.randn(2, window, n_channels)
    torch.onnx.export(m, (x,), str(path), input_names=["window"], output_names=["rul", "fail_prob"],
                      dynamic_axes={"window": {0: "batch"}, "rul": {0: "batch"}, "fail_prob": {0: "batch"}},
                      opset_version=opset, dynamo=False)
    return path


def quantize_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def ort_session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def benchmark_onnx(path: Path, window: int, n_channels: int, repeats: int = 200, warmup: int = 10) -> Dict:
    sess = ort_session(path)
    x = np.random.randn(1, window, n_channels).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"window": x})
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); sess.run(None, {"window": x}); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"model": Path(path).name, "mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)),
            "throughput_seq_per_s": float(1000.0 / ts.mean()), "repeats": repeats, "batch": 1, "threads": 1,
            "host": {"machine": platform.machine(), "processor": platform.processor() or "unknown",
                     "cpu_count": __import__("os").cpu_count(), "system": platform.platform()},
            "target_ms": 120, "meets_120ms_on_this_host": bool(ts.mean() < 120),
            "note": "Measured on this host only. Not a measurement of Jetson / Raspberry Pi / A100."}


def onnx_parity(trained, path: Path, X: torch.Tensor) -> Dict:
    m = _Export(copy.deepcopy(trained.model).cpu().eval(), trained.variant.use_weibull).eval()
    with torch.no_grad():
        r, f = m(X)
    o = ort_session(path).run(None, {"window": X.numpy().astype(np.float32)})
    return {"max_abs_diff_rul": float(np.abs(r.numpy() - o[0]).max()), "max_abs_diff_fail": float(np.abs(f.numpy() - o[1]).max())}


def onnx_auprc(path: Path, bundle: Bundle, rows: np.ndarray) -> float:
    sess = ort_session(path)
    X = bundle.X[torch.from_numpy(rows)].numpy().astype(np.float32)
    p = np.concatenate([sess.run(None, {"window": X[i:i + 256]})[1] for i in range(0, len(X), 256)])
    return auprc(bundle.fail[rows].numpy(), p)
