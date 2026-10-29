"""Staged training (§3.10.3) for one (variant, seed).

  Stage 1  backbone initialisation: random, or compatible pretrained weights if model.pretrained_init is set
  Stage 2  SimCLR on unlabelled Nigerian-adapted target windows           (use_simclr)
  Stage 3  joint adversarial + supervised (+ contrastive + survival)     (always; GRL only if use_grl)
  Stage 4  Weibull-PH head fine-tuning, encoder frozen                    (use_weibull)
  Stage 5  XGBoost on frozen embeddings + interpretable features          (use_xgb)
  Stage 6  ONNX export (scripts/export_onnx.py / export_onnx())

Each stage checkpoints to <ckpt>/<variant>/seed<k>/ and is skipped when its file exists (resume).
Source rows carry the labels used by L_task; target rows are used unlabelled (L_domain, L_contrastive)
unless training.target_label_in_task_loss is set.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.schema import DATASETS, DATASET_ID
from ..evaluation.features import window_features
from ..models.nadip import NADIP, ganin_lambda, load_compatible, masked_bce, masked_mse, nt_xent_loss
from .data import Bundle
from .variants import Variant
from .views import ViewMaker, standard_augment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: Dict, n_channels: int) -> NADIP:
    return NADIP(n_channels, cfg["model"], rul_scale=float(cfg["data"]["rul_cap"])).to(DEVICE)


# ------------------------------------------------------------------------------ inference
@torch.no_grad()
def predict(model: NADIP, X: torch.Tensor, use_weibull: bool, batch: int = 256) -> Dict[str, np.ndarray]:
    model.eval()
    zs, rl, rw, fl = [], [], [], []
    for i in range(0, len(X), batch):
        out = model(X[i:i + batch].to(DEVICE))
        zs.append(out["z"].cpu()); rl.append(out["rul"].cpu()); fl.append(torch.sigmoid(out["fail_logit"]).cpu())
        rw.append(model.weibull.rul_median(out["z"]).cpu())
    cap = model.rul_scale
    rul_lin = torch.cat(rl).clamp(0, cap).numpy()
    rul_wb = torch.cat(rw).clamp(0, cap).numpy()
    return {"z": torch.cat(zs).numpy(), "rul_linear": rul_lin, "rul_weibull": rul_wb,
            "rul": rul_wb if use_weibull else rul_lin, "fail_prob": torch.cat(fl).numpy()}


# -------------------------------------------------------------------------------- results
@dataclass
class Trained:
    variant: Variant
    seed: int
    model: Optional[NADIP] = None
    xgb: Optional[Dict] = None
    history: Dict[str, List] = field(default_factory=dict)
    info: Dict = field(default_factory=dict)
    kind: str = "nadip"
    baseline: Optional[object] = None                       # xgboost / lstm baselines

    def predict(self, X: torch.Tensor, dataset_ids: np.ndarray) -> Dict[str, np.ndarray]:
        return predict_trained(self, X, dataset_ids)


def predict_trained(t: Trained, X: torch.Tensor, dataset_ids: np.ndarray) -> Dict[str, np.ndarray]:
    if t.kind != "nadip":
        return t.baseline.predict(X, dataset_ids)
    out = predict(t.model, X, t.variant.use_weibull)
    out["transformer_rul"], out["transformer_fail_prob"] = out["rul"].copy(), out["fail_prob"].copy()
    if t.xgb is not None:
        feats, _ = window_features(X.numpy(), dataset_ids)
        Z = np.concatenate([out["z"], feats, out["rul"][:, None]], axis=1)
        w = t.xgb["weight"]
        if t.xgb.get("reg") is not None:
            out["rul"] = (1 - w) * out["rul"] + w * np.clip(t.xgb["reg"].predict(Z), 0, t.model.rul_scale)
        if t.xgb.get("clf") is not None:
            out["fail_prob"] = (1 - w) * out["fail_prob"] + w * t.xgb["clf"].predict_proba(Z)[:, 1]
    return out


# ----------------------------------------------------------------------------- helpers
def _batches(idx: np.ndarray, bs: int, rng: np.random.Generator, drop_last: bool = True):
    idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        if drop_last and len(b) < max(2, bs // 2):
            continue
        yield b


def _task_losses(model: NADIP, out: Dict, bundle: Bundle, rows: np.ndarray, use_weibull: bool, lam_surv: float):
    t = torch.from_numpy(rows)
    scale = model.rul_scale
    m_rul = bundle.has_rul[t].to(DEVICE)
    rul = bundle.rul[t].to(DEVICE)
    fail = bundle.fail[t].to(DEVICE)
    ones = torch.ones(len(rows), dtype=torch.bool, device=DEVICE)
    l_rul = masked_mse(out["rul"] / scale, rul / scale, m_rul)
    l_cls = masked_bce(out["fail_logit"], fail, ones)
    l_surv = out["z"].sum() * 0.0
    if use_weibull and m_rul.any():
        ev = bundle.event[t].to(DEVICE)
        l_surv = lam_surv * model.weibull.nll(rul[m_rul].clamp(min=1.0), ev[m_rul], out["z"][m_rul])
    return l_rul, l_cls, l_surv


def _eval_source_val(model: NADIP, bundle: Bundle, rows: np.ndarray, use_weibull: bool) -> float:
    """Validation criterion: normalised RUL MSE + fault BCE on held-out *source* validation rows."""
    if len(rows) == 0:
        return float("nan")
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            b = rows[i:i + 256]
            out = model(bundle.X[torch.from_numpy(b)].to(DEVICE))
            l_rul, l_cls, _ = _task_losses(model, out, bundle, b, False, 0.0)
            tot += float((l_rul + l_cls).item()) * len(b); n += len(b)
    return tot / max(n, 1)


def _disc_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(((logits > 0).float() == labels).float().mean().item())


# ---------------------------------------------------------------------------- stage 2
def stage2_simclr(model: NADIP, bundle: Bundle, vm: ViewMaker, cfg: Dict, rows_t: np.ndarray, seed: int) -> List[Dict]:
    c = cfg["training"]["stage2_simclr"]
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(list(model.backbone.parameters()) + list(model.projector.parameters()), lr=c["lr"])
    bs = cfg["training"]["batch_size"]
    weights = np.ones(len(rows_t))
    for name, w in (c.get("sampling_weight") or {}).items():
        weights[bundle.dataset[rows_t] == DATASET_ID[name]] = float(w)
    weights /= weights.sum()
    per_epoch = min(len(rows_t), int(c.get("rows_per_epoch", 4096)))
    hist = []
    for ep in range(c["epochs"]):
        model.train()
        rows = rng.choice(rows_t, size=per_epoch, replace=len(rows_t) < per_epoch, p=weights)
        vb = vm.view_b(rows)                                                  # fresh physics views, refreshed every epoch
        order = rng.permutation(len(rows))
        losses = []
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            if len(sel) < 4:
                continue
            xa = standard_augment(bundle.X[torch.from_numpy(rows[sel])], c)
            xb = standard_augment(vb[sel], c)
            za = model(xa.to(DEVICE))["proj"]; zb = model(xb.to(DEVICE))["proj"]
            loss = nt_xent_loss(za, zb, c["temperature"])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        hist.append({"epoch": ep + 1, "nt_xent": float(np.mean(losses)) if losses else float("nan")})
    return hist


# ---------------------------------------------------------------------------- stage 3
def stage3_joint(model: NADIP, bundle: Bundle, vm: Optional[ViewMaker], cfg: Dict, variant: Variant,
                 rows_s: np.ndarray, rows_t: np.ndarray, rows_val: np.ndarray, seed: int) -> List[Dict]:
    c, tr = cfg["training"]["stage3_joint"], cfg["training"]
    rng = np.random.default_rng(seed + 1)
    main_p, disc_p = model.param_groups()
    opt = torch.optim.Adam([p for p in main_p if p.requires_grad], lr=c["lr_start"])
    opt_d = torch.optim.Adam(disc_p, lr=c["lr_start"])
    bs = tr["batch_size"]
    steps_per_epoch = max(1, len(rows_s) // bs)
    total = c["epochs"] * steps_per_epoch
    step, best, best_state, bad, hist = 0, float("inf"), None, 0, []
    use_target = variant.uses_target and len(rows_t) > 0
    pre = cfg["model"].get("pretrained_init") and variant.use_pretrained
    for ep in range(c["epochs"]):
        lr = c["lr_end"] + 0.5 * (c["lr_start"] - c["lr_end"]) * (1 + math.cos(math.pi * ep / max(c["epochs"] - 1, 1)))
        for o in (opt, opt_d):
            for g in o.param_groups:
                g["lr"] = lr
        if pre:                                                               # §3.10.2: lower layers frozen early
            model.backbone.set_frozen_lower(cfg["model"]["freeze_lower_fraction"], ep < cfg["model"]["freeze_epochs"])
        model.train()
        vb_cache = None
        if variant.use_simclr and use_target:
            sub = rng.choice(rows_t, size=min(len(rows_t), steps_per_epoch * bs), replace=False)
            vb_cache = (sub, vm.view_b(sub))
        agg = {"task_rul": 0.0, "task_cls": 0.0, "domain": 0.0, "contrastive": 0.0, "survival": 0.0, "disc_acc": 0.0}
        nb = 0
        for b_s in _batches(rows_s, bs, rng):
            lam = ganin_lambda(step / max(total, 1), c["lambda_d_max"]) if variant.use_grl else 0.0
            model.set_grl_lambda(lam)
            xs = bundle.X[torch.from_numpy(b_s)]
            if use_target:
                b_t = rng.choice(rows_t, size=len(b_s), replace=len(rows_t) < len(b_s))
                x = torch.cat([xs, bundle.X[torch.from_numpy(b_t)]]).to(DEVICE)
            else:
                x = xs.to(DEVICE)
            out = model(x)
            n_s = len(b_s)
            out_s = {k: v[:n_s] for k, v in out.items()}
            l_rul, l_cls, l_surv = _task_losses(model, out_s, bundle, b_s, variant.use_weibull, c["lambda_survival"])
            loss = c["rul_weight"] * l_rul + c["cls_weight"] * l_cls + l_surv
            l_dom = l_con = torch.zeros((), device=DEVICE)
            acc = 0.5
            if use_target and variant.use_grl:
                dl = torch.cat([torch.zeros(n_s), torch.ones(len(x) - n_s)]).to(DEVICE)
                l_dom = F.binary_cross_entropy_with_logits(out["domain_logit"], dl)
                loss = loss + l_dom
                acc = _disc_accuracy(out["domain_logit"].detach(), dl)
            if use_target and variant.use_simclr and vb_cache is not None:
                sel = rng.choice(len(vb_cache[0]), size=min(len(b_s), len(vb_cache[0])), replace=False)
                sc = cfg["training"]["stage2_simclr"]
                xa = standard_augment(bundle.X[torch.from_numpy(vb_cache[0][sel])], sc).to(DEVICE)
                xb = standard_augment(vb_cache[1][sel], sc).to(DEVICE)
                l_con = nt_xent_loss(model(xa)["proj"], model(xb)["proj"], sc["temperature"])
                loss = loss + c["lambda_c"] * l_con
            opt.zero_grad(); opt_d.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); opt_d.step()
            step += 1; nb += 1
            for k, v in (("task_rul", l_rul), ("task_cls", l_cls), ("domain", l_dom), ("contrastive", l_con),
                         ("survival", l_surv)):
                agg[k] += float(v.item())
            agg["disc_acc"] += acc
        rec = {"epoch": ep + 1, "lr": lr, "grl_lambda": lam if nb else 0.0, **{k: v / max(nb, 1) for k, v in agg.items()}}
        rec["val_source"] = _eval_source_val(model, bundle, rows_val, variant.use_weibull)
        hist.append(rec)
        v = rec["val_source"]
        if not math.isnan(v):
            if v < best - 1e-6:
                best, bad, best_state = v, 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= c["early_stop_patience"]:
                    rec["stopped_early"] = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


# ---------------------------------------------------------------------------- stage 4
def stage4_weibull(model: NADIP, bundle: Bundle, cfg: Dict, rows_s: np.ndarray, rows_val: np.ndarray, seed: int) -> List[Dict]:
    c = cfg["training"]["stage4_weibull"]
    sel = lambda rows: rows[bundle.has_rul[torch.from_numpy(rows)].numpy()]
    tr, va = sel(rows_s), sel(rows_val)
    if len(tr) == 0:
        return [{"skipped": "no labelled RUL rows in training data"}]
    model.eval()
    with torch.no_grad():
        z_tr = predict(model, bundle.X[torch.from_numpy(tr)], True)["z"]
        z_va = predict(model, bundle.X[torch.from_numpy(va)], True)["z"] if len(va) else None
    z_tr = torch.from_numpy(z_tr).to(DEVICE)
    t_tr, e_tr = bundle.rul[torch.from_numpy(tr)].to(DEVICE), bundle.event[torch.from_numpy(tr)].to(DEVICE)
    params = list(model.weibull.parameters())
    opt = torch.optim.Adam(params, lr=c["lr"])
    loss_fn = model.weibull.nll if c["loss"] == "weibull" else (lambda t, e, z: model.weibull.cox_partial_nll(t, e, z))
    hist, rng, bs = [], np.random.default_rng(seed + 2), cfg["training"]["batch_size"]
    for ep in range(c["epochs"]):
        losses = []
        for b in _batches(np.arange(len(tr)), bs, rng, drop_last=False):
            l = loss_fn(t_tr[b].clamp(min=1.0), e_tr[b], z_tr[b])
            opt.zero_grad(); l.backward(); opt.step(); losses.append(l.item())
        rec = {"epoch": ep + 1, "nll": float(np.mean(losses)), "beta": float(model.weibull.beta.item()),
               "eta": float(model.weibull.eta.item())}
        if z_va is not None:
            with torch.no_grad():
                tv = bundle.rul[torch.from_numpy(va)].to(DEVICE).clamp(min=1.0)
                ev = bundle.event[torch.from_numpy(va)].to(DEVICE)
                rec["val_nll"] = float(model.weibull.nll(tv, ev, torch.from_numpy(z_va).to(DEVICE)).item())
        hist.append(rec)
    return hist


# ---------------------------------------------------------------------------- stage 5
def stage5_xgboost(model: NADIP, bundle: Bundle, cfg: Dict, rows_s: np.ndarray, rows_val: np.ndarray,
                   variant: Variant, seed: int) -> Optional[Dict]:
    import xgboost as xgb
    c = cfg["training"]["stage5_xgboost"]

    def feats(rows):
        X = bundle.X[torch.from_numpy(rows)]
        p = predict(model, X, variant.use_weibull)
        f, _ = window_features(X.numpy(), bundle.dataset[rows])
        return np.concatenate([p["z"], f, p["rul"][:, None]], axis=1)

    Ztr, Zva = feats(rows_s), feats(rows_val) if len(rows_val) else None
    kw = dict(n_estimators=c["n_estimators"], max_depth=c["max_depth"], random_state=seed, n_jobs=2,
              tree_method="hist", learning_rate=0.1)
    es = c["early_stopping_rounds"]
    out: Dict = {"weight": float(c.get("ensemble_weight", 0.5)), "reg": None, "clf": None}
    m = bundle.has_rul[torch.from_numpy(rows_s)].numpy()
    if m.sum() > 10:
        y = bundle.rul[torch.from_numpy(rows_s)].numpy()
        reg = xgb.XGBRegressor(early_stopping_rounds=es if Zva is not None else None, **kw)
        ev = None
        if Zva is not None:
            mv = bundle.has_rul[torch.from_numpy(rows_val)].numpy()
            if mv.sum() > 5:
                ev = [(Zva[mv], bundle.rul[torch.from_numpy(rows_val)].numpy()[mv])]
        if ev is None:
            reg = xgb.XGBRegressor(**kw)
        reg.fit(Ztr[m], y[m], eval_set=ev, verbose=False)
        out["reg"] = reg
    yf = bundle.fail[torch.from_numpy(rows_s)].numpy()
    if 0 < yf.sum() < len(yf):
        yv = bundle.fail[torch.from_numpy(rows_val)].numpy() if Zva is not None else None
        ev = [(Zva, yv)] if (yv is not None and 0 < yv.sum() < len(yv)) else None
        clf = xgb.XGBClassifier(early_stopping_rounds=es if ev else None, **kw)
        clf.fit(Ztr, yf, eval_set=ev, verbose=False)
        out["clf"] = clf
    return out


# ----------------------------------------------------------------------------- driver
def _stage_path(root: Path, name: str) -> Path:
    return root / f"{name}.pt"


def train_variant(cfg: Dict, bundle: Bundle, meta: Dict, variant: Variant, seed: int, ckpt_root: Optional[Path] = None,
                  force: bool = False, log=print) -> Trained:
    """Train one variant for one seed. Baselines are dispatched to src/training/baselines.py."""
    if variant.kind != "nadip":
        from .baselines import train_baseline
        return train_baseline(cfg, bundle, variant, seed, log=log)

    seed_everything(seed)
    root = (ckpt_root / variant.name / f"seed{seed}") if ckpt_root else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, bundle.C)
    t = Trained(variant, seed, model=model)
    ds = list(variant.datasets) if variant.datasets else None
    rows_s = bundle.idx("train", 0, ds)
    rows_val = bundle.idx("val", 0, ds)
    rows_t = bundle.idx("train", 1, ds) if variant.uses_target else np.empty(0, dtype=int)
    n_source_only = int(len(rows_s))
    if cfg["data"].get("target_label_in_task_loss") and len(rows_t):
        # optional: the 30% of target windows whose labels were retained join the supervised loss (default: off, §3.10.1)
        rows_s = np.concatenate([rows_s, rows_t[bundle.label_avail[rows_t]]])
    t.info = {"n_source_train": n_source_only, "n_supervised_train": int(len(rows_s)), "n_target_train": int(len(rows_t)), "n_source_val": int(len(rows_val)),
              "pretrained": False}
    if len(rows_s) == 0:
        raise RuntimeError(f"{variant.name}: no labelled source training rows")

    # Stage 1
    pre = cfg["model"].get("pretrained_init")
    if variant.use_pretrained and pre:
        state = torch.load(pre, map_location="cpu")
        state = state.get("model_state", state) if isinstance(state, dict) else state
        rep = load_compatible(model.backbone, state)
        t.info["pretrained"] = rep["loaded"] > 0
        t.info["pretrained_report"] = rep
        log(f"[{variant.name}] stage 1: loaded {rep['loaded']} tensors from {pre} ({rep['shape_mismatch']} shape mismatches)")
    elif variant.use_pretrained:
        log(f"[{variant.name}] stage 1: model.pretrained_init is not set -> random initialisation "
            f"(no Kinetics-400 weights are used)")
    vm = ViewMaker(bundle, meta, cfg["augmentation"], seed) if variant.use_simclr and len(rows_t) else None

    def run(name, fn):
        p = _stage_path(root, name) if root else None
        if p is not None and p.exists() and not force:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["model_state"]); t.history[name] = blob.get("history", [])
            log(f"[{variant.name}|seed {seed}] {name}: resumed from {p.name}")
            return
        t0 = time.time()
        t.history[name] = fn()
        log(f"[{variant.name}|seed {seed}] {name}: done in {time.time() - t0:.1f}s")
        if p is not None:
            torch.save({"model_state": model.state_dict(), "history": t.history[name]}, p)

    if variant.use_simclr and len(rows_t):
        run("stage2_simclr", lambda: stage2_simclr(model, bundle, vm, cfg, rows_t, seed))
    run("stage3_joint", lambda: stage3_joint(model, bundle, vm, cfg, variant, rows_s, rows_t, rows_val, seed))
    if variant.use_weibull:
        run("stage4_weibull", lambda: stage4_weibull(model, bundle, cfg, rows_s, rows_val, seed))
    if variant.use_xgb:
        import joblib
        xp = (root / "stage5_xgb.joblib") if root else None
        if xp is not None and xp.exists() and not force:
            t.xgb = joblib.load(xp)
            log(f"[{variant.name}|seed {seed}] stage5_xgboost: resumed")
        else:
            t.xgb = stage5_xgboost(model, bundle, cfg, rows_s, rows_val, variant, seed)
            if xp is not None and t.xgb is not None:
                joblib.dump(t.xgb, xp)
    return t
