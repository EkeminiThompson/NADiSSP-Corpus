#!/usr/bin/env python3
"""Load the full NADiSSP 43k+ corpus (source + all target batches)."""
from pathlib import Path
import numpy as np

def load_corpus(root=None):
    if root is None:
        root = Path(__file__).resolve().parents[1] / "processed"
    root = Path(root)
    src = np.load(root / "source" / "full_source_43k.npz")
    batches = sorted((root / "target_batches").glob("batch_*.npz"))
    Xs, ys, fs = [], [], []
    for b in batches:
        d = np.load(b)
        Xs.append(d["X"]); ys.append(d["y"]); fs.append(d["failure"])
    target_X = np.concatenate(Xs)
    target_y = np.concatenate(ys)
    target_f = np.concatenate(fs)
    return {
        "source_X": src["X"], "source_y": src["y"], "source_failure": src["failure"],
        "target_X": target_X, "target_y": target_y, "target_failure": target_f,
        "n_source": len(src["y"]), "n_target": len(target_y),
        "total": len(src["y"]) + len(target_y)
    }

if __name__ == "__main__":
    d = load_corpus()
    print(f"Source {d['n_source']:,}  Target {d['n_target']:,}  Total {d['total']:,}")
