#!/usr/bin/env python
"""Run ON the target device (Raspberry Pi, Jetson, ...) to measure latency of an exported model (Table 4.10).

    python scripts/benchmark_latency.py models/nadip.onnx --window 50 --channels 41
Requires only numpy + onnxruntime. Prints one JSON object; the repo reports only what this prints.
"""
import argparse
import json
import platform
import time

import numpy as np
import onnxruntime as ort

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--channels", type=int, default=41)
    ap.add_argument("--repeats", type=int, default=200)
    a = ap.parse_args()
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, a.window, a.channels).astype(np.float32)
    for _ in range(10):
        s.run(None, {"window": x})
    ts = []
    for _ in range(a.repeats):
        t0 = time.perf_counter(); s.run(None, {"window": x}); ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print(json.dumps({"model": a.model, "mean_ms": ts.mean(), "p95_ms": np.percentile(ts, 95), "throughput_seq_per_s": 1000 / ts.mean(),
                      "machine": platform.machine(), "system": platform.platform(), "meets_120ms": bool(ts.mean() < 120)}, indent=2))
