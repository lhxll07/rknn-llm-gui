"""Verify direct open NPU GEMM, including actual Qwen3.5 projection weights."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from native_npu.backend import OpenNPU
from native_converter.weights import Weights
from native_converter.qwen35 import transform

UPSTREAM = "https://github.com/fukumori/iwagumi"
REVISION = "8fafdbd28cce6eabb0ec2072bb549beb1035a141"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Optional previously verified Qwen3.5-0.8B source directory")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    rng = np.random.default_rng(20261007)
    results = []
    weights = Weights(args.model) if args.model else None
    try:
        with OpenNPU(args.library) as backend:
            def check(label, w, rows):
                n, k = w.shape
                a = rng.uniform(-0.25, 0.25, size=(rows, k)).astype("<f2")
                w = np.ascontiguousarray(w, dtype="<f2")
                start = time.monotonic()
                got = backend.matmul(a, w)
                elapsed = time.monotonic() - start
                # CPU oracle accumulates exactly the FP16 operands supplied
                # to hardware in FP32, so packing/compiler errors are visible.
                expected = a.astype("<f4") @ w.astype("<f4").T
                error = np.abs(got - expected)
                bound = 5e-4 + 2e-3 * np.abs(expected)
                valid = bool(np.isfinite(got).all() and np.all(error <= bound))
                result = {"name": label, "M": rows, "K": k, "N": n, "valid": valid,
                          "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
                          "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12)),
                          "seconds_including_pack_and_allocation": elapsed,
                          "result_sha256": hashlib.sha256(got.tobytes()).hexdigest()}
                print(json.dumps(result, ensure_ascii=False), flush=True)
                results.append(result)
                if not valid:
                    raise RuntimeError(f"NPU/CPU numeric mismatch: {label}, M={rows}")
            for rows in (1, 2, 16):
                check("synthetic-fp16", rng.uniform(-0.25, 0.25, size=(64, 64)), rows)
            if weights:
                specs = [
                    ("blk.0.attn_qkv.weight", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
                    ("blk.0.ffn_gate.weight", "model.language_model.layers.0.mlp.gate_proj.weight"),
                    ("blk.0.ffn_down.weight", "model.language_model.layers.0.mlp.down_proj.weight"),
                    ("blk.3.attn_q.weight", "model.language_model.layers.3.self_attn.q_proj.weight"),
                ]
                for name, source in specs:
                    w = transform(weights.get(source), name)
                    for rows in (1, 16):
                        check(name, w, rows)
            maps = Path("/proc/self/maps").read_text()
            vendor = sorted({line.split()[-1] for line in maps.splitlines()
                             if "librknnrt" in line or "librkllmrt" in line})
            if vendor:
                raise RuntimeError(f"Unexpected vendor runtime mapping: {vendor}")
            report = {"date": "2026-10-07", "architecture": platform.machine(), "kernel": platform.release(),
                      "soc": backend.caps.soc.decode(), "card_index": backend.caps.card_index,
                      "open_engine": UPSTREAM, "source_revision": REVISION,
                      "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
                      "numpy_version": np.__version__, "official_runtime_mappings": vendor,
                      "cached_rkllm_program_used": False, "toolkit_used": False,
                      "successful_matmul_submits": backend.submits, "results": results,
                      "scope": "projection kernels only; synthetic activations; complete language model not executed",
                      "oracle": "FP16 operands accumulated in CPU FP32; atol=0.0005, rtol=0.002"}
        with args.report.open("x") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
    finally:
        if weights:
            weights.close()


if __name__ == "__main__":
    main()
