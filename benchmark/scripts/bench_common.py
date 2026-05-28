#!/usr/bin/env python3
"""Shared benchmark harness for all optimization levels."""

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BENCHMARK_DIR / "data"
ONNX_DIR = BENCHMARK_DIR / "onnx"
ENGINE_DIR = BENCHMARK_DIR / "engines"
RESULTS_DIR = BENCHMARK_DIR / "results"

PROMPT = "person"
NUM_WARMUP = 3
NUM_RUNS = 10


def get_image_paths(data_dir: Path = DATA_DIR) -> list[str]:
    """Return sorted list of .jpg paths in the data directory."""
    paths = sorted(str(p) for p in data_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"No .jpg files in {data_dir}. Run download_coco.py first."
        )
    return paths


def log_gpu_state():
    """Print GPU name, driver, memory usage."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
             "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"GPU: {out}")
    except Exception:
        print("GPU: (nvidia-smi not available)")


def benchmark(infer_fn, images: list, num_warmup: int = NUM_WARMUP,
              num_runs: int = NUM_RUNS, label: str = "") -> dict:
    """
    Run a benchmark with warmup and timed passes.

    Args:
        infer_fn: Callable that takes an image path and runs inference.
        images: List of image paths.
        num_warmup: Number of warmup passes over all images.
        num_runs: Number of timed passes over all images.
        label: Label for this benchmark level.

    Returns:
        Dict with timing results.
    """
    n = len(images)
    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  Images: {n}, Warmup: {num_warmup}, Timed runs: {num_runs}")
    log_gpu_state()
    print(f"{'='*60}")

    # Warmup
    print("  Warming up...", end="", flush=True)
    for w in range(num_warmup):
        for img_path in images:
            infer_fn(img_path)
        print(f" {w+1}", end="", flush=True)
    print(" done.")

    # Timed runs
    run_times_ms = []
    for r in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for img_path in images:
            infer_fn(img_path)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_per_image = (t1 - t0) / n * 1000
        run_times_ms.append(ms_per_image)
        print(f"  Run {r+1:2d}: {ms_per_image:.2f} ms/image "
              f"({1000/ms_per_image:.1f} FPS)")

    results = {
        "label": label,
        "num_images": n,
        "num_warmup": num_warmup,
        "num_runs": num_runs,
        "median_ms": round(statistics.median(run_times_ms), 2),
        "mean_ms": round(statistics.mean(run_times_ms), 2),
        "best_ms": round(min(run_times_ms), 2),
        "worst_ms": round(max(run_times_ms), 2),
        "best_fps": round(1000 / min(run_times_ms), 1),
        "median_fps": round(1000 / statistics.median(run_times_ms), 1),
    }

    print(f"\n  Median: {results['median_ms']:.2f} ms/image "
          f"({results['median_fps']:.1f} FPS)")
    print(f"  Best:   {results['best_ms']:.2f} ms/image "
          f"({results['best_fps']:.1f} FPS)")

    return results


def save_results(results: dict, filename: str):
    """Save results to JSON file in the results directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {path}")


def load_results(filename: str) -> dict:
    """Load results from JSON file."""
    path = RESULTS_DIR / filename
    with open(path) as f:
        return json.load(f)


def print_summary_table():
    """Load all results and print a comparison table."""
    result_files = sorted(RESULTS_DIR.glob("level*.json"))
    if not result_files:
        print("No results found.")
        return

    all_results = []
    for rf in result_files:
        with open(rf) as f:
            all_results.append(json.load(f))

    baseline_ms = all_results[0]["median_ms"] if all_results else 1

    print(f"\n{'='*72}")
    print(f"{'Level':<8} {'Description':<28} {'ms/img':>8} {'FPS':>8} {'Speedup':>8}")
    print(f"{'-'*72}")
    for r in all_results:
        speedup = baseline_ms / r["median_ms"] if r["median_ms"] > 0 else 0
        print(f"{r.get('level', '?'):<8} {r['label']:<28} "
              f"{r['median_ms']:>8.1f} {r['median_fps']:>8.1f} {speedup:>7.2f}x")
    print(f"{'='*72}")
