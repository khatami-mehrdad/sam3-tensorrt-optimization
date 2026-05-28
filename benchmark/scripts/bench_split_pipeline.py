#!/usr/bin/env python3
"""Benchmark FPS and memory for the split-module TRT pipeline.

Tests 6 configurations:
  1. Vision Encoder only
  2. Text Encoder only
  3. Decoder only
  4. Full sequential (VE + TE + Dec)
  5. Pipelined: (VE + TE for frame N) || (Dec for frame N-1)
  6. Text cached + pipelined: VE(frame N) || Dec(frame N-1)

Usage:
    python3 bench_split_pipeline.py
    python3 bench_split_pipeline.py --warmup 3 --runs 10
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench_common import get_image_paths, PROMPT, RESULTS_DIR

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SPLIT_ENGINE_DIR = BENCHMARK_DIR / "engines" / "split"

NUM_WARMUP = 5
NUM_RUNS = 20


class TRTModule:
    """TRT engine runner with GPU-native tensor support (no CPU round-trips)."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.output_names = []
        self.output_meta = {}
        self.input_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                self.output_meta[name] = {"dtype": dtype}
            else:
                self.input_names.append(name)

    def infer_gpu(self, inputs: dict) -> dict:
        """Run inference with GPU tensors in and out (zero CPU copies between modules).

        Args:
            inputs: dict of {name: torch.Tensor on CUDA} or {name: np.ndarray}
                    numpy arrays are copied to GPU; torch CUDA tensors are used directly.
        Returns:
            dict of {name: torch.Tensor on CUDA}
        """
        buffers = {}

        for name, data in inputs.items():
            if isinstance(data, torch.Tensor):
                t = data.cuda().contiguous()
            else:
                t = torch.from_numpy(data).cuda().contiguous()
            buffers[name] = t
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.output_meta[name]["dtype"]
            torch_dtype = torch.from_numpy(np.array([], dtype=dtype)).dtype
            buf = torch.zeros(shape, dtype=torch_dtype, device="cuda")
            buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return {name: buffers[name] for name in self.output_names}

    def infer(self, inputs: dict) -> dict:
        """Run inference with numpy in/out (legacy, used for fidelity checks)."""
        gpu_inputs = {}
        for name, data in inputs.items():
            if isinstance(data, torch.Tensor):
                gpu_inputs[name] = data
            else:
                gpu_inputs[name] = torch.from_numpy(data).cuda().contiguous()

        gpu_out = self.infer_gpu(gpu_inputs)
        return {name: t.cpu().numpy().astype(np.float32) for name, t in gpu_out.items()}


def get_gpu_memory_mb() -> float:
    """Get actual GPU memory used by this process via nvidia-smi."""
    import subprocess
    pid = str(os.getpid())
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True
        )
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] == pid:
                return float(parts[1])
    except Exception:
        pass
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_memory_stats():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def prepare_inputs(images: list[str]):
    """Prepare preprocessed inputs for benchmarking."""
    from transformers.models.sam3 import Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    all_pixel_values = []
    for img_path in images:
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt")
        all_pixel_values.append(inputs["pixel_values"].numpy())

    # Text inputs are the same for all images (same prompt)
    dummy_img = Image.new("RGB", (224, 224))
    tok = processor(images=dummy_img, text=PROMPT, return_tensors="pt")
    input_ids = tok["input_ids"].numpy().astype(np.int64)
    attention_mask = tok["attention_mask"].numpy().astype(np.int64)

    return all_pixel_values, input_ids, attention_mask


def bench_vision_encoder(ve: TRTModule, pixel_values_list: list,
                         num_warmup: int, num_runs: int) -> dict:
    """Test 1: Vision Encoder only."""
    n = len(pixel_values_list)
    # Pre-upload to GPU
    gpu_inputs = [torch.from_numpy(pv).cuda().contiguous() for pv in pixel_values_list]

    for _ in range(num_warmup):
        for pv in gpu_inputs:
            ve.infer_gpu({"images": pv})

    reset_memory_stats()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for pv in gpu_inputs:
            ve.infer_gpu({"images": pv})
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times)), "peak_gpu_mb": get_gpu_memory_mb()}


def bench_text_encoder(te: TRTModule, input_ids, attention_mask,
                       num_warmup: int, num_runs: int) -> dict:
    """Test 2: Text Encoder only."""
    gpu_ids = torch.from_numpy(input_ids).cuda().contiguous()
    gpu_mask = torch.from_numpy(attention_mask).cuda().contiguous()

    for _ in range(num_warmup):
        te.infer_gpu({"input_ids": gpu_ids, "attention_mask": gpu_mask})

    reset_memory_stats()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        te.infer_gpu({"input_ids": gpu_ids, "attention_mask": gpu_mask})
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {"median_ms": float(np.median(times)), "peak_gpu_mb": get_gpu_memory_mb()}


def bench_decoder(dec: TRTModule, dec_inputs: dict,
                  num_warmup: int, num_runs: int) -> dict:
    """Test 3: Decoder only (inputs already on GPU as torch tensors)."""
    for _ in range(num_warmup):
        dec.infer_gpu(dec_inputs)

    reset_memory_stats()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dec.infer_gpu(dec_inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {"median_ms": float(np.median(times)), "peak_gpu_mb": get_gpu_memory_mb()}


def bench_full_sequential(ve: TRTModule, te: TRTModule, dec: TRTModule,
                          pixel_values_list: list, input_ids, attention_mask,
                          num_warmup: int, num_runs: int) -> dict:
    """Test 4: Full sequential pipeline — all on GPU, no CPU copies between modules."""
    n = len(pixel_values_list)
    gpu_inputs = [torch.from_numpy(pv).cuda().contiguous() for pv in pixel_values_list]
    gpu_ids = torch.from_numpy(input_ids).cuda().contiguous()
    gpu_mask = torch.from_numpy(attention_mask).cuda().contiguous()

    def run_one(pv):
        ve_out = ve.infer_gpu({"images": pv})
        te_out = te.infer_gpu({"input_ids": gpu_ids, "attention_mask": gpu_mask})
        dec.infer_gpu({
            "fpn_feat_0": ve_out["fpn_feat_0"],
            "fpn_feat_1": ve_out["fpn_feat_1"],
            "fpn_feat_2": ve_out["fpn_feat_2"],
            "fpn_pos_2": ve_out["fpn_pos_2"],
            "text_features": te_out["text_features"],
            "text_mask": te_out["text_mask"],
        })

    for _ in range(num_warmup):
        for pv in gpu_inputs:
            run_one(pv)

    reset_memory_stats()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for pv in gpu_inputs:
            run_one(pv)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times)), "peak_gpu_mb": get_gpu_memory_mb()}


def bench_pipelined(ve: TRTModule, te: TRTModule, dec: TRTModule,
                    pixel_values_list: list, input_ids, attention_mask,
                    num_warmup: int, num_runs: int) -> dict:
    """Test 5: Pipelined — VE+TE(frame N) || Decoder(frame N-1), all on GPU."""
    n = len(pixel_values_list)
    gpu_inputs = [torch.from_numpy(pv).cuda().contiguous() for pv in pixel_values_list]
    gpu_ids = torch.from_numpy(input_ids).cuda().contiguous()
    gpu_mask = torch.from_numpy(attention_mask).cuda().contiguous()

    # Cache TE output (same prompt every frame)
    te_out = te.infer_gpu({"input_ids": gpu_ids, "attention_mask": gpu_mask})

    def run_pipeline():
        prev_ve_out = None

        for pv in gpu_inputs:
            ve_out = ve.infer_gpu({"images": pv})

            if prev_ve_out is not None:
                dec.infer_gpu({
                    "fpn_feat_0": prev_ve_out["fpn_feat_0"],
                    "fpn_feat_1": prev_ve_out["fpn_feat_1"],
                    "fpn_feat_2": prev_ve_out["fpn_feat_2"],
                    "fpn_pos_2": prev_ve_out["fpn_pos_2"],
                    "text_features": te_out["text_features"],
                    "text_mask": te_out["text_mask"],
                })

            prev_ve_out = ve_out

        # Decode last frame
        if prev_ve_out is not None:
            dec.infer_gpu({
                "fpn_feat_0": prev_ve_out["fpn_feat_0"],
                "fpn_feat_1": prev_ve_out["fpn_feat_1"],
                "fpn_feat_2": prev_ve_out["fpn_feat_2"],
                "fpn_pos_2": prev_ve_out["fpn_pos_2"],
                "text_features": te_out["text_features"],
                "text_mask": te_out["text_mask"],
            })

    for _ in range(num_warmup):
        run_pipeline()
        torch.cuda.synchronize()

    reset_memory_stats()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_pipeline()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times)), "peak_gpu_mb": get_gpu_memory_mb()}


def bench_text_cached_pipelined(ve: TRTModule, dec: TRTModule,
                                pixel_values_list: list, te_out_cached: dict,
                                num_warmup: int, num_runs: int) -> dict:
    """Test 6: Text cached — VE(frame N) || Decoder(frame N-1), all on GPU."""
    n = len(pixel_values_list)
    gpu_inputs = [torch.from_numpy(pv).cuda().contiguous() for pv in pixel_values_list]

    def run_pipeline():
        prev_ve_out = None

        for pv in gpu_inputs:
            ve_out = ve.infer_gpu({"images": pv})

            if prev_ve_out is not None:
                dec.infer_gpu({
                    "fpn_feat_0": prev_ve_out["fpn_feat_0"],
                    "fpn_feat_1": prev_ve_out["fpn_feat_1"],
                    "fpn_feat_2": prev_ve_out["fpn_feat_2"],
                    "fpn_pos_2": prev_ve_out["fpn_pos_2"],
                    "text_features": te_out_cached["text_features"],
                    "text_mask": te_out_cached["text_mask"],
                })

            prev_ve_out = ve_out

        # Decode last frame
        if prev_ve_out is not None:
            dec.infer_gpu({
                "fpn_feat_0": prev_ve_out["fpn_feat_0"],
                "fpn_feat_1": prev_ve_out["fpn_feat_1"],
                "fpn_feat_2": prev_ve_out["fpn_feat_2"],
                "fpn_pos_2": prev_ve_out["fpn_pos_2"],
                "text_features": te_out_cached["text_features"],
                "text_mask": te_out_cached["text_mask"],
            })

    for _ in range(num_warmup):
        run_pipeline()
        torch.cuda.synchronize()

    reset_memory_stats()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_pipeline()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times)), "peak_gpu_mb": get_gpu_memory_mb()}


def main():
    parser = argparse.ArgumentParser(description="Benchmark split TRT pipeline")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP)
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of images to use (default 5)")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a single test (1-6). Omit to run all.")
    parser.add_argument("--engine-label", type=str, default="fp16_mixed",
                        choices=["fp16_mixed", "fp16_pure", "fp32"])
    args = parser.parse_args()

    label = args.engine_label
    ve_path = SPLIT_ENGINE_DIR / f"vision-encoder_{label}.plan"
    te_path = SPLIT_ENGINE_DIR / f"text-encoder_{label}.plan"
    dec_path = SPLIT_ENGINE_DIR / f"decoder_{label}.plan"

    for p in [ve_path, te_path, dec_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run build_split_engines.py first.")
            return

    images = get_image_paths()[:args.num_images]
    print(f"Images: {len(images)}")
    print(f"Engines: {label}")
    print(f"Warmup: {args.warmup}, Timed runs: {args.runs}")
    if args.test:
        print(f"Running test {args.test} only")
    print()

    # Prepare inputs
    print("Preparing inputs (preprocessing images)...")
    pixel_values_list, input_ids, attention_mask = prepare_inputs(images)
    print(f"  {len(pixel_values_list)} images preprocessed")
    print(f"  pixel_values: {pixel_values_list[0].shape}")
    print(f"  input_ids: {input_ids.shape}")
    print()

    # Load engines
    print("Loading TRT engines...")
    ve = TRTModule(str(ve_path))
    te = TRTModule(str(te_path))
    dec = TRTModule(str(dec_path))
    print("  All engines loaded.")
    print()

    results = {}
    run_test = args.test

    # Test 1: Vision Encoder
    if not run_test or run_test == 1:
        print("=" * 60)
        print("Test 1: Vision Encoder Only")
        print("=" * 60)
        r = bench_vision_encoder(ve, pixel_values_list, args.warmup, args.runs)
        r["fps"] = 1000 / r["median_ms"]
        results["vision_encoder"] = r
        print(f"  {r['median_ms']:.2f} ms/frame  |  {r['fps']:.1f} FPS  |  {r['peak_gpu_mb']:.0f} MB")

    # Test 2: Text Encoder
    if not run_test or run_test == 2:
        print("\n" + "=" * 60)
        print("Test 2: Text Encoder Only")
        print("=" * 60)
        r = bench_text_encoder(te, input_ids, attention_mask, args.warmup, args.runs)
        r["fps"] = 1000 / r["median_ms"]
        results["text_encoder"] = r
        print(f"  {r['median_ms']:.2f} ms/frame  |  {r['fps']:.1f} FPS  |  {r['peak_gpu_mb']:.0f} MB")

    # Pre-compute decoder inputs for Test 3
    if not run_test or run_test in (3, 4, 5, 6):
        gpu_pv0 = torch.from_numpy(pixel_values_list[0]).cuda().contiguous()
        ve_out = ve.infer_gpu({"images": gpu_pv0})
        gpu_ids = torch.from_numpy(input_ids).cuda().contiguous()
        gpu_mask = torch.from_numpy(attention_mask).cuda().contiguous()
        te_out = te.infer_gpu({"input_ids": gpu_ids, "attention_mask": gpu_mask})
        dec_inputs = {
            "fpn_feat_0": ve_out["fpn_feat_0"],
            "fpn_feat_1": ve_out["fpn_feat_1"],
            "fpn_feat_2": ve_out["fpn_feat_2"],
            "fpn_pos_2": ve_out["fpn_pos_2"],
            "text_features": te_out["text_features"],
            "text_mask": te_out["text_mask"],
        }

    # Test 3: Decoder
    if not run_test or run_test == 3:
        print("\n" + "=" * 60)
        print("Test 3: Decoder Only")
        print("=" * 60)
        r = bench_decoder(dec, dec_inputs, args.warmup, args.runs)
        r["fps"] = 1000 / r["median_ms"]
        results["decoder"] = r
        print(f"  {r['median_ms']:.2f} ms/frame  |  {r['fps']:.1f} FPS  |  {r['peak_gpu_mb']:.0f} MB")

    # Test 4: Full Sequential
    if not run_test or run_test == 4:
        print("\n" + "=" * 60)
        print("Test 4: Full Sequential (VE + TE + Dec)")
        print("=" * 60)
        r = bench_full_sequential(ve, te, dec, pixel_values_list, input_ids,
                                  attention_mask, args.warmup, args.runs)
        r["fps"] = 1000 / r["median_ms"]
        results["full_sequential"] = r
        print(f"  {r['median_ms']:.2f} ms/frame  |  {r['fps']:.1f} FPS  |  {r['peak_gpu_mb']:.0f} MB")

    # Test 5: Pipelined
    if not run_test or run_test == 5:
        print("\n" + "=" * 60)
        print("Test 5: Pipelined — VE+TE(frame N) || Dec(frame N-1)")
        print("=" * 60)
        r = bench_pipelined(ve, te, dec, pixel_values_list, input_ids,
                            attention_mask, args.warmup, args.runs)
        r["fps"] = 1000 / r["median_ms"]
        results["pipelined"] = r
        print(f"  {r['median_ms']:.2f} ms/frame  |  {r['fps']:.1f} FPS  |  {r['peak_gpu_mb']:.0f} MB")

    # Test 6: Text Cached + Pipelined
    if not run_test or run_test == 6:
        print("\n" + "=" * 60)
        print("Test 6: Text Cached — VE(frame N) || Dec(frame N-1)")
        print("=" * 60)
        if 'te_out' not in locals():
            gpu_ids = torch.from_numpy(input_ids).cuda().contiguous()
            gpu_mask = torch.from_numpy(attention_mask).cuda().contiguous()
            te_out = te.infer_gpu({"input_ids": gpu_ids, "attention_mask": gpu_mask})
        r = bench_text_cached_pipelined(ve, dec, pixel_values_list, te_out,
                                        args.warmup, args.runs)
        r["fps"] = 1000 / r["median_ms"]
        results["text_cached_pipelined"] = r
        print(f"  {r['median_ms']:.2f} ms/frame  |  {r['fps']:.1f} FPS  |  {r['peak_gpu_mb']:.0f} MB")

    # Summary table
    print("\n\n" + "=" * 70)
    print(f"{'Configuration':<30} {'ms/frame':>10} {'FPS':>8} {'Peak GPU MB':>12}")
    print("-" * 70)
    labels = {
        "vision_encoder": "Vision Encoder",
        "text_encoder": "Text Encoder",
        "decoder": "Decoder",
        "full_sequential": "Full Sequential",
        "pipelined": "Pipelined (VE+TE||Dec)",
        "text_cached_pipelined": "Text Cached (VE||Dec)",
    }
    for key, display in labels.items():
        if key not in results:
            continue
        r = results[key]
        print(f"{display:<30} {r['median_ms']:>10.2f} {r['fps']:>8.1f} {r['peak_gpu_mb']:>12.0f}")
    print("=" * 70)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "split_pipeline_fps.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
