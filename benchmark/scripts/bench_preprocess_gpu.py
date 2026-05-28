#!/usr/bin/env python3
"""Benchmark: CPU preprocessing vs GPU preprocessing for the TRT pipeline.

Compares:
  A) CPU preprocessing (Sam3Processor via HuggingFace) + TRT inference
  B) GPU preprocessing (torchvision v2 on CUDA) + TRT inference
  C) Fully pipelined: async CPU preprocess overlap with GPU inference

Also validates that GPU preprocessing produces identical TRT outputs to
CPU preprocessing (fidelity check).

Usage:
    python3 bench_preprocess_gpu.py
    python3 bench_preprocess_gpu.py --num-images 10
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from PIL import Image

from bench_common import get_image_paths, PROMPT, RESULTS_DIR

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SPLIT_ENGINE_DIR = BENCHMARK_DIR / "engines" / "split"

NUM_WARMUP = 5
NUM_RUNS = 20


def get_process_gpu_memory_mb() -> float:
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
    return 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# =============================================================================
# TRT Module (same as bench_split_pipeline.py)
# =============================================================================

class TRTModule:
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


# =============================================================================
# Preprocessing Methods
# =============================================================================

class CPUPreprocessor:
    """Standard HuggingFace Sam3Processor (CPU-based)."""

    def __init__(self):
        from transformers.models.sam3 import Sam3Processor
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        # Cache tokenized text (same prompt every frame)
        dummy = Image.new("RGB", (100, 100))
        tok = self.processor(images=dummy, text=PROMPT, return_tensors="pt")
        self.input_ids = tok["input_ids"].to(torch.int64).cuda()
        self.attention_mask = tok["attention_mask"].to(torch.int64).cuda()

    def preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        """Returns pixel_values as CUDA tensor [1, 3, 1008, 1008]."""
        inputs = self.processor(images=pil_img, text=PROMPT, return_tensors="pt")
        return inputs["pixel_values"].cuda()


class GPUPreprocessor:
    """GPU-based preprocessing using torchvision v2 transforms.

    Replicates Sam3Processor exactly:
      1. Resize to 1008x1008 (bilinear, stretch)
      2. Scale uint8 [0,255] -> float32 [0,1]
      3. Normalize with mean=0.5, std=0.5 -> [-1, 1]
    """

    def __init__(self):
        from transformers.models.sam3 import Sam3Processor
        # Cache tokenized text
        proc = Sam3Processor.from_pretrained("facebook/sam3")
        dummy = Image.new("RGB", (100, 100))
        tok = proc(images=dummy, text=PROMPT, return_tensors="pt")
        self.input_ids = tok["input_ids"].to(torch.int64).cuda()
        self.attention_mask = tok["attention_mask"].to(torch.int64).cuda()
        del proc

        self.transform = v2.Compose([
            v2.Resize((1008, 1008), interpolation=v2.InterpolationMode.BILINEAR,
                      antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        """Returns pixel_values as CUDA tensor [1, 3, 1008, 1008]."""
        # Convert PIL to uint8 tensor on GPU (minimal transfer: 1 byte/pixel)
        img_np = np.asarray(pil_img)  # HWC uint8
        img_gpu = torch.from_numpy(img_np).cuda().permute(2, 0, 1)  # CHW uint8 on GPU
        pixel_values = self.transform(img_gpu)
        return pixel_values.unsqueeze(0)  # [1, 3, 1008, 1008]

    def preprocess_from_tensor(self, img_gpu_chw: torch.Tensor) -> torch.Tensor:
        """Preprocess from a uint8 CHW tensor already on GPU."""
        pixel_values = self.transform(img_gpu_chw)
        return pixel_values.unsqueeze(0)


# =============================================================================
# TRT Pipeline
# =============================================================================

class TRTPipeline:
    """Loads all 3 TRT engines for sequential inference."""

    def __init__(self):
        self.ve = TRTModule(str(SPLIT_ENGINE_DIR / "vision-encoder_fp16_mixed.plan"))
        self.te = TRTModule(str(SPLIT_ENGINE_DIR / "text-encoder_fp16_mixed.plan"))
        self.dec = TRTModule(str(SPLIT_ENGINE_DIR / "decoder_fp16_mixed.plan"))

    def infer(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
              attention_mask: torch.Tensor) -> dict:
        ve_out = self.ve.infer_gpu({"images": pixel_values})
        te_out = self.te.infer_gpu({"input_ids": input_ids,
                                    "attention_mask": attention_mask})
        dec_out = self.dec.infer_gpu({
            "fpn_feat_0": ve_out["fpn_feat_0"],
            "fpn_feat_1": ve_out["fpn_feat_1"],
            "fpn_feat_2": ve_out["fpn_feat_2"],
            "fpn_pos_2": ve_out["fpn_pos_2"],
            "text_features": te_out["text_features"],
            "text_mask": te_out["text_mask"],
        })
        return dec_out


# =============================================================================
# Benchmark Runners
# =============================================================================

def bench_cpu_preprocess_only(cpu_prep: CPUPreprocessor, pil_images: list,
                              num_warmup: int, num_runs: int) -> dict:
    """Measure CPU preprocessing time alone."""
    n = len(pil_images)
    for _ in range(num_warmup):
        for img in pil_images:
            cpu_prep.preprocess(img)

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        for img in pil_images:
            cpu_prep.preprocess(img)
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times))}


def bench_gpu_preprocess_only(gpu_prep: GPUPreprocessor, pil_images: list,
                              num_warmup: int, num_runs: int) -> dict:
    """Measure GPU preprocessing time alone."""
    n = len(pil_images)
    # Pre-upload images as uint8 tensors on GPU
    gpu_images = []
    for img in pil_images:
        img_np = np.asarray(img)
        gpu_images.append(torch.from_numpy(img_np).cuda().permute(2, 0, 1))

    for _ in range(num_warmup):
        for img_gpu in gpu_images:
            gpu_prep.preprocess_from_tensor(img_gpu)

    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for img_gpu in gpu_images:
            gpu_prep.preprocess_from_tensor(img_gpu)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times))}


def bench_end_to_end_cpu(cpu_prep: CPUPreprocessor, pipeline: TRTPipeline,
                         pil_images: list, num_warmup: int, num_runs: int) -> dict:
    """End-to-end: CPU preprocess + TRT inference (sequential)."""
    n = len(pil_images)

    def run_all():
        for img in pil_images:
            pv = cpu_prep.preprocess(img)
            pipeline.infer(pv, cpu_prep.input_ids, cpu_prep.attention_mask)

    for _ in range(num_warmup):
        run_all()

    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_all()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times)), "gpu_mb": get_process_gpu_memory_mb()}


def bench_end_to_end_gpu(gpu_prep: GPUPreprocessor, pipeline: TRTPipeline,
                         pil_images: list, num_warmup: int, num_runs: int) -> dict:
    """End-to-end: GPU preprocess + TRT inference (sequential)."""
    n = len(pil_images)
    # Pre-upload raw uint8 images to GPU
    gpu_images = []
    for img in pil_images:
        img_np = np.asarray(img)
        gpu_images.append(torch.from_numpy(img_np).cuda().permute(2, 0, 1))

    def run_all():
        for img_gpu in gpu_images:
            pv = gpu_prep.preprocess_from_tensor(img_gpu)
            pipeline.infer(pv, gpu_prep.input_ids, gpu_prep.attention_mask)

    for _ in range(num_warmup):
        run_all()

    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_all()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times)), "gpu_mb": get_process_gpu_memory_mb()}


def bench_pipelined_gpu(gpu_prep: GPUPreprocessor, pipeline: TRTPipeline,
                        pil_images: list, num_warmup: int, num_runs: int) -> dict:
    """Pipelined: GPU preprocess frame N+1 overlapped with TRT decode frame N."""
    n = len(pil_images)
    gpu_images = []
    for img in pil_images:
        img_np = np.asarray(img)
        gpu_images.append(torch.from_numpy(img_np).cuda().permute(2, 0, 1))

    # Cache text encoder output
    te_out = pipeline.te.infer_gpu({
        "input_ids": gpu_prep.input_ids,
        "attention_mask": gpu_prep.attention_mask,
    })

    def run_pipelined():
        prev_ve_out = None

        for img_gpu in gpu_images:
            # Preprocess + VE for current frame
            pv = gpu_prep.preprocess_from_tensor(img_gpu)
            ve_out = pipeline.ve.infer_gpu({"images": pv})

            # Decode previous frame
            if prev_ve_out is not None:
                pipeline.dec.infer_gpu({
                    "fpn_feat_0": prev_ve_out["fpn_feat_0"],
                    "fpn_feat_1": prev_ve_out["fpn_feat_1"],
                    "fpn_feat_2": prev_ve_out["fpn_feat_2"],
                    "fpn_pos_2": prev_ve_out["fpn_pos_2"],
                    "text_features": te_out["text_features"],
                    "text_mask": te_out["text_mask"],
                })
            prev_ve_out = ve_out

        # Flush last frame
        if prev_ve_out is not None:
            pipeline.dec.infer_gpu({
                "fpn_feat_0": prev_ve_out["fpn_feat_0"],
                "fpn_feat_1": prev_ve_out["fpn_feat_1"],
                "fpn_feat_2": prev_ve_out["fpn_feat_2"],
                "fpn_pos_2": prev_ve_out["fpn_pos_2"],
                "text_features": te_out["text_features"],
                "text_mask": te_out["text_mask"],
            })

    for _ in range(num_warmup):
        run_pipelined()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_pipelined()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / n * 1000)

    return {"median_ms": float(np.median(times)), "gpu_mb": get_process_gpu_memory_mb()}


# =============================================================================
# Fidelity Check
# =============================================================================

def check_fidelity(cpu_prep: CPUPreprocessor, gpu_prep: GPUPreprocessor,
                   pipeline: TRTPipeline, pil_images: list) -> dict:
    """Compare TRT outputs: CPU-preprocessed vs GPU-preprocessed inputs."""
    cosines = []
    max_diffs = []
    preprocess_max_diffs = []

    for img in pil_images:
        # CPU preprocess
        pv_cpu = cpu_prep.preprocess(img)
        # GPU preprocess
        img_np = np.asarray(img)
        img_gpu = torch.from_numpy(img_np).cuda().permute(2, 0, 1)
        pv_gpu = gpu_prep.preprocess_from_tensor(img_gpu)

        # Check preprocessing match
        pp_diff = (pv_cpu - pv_gpu).abs().max().item()
        preprocess_max_diffs.append(pp_diff)

        # Run TRT with CPU-preprocessed input
        out_cpu = pipeline.infer(pv_cpu, cpu_prep.input_ids, cpu_prep.attention_mask)
        masks_cpu = out_cpu["pred_masks"].float().cpu().numpy()

        # Run TRT with GPU-preprocessed input
        out_gpu = pipeline.infer(pv_gpu, gpu_prep.input_ids, gpu_prep.attention_mask)
        masks_gpu = out_gpu["pred_masks"].float().cpu().numpy()

        cos = cosine_similarity(masks_cpu, masks_gpu)
        max_diff = float(np.abs(masks_cpu - masks_gpu).max())
        cosines.append(cos)
        max_diffs.append(max_diff)

    return {
        "avg_preprocess_max_diff": float(np.mean(preprocess_max_diffs)),
        "max_preprocess_max_diff": float(np.max(preprocess_max_diffs)),
        "avg_cosine_similarity": float(np.mean(cosines)),
        "min_cosine_similarity": float(np.min(cosines)),
        "avg_max_abs_diff": float(np.mean(max_diffs)),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU preprocessing benchmark")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP)
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    args = parser.parse_args()

    image_paths = get_image_paths()
    if args.num_images:
        image_paths = image_paths[:args.num_images]
    n = len(image_paths)

    print("=" * 70)
    print("PREPROCESSING BENCHMARK: CPU vs GPU")
    print(f"  Images: {n}, Warmup: {args.warmup}, Runs: {args.runs}")
    print("=" * 70)

    # Load images into memory
    print("\nLoading images...")
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"  {n} images loaded. Sizes: {[img.size for img in pil_images[:3]]}...")

    # Initialize preprocessors
    print("\nInitializing preprocessors...")
    cpu_prep = CPUPreprocessor()
    gpu_prep = GPUPreprocessor()
    print("  Done.")

    # Load TRT pipeline
    print("\nLoading TRT engines...")
    pipeline = TRTPipeline()
    print(f"  Engines loaded. GPU memory: {get_process_gpu_memory_mb():.0f} MB")

    # =========================================================================
    # Test 1: Preprocessing only (CPU vs GPU)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Preprocessing Time (CPU vs GPU)")
    print("=" * 70)

    r_cpu_pp = bench_cpu_preprocess_only(cpu_prep, pil_images, args.warmup, args.runs)
    print(f"  CPU preprocess: {r_cpu_pp['median_ms']:.2f} ms/frame")

    r_gpu_pp = bench_gpu_preprocess_only(gpu_prep, pil_images, args.warmup, args.runs)
    print(f"  GPU preprocess: {r_gpu_pp['median_ms']:.2f} ms/frame")

    speedup_pp = r_cpu_pp["median_ms"] / r_gpu_pp["median_ms"]
    print(f"  GPU speedup: {speedup_pp:.1f}x")

    # =========================================================================
    # Test 2: Fidelity check (CPU vs GPU preprocessing → same TRT output?)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Fidelity (CPU preprocess vs GPU preprocess → TRT output)")
    print("=" * 70)

    fid = check_fidelity(cpu_prep, gpu_prep, pipeline, pil_images[:min(10, n)])
    print(f"  Preprocessing pixel diff (avg max): {fid['avg_preprocess_max_diff']:.6f}")
    print(f"  Preprocessing pixel diff (worst):   {fid['max_preprocess_max_diff']:.6f}")
    print(f"  TRT output cosine similarity (avg): {fid['avg_cosine_similarity']:.6f}")
    print(f"  TRT output cosine similarity (min): {fid['min_cosine_similarity']:.6f}")
    print(f"  TRT output max abs diff (avg):      {fid['avg_max_abs_diff']:.4f}")

    # =========================================================================
    # Test 3: End-to-end (preprocess + TRT inference)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: End-to-End (Preprocess + TRT Full Sequential)")
    print("=" * 70)

    r_e2e_cpu = bench_end_to_end_cpu(cpu_prep, pipeline, pil_images,
                                     args.warmup, args.runs)
    fps_cpu = 1000 / r_e2e_cpu["median_ms"]
    print(f"  CPU preprocess + TRT: {r_e2e_cpu['median_ms']:.2f} ms/frame | "
          f"{fps_cpu:.1f} FPS | {r_e2e_cpu['gpu_mb']:.0f} MB")

    r_e2e_gpu = bench_end_to_end_gpu(gpu_prep, pipeline, pil_images,
                                     args.warmup, args.runs)
    fps_gpu = 1000 / r_e2e_gpu["median_ms"]
    print(f"  GPU preprocess + TRT: {r_e2e_gpu['median_ms']:.2f} ms/frame | "
          f"{fps_gpu:.1f} FPS | {r_e2e_gpu['gpu_mb']:.0f} MB")

    speedup_e2e = r_e2e_cpu["median_ms"] / r_e2e_gpu["median_ms"]
    print(f"  GPU preprocess speedup (end-to-end): {speedup_e2e:.2f}x")

    # =========================================================================
    # Test 4: Pipelined (GPU preprocess + VE frame N, Dec frame N-1)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Pipelined (GPU prep + VE frame N || Dec frame N-1)")
    print("=" * 70)

    r_pipe = bench_pipelined_gpu(gpu_prep, pipeline, pil_images,
                                 args.warmup, args.runs)
    fps_pipe = 1000 / r_pipe["median_ms"]
    print(f"  Pipelined: {r_pipe['median_ms']:.2f} ms/frame | "
          f"{fps_pipe:.1f} FPS | {r_pipe['gpu_mb']:.0f} MB")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Configuration':<40} {'ms/frame':>10} {'FPS':>8} {'GPU MB':>8}")
    print(f"  {'-'*66}")
    print(f"  {'CPU preprocess only':<40} {r_cpu_pp['median_ms']:>10.2f} {'—':>8} {'—':>8}")
    print(f"  {'GPU preprocess only':<40} {r_gpu_pp['median_ms']:>10.2f} {'—':>8} {'—':>8}")
    print(f"  {'CPU preprocess + TRT (sequential)':<40} {r_e2e_cpu['median_ms']:>10.2f} {fps_cpu:>8.1f} {r_e2e_cpu['gpu_mb']:>8.0f}")
    print(f"  {'GPU preprocess + TRT (sequential)':<40} {r_e2e_gpu['median_ms']:>10.2f} {fps_gpu:>8.1f} {r_e2e_gpu['gpu_mb']:>8.0f}")
    print(f"  {'GPU preprocess + TRT (pipelined)':<40} {r_pipe['median_ms']:>10.2f} {fps_pipe:>8.1f} {r_pipe['gpu_mb']:>8.0f}")
    print(f"  {'-'*66}")
    print(f"\n  Preprocessing speedup (GPU vs CPU): {speedup_pp:.1f}x")
    print(f"  End-to-end speedup (GPU vs CPU):     {speedup_e2e:.2f}x")
    print(f"  Fidelity (GPU preprocess):           cosine={fid['avg_cosine_similarity']:.6f}")
    print("=" * 70)

    # Save results
    results = {
        "num_images": n,
        "preprocessing": {
            "cpu_ms": r_cpu_pp["median_ms"],
            "gpu_ms": r_gpu_pp["median_ms"],
            "speedup": speedup_pp,
        },
        "end_to_end_cpu": {
            "ms": r_e2e_cpu["median_ms"],
            "fps": fps_cpu,
            "gpu_mb": r_e2e_cpu["gpu_mb"],
        },
        "end_to_end_gpu": {
            "ms": r_e2e_gpu["median_ms"],
            "fps": fps_gpu,
            "gpu_mb": r_e2e_gpu["gpu_mb"],
        },
        "pipelined_gpu": {
            "ms": r_pipe["median_ms"],
            "fps": fps_pipe,
            "gpu_mb": r_pipe["gpu_mb"],
        },
        "fidelity": fid,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "preprocess_gpu_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
