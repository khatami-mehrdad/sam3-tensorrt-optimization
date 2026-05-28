#!/usr/bin/env python3
"""Full pipeline comparison: PyTorch BF16 vs TRT Sequential vs TRT Pipelined.

Measures FPS, GPU memory, and fidelity (cosine similarity, max/mean abs diff)
on all images in benchmark/data, using PyTorch as ground truth.

Usage:
    python3 full_pipeline_comparison.py
    python3 full_pipeline_comparison.py --num-images 10
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench_common import get_image_paths, PROMPT, RESULTS_DIR

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SPLIT_ENGINE_DIR = BENCHMARK_DIR / "engines" / "split"

NUM_WARMUP = 5


def get_process_gpu_memory_mb() -> float:
    """Get GPU memory used by this process via nvidia-smi."""
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
# TRT Module
# =============================================================================

class TRTModule:
    """TRT engine runner — GPU-native tensors in/out."""

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
        """Run inference: accepts torch CUDA tensors, returns torch CUDA tensors."""
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
# PyTorch BF16 Runner
# =============================================================================

class PyTorchRunner:
    """PyTorch SAM3 with BF16 autocast (matching production usage)."""

    def __init__(self, device="cuda"):
        from transformers.models.sam3 import Sam3Processor, Sam3Model

        self.device = device
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def preprocess(self, pil_img: Image.Image) -> dict:
        inputs = self.processor(images=pil_img, text=PROMPT, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def infer(self, inputs: dict) -> dict:
        """Run with BF16 autocast, returns pred_masks as float32 numpy."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**inputs)
        return {
            "pred_masks": outputs.pred_masks.float().cpu().numpy(),
        }

    def get_trt_inputs(self, inputs: dict) -> tuple:
        """Extract pixel_values, input_ids, attention_mask for TRT."""
        return (
            inputs["pixel_values"].cpu().numpy(),
            inputs["input_ids"].cpu().numpy().astype(np.int64),
            inputs["attention_mask"].cpu().numpy().astype(np.int64),
        )


# =============================================================================
# TRT Pipeline Runners
# =============================================================================

class TRTSequentialRunner:
    """TRT Split Pipeline: VE -> TE -> Decoder (sequential)."""

    def __init__(self):
        ve_path = SPLIT_ENGINE_DIR / "vision-encoder_fp16_mixed.plan"
        te_path = SPLIT_ENGINE_DIR / "text-encoder_fp16_mixed.plan"
        dec_path = SPLIT_ENGINE_DIR / "decoder_fp16_mixed.plan"

        for p in [ve_path, te_path, dec_path]:
            if not p.exists():
                raise FileNotFoundError(f"Engine not found: {p}")

        self.ve = TRTModule(str(ve_path))
        self.te = TRTModule(str(te_path))
        self.dec = TRTModule(str(dec_path))

    def infer(self, pixel_values_gpu, input_ids_gpu, attention_mask_gpu) -> dict:
        """Run full sequential pipeline on GPU, return pred_masks as numpy."""
        ve_out = self.ve.infer_gpu({"images": pixel_values_gpu})
        te_out = self.te.infer_gpu({"input_ids": input_ids_gpu,
                                    "attention_mask": attention_mask_gpu})
        dec_out = self.dec.infer_gpu({
            "fpn_feat_0": ve_out["fpn_feat_0"],
            "fpn_feat_1": ve_out["fpn_feat_1"],
            "fpn_feat_2": ve_out["fpn_feat_2"],
            "fpn_pos_2": ve_out["fpn_pos_2"],
            "text_features": te_out["text_features"],
            "text_mask": te_out["text_mask"],
        })
        return {
            "pred_masks": dec_out["pred_masks"].float().cpu().numpy(),
        }


class TRTPipelinedRunner:
    """TRT Pipelined: VE(frame N) || Dec(frame N-1), text cached."""

    def __init__(self):
        ve_path = SPLIT_ENGINE_DIR / "vision-encoder_fp16_mixed.plan"
        te_path = SPLIT_ENGINE_DIR / "text-encoder_fp16_mixed.plan"
        dec_path = SPLIT_ENGINE_DIR / "decoder_fp16_mixed.plan"

        for p in [ve_path, te_path, dec_path]:
            if not p.exists():
                raise FileNotFoundError(f"Engine not found: {p}")

        self.ve = TRTModule(str(ve_path))
        self.te = TRTModule(str(te_path))
        self.dec = TRTModule(str(dec_path))
        self.te_out_cached = None
        self.prev_ve_out = None

    def cache_text(self, input_ids_gpu, attention_mask_gpu):
        """Pre-compute text features (done once per prompt)."""
        self.te_out_cached = self.te.infer_gpu({
            "input_ids": input_ids_gpu,
            "attention_mask": attention_mask_gpu,
        })

    def infer_pipelined(self, pixel_values_gpu) -> dict | None:
        """Process one frame: VE on current, Dec on previous.

        Returns decoded output for the PREVIOUS frame (None on first call).
        """
        ve_out = self.ve.infer_gpu({"images": pixel_values_gpu})

        result = None
        if self.prev_ve_out is not None:
            dec_out = self.dec.infer_gpu({
                "fpn_feat_0": self.prev_ve_out["fpn_feat_0"],
                "fpn_feat_1": self.prev_ve_out["fpn_feat_1"],
                "fpn_feat_2": self.prev_ve_out["fpn_feat_2"],
                "fpn_pos_2": self.prev_ve_out["fpn_pos_2"],
                "text_features": self.te_out_cached["text_features"],
                "text_mask": self.te_out_cached["text_mask"],
            })
            result = {"pred_masks": dec_out["pred_masks"].float().cpu().numpy()}

        self.prev_ve_out = ve_out
        return result

    def flush(self) -> dict:
        """Decode the last frame."""
        dec_out = self.dec.infer_gpu({
            "fpn_feat_0": self.prev_ve_out["fpn_feat_0"],
            "fpn_feat_1": self.prev_ve_out["fpn_feat_1"],
            "fpn_feat_2": self.prev_ve_out["fpn_feat_2"],
            "fpn_pos_2": self.prev_ve_out["fpn_pos_2"],
            "text_features": self.te_out_cached["text_features"],
            "text_mask": self.te_out_cached["text_mask"],
        })
        self.prev_ve_out = None
        return {"pred_masks": dec_out["pred_masks"].float().cpu().numpy()}


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_pytorch(runner: PyTorchRunner, pil_images: list,
                      num_warmup: int) -> tuple[list[dict], float, float]:
    """Run PyTorch BF16 on all images. Returns outputs, median_ms, gpu_mb."""
    n = len(pil_images)
    preprocessed = [runner.preprocess(img) for img in pil_images]

    # Warmup
    print(f"  Warmup ({num_warmup})...", end="", flush=True)
    for w in range(num_warmup):
        for inputs in preprocessed:
            runner.infer(inputs)
        print(f" {w+1}", end="", flush=True)
    print(" done.")

    # Timed run (single pass, collect outputs)
    outputs = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for inputs in preprocessed:
        out = runner.infer(inputs)
        outputs.append(out)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000
    median_ms = total_ms / n

    gpu_mb = get_process_gpu_memory_mb()
    return outputs, median_ms, gpu_mb


def benchmark_trt_sequential(runner: TRTSequentialRunner, preprocessed_gpu: list,
                             num_warmup: int) -> tuple[list[dict], float, float]:
    """Run TRT Sequential on all images. Returns outputs, median_ms, gpu_mb."""
    n = len(preprocessed_gpu)

    # Warmup
    print(f"  Warmup ({num_warmup})...", end="", flush=True)
    for w in range(num_warmup):
        for pv, ids, mask in preprocessed_gpu:
            runner.infer(pv, ids, mask)
        print(f" {w+1}", end="", flush=True)
    print(" done.")

    # Timed run
    outputs = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for pv, ids, mask in preprocessed_gpu:
        out = runner.infer(pv, ids, mask)
        outputs.append(out)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000
    median_ms = total_ms / n

    gpu_mb = get_process_gpu_memory_mb()
    return outputs, median_ms, gpu_mb


def benchmark_trt_pipelined(runner: TRTPipelinedRunner, preprocessed_gpu: list,
                            num_warmup: int) -> tuple[list[dict], float, float]:
    """Run TRT Pipelined on all images. Returns outputs, median_ms, gpu_mb."""
    n = len(preprocessed_gpu)

    # Warmup
    print(f"  Warmup ({num_warmup})...", end="", flush=True)
    for w in range(num_warmup):
        runner.prev_ve_out = None
        for pv, _, _ in preprocessed_gpu:
            runner.infer_pipelined(pv)
        runner.flush()
        print(f" {w+1}", end="", flush=True)
    print(" done.")

    # Timed run — collect all outputs in order
    outputs = []
    runner.prev_ve_out = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for pv, _, _ in preprocessed_gpu:
        result = runner.infer_pipelined(pv)
        if result is not None:
            outputs.append(result)
    outputs.append(runner.flush())
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000
    median_ms = total_ms / n

    gpu_mb = get_process_gpu_memory_mb()
    return outputs, median_ms, gpu_mb


# =============================================================================
# Fidelity Comparison
# =============================================================================

def compute_fidelity(ref_outputs: list[dict], test_outputs: list[dict]) -> dict:
    """Compare test outputs against reference (PyTorch). Per-image + aggregate."""
    cosines = []
    max_abs_diffs = []
    mean_abs_diffs = []

    for i, (ref, test) in enumerate(zip(ref_outputs, test_outputs)):
        ref_masks = ref["pred_masks"]
        test_masks = test["pred_masks"]

        if ref_masks.shape != test_masks.shape:
            min_shape = tuple(min(a, b) for a, b in zip(ref_masks.shape, test_masks.shape))
            slices = tuple(slice(0, s) for s in min_shape)
            ref_masks = ref_masks[slices]
            test_masks = test_masks[slices]

        cos = cosine_similarity(ref_masks, test_masks)
        max_abs = float(np.abs(ref_masks - test_masks).max())
        mean_abs = float(np.abs(ref_masks - test_masks).mean())

        cosines.append(cos)
        max_abs_diffs.append(max_abs)
        mean_abs_diffs.append(mean_abs)

    return {
        "avg_cosine_similarity": float(np.mean(cosines)),
        "min_cosine_similarity": float(np.min(cosines)),
        "avg_max_abs_diff": float(np.mean(max_abs_diffs)),
        "avg_mean_abs_diff": float(np.mean(mean_abs_diffs)),
        "per_image_cosine": cosines,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full pipeline comparison")
    parser.add_argument("--num-images", type=int, default=None,
                        help="Limit number of images (default: all)")
    parser.add_argument("--warmup", type=int, default=NUM_WARMUP)
    args = parser.parse_args()

    image_paths = get_image_paths()
    if args.num_images:
        image_paths = image_paths[:args.num_images]
    n = len(image_paths)

    print("=" * 70)
    print("FULL PIPELINE COMPARISON")
    print(f"  Images: {n}")
    print(f"  Prompt: '{PROMPT}'")
    print(f"  Warmup: {args.warmup}")
    print("=" * 70)

    # Load images
    print("\nLoading images...")
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"  {n} images loaded.")

    # =========================================================================
    # 1. PyTorch BF16
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. PyTorch BF16 (ground truth)")
    print("=" * 70)
    print("  Loading model...")
    pt_runner = PyTorchRunner()
    print(f"  Model loaded. GPU memory: {get_process_gpu_memory_mb():.0f} MB")

    pt_outputs, pt_ms, pt_gpu_mb = benchmark_pytorch(
        pt_runner, pil_images, args.warmup)
    pt_fps = 1000.0 / pt_ms
    print(f"  Result: {pt_ms:.2f} ms/frame | {pt_fps:.1f} FPS | {pt_gpu_mb:.0f} MB GPU")

    # Get TRT-compatible inputs from PyTorch preprocessing
    print("  Extracting preprocessed inputs for TRT...")
    preprocessed_gpu = []
    for pil_img in pil_images:
        inputs = pt_runner.preprocess(pil_img)
        pv = inputs["pixel_values"]
        ids = inputs["input_ids"].to(torch.int64)
        mask = inputs["attention_mask"].to(torch.int64)
        preprocessed_gpu.append((pv, ids, mask))

    # Free PyTorch model to get fair memory readings for TRT
    del pt_runner
    torch.cuda.empty_cache()
    time.sleep(2)
    print(f"  PyTorch model freed. GPU memory: {get_process_gpu_memory_mb():.0f} MB")

    # =========================================================================
    # 2. TRT Full Sequential
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. TRT Full Sequential (VE + TE + Dec)")
    print("=" * 70)
    print("  Loading engines...")
    trt_seq = TRTSequentialRunner()
    print(f"  Engines loaded. GPU memory: {get_process_gpu_memory_mb():.0f} MB")

    trt_seq_outputs, trt_seq_ms, trt_seq_gpu_mb = benchmark_trt_sequential(
        trt_seq, preprocessed_gpu, args.warmup)
    trt_seq_fps = 1000.0 / trt_seq_ms
    print(f"  Result: {trt_seq_ms:.2f} ms/frame | {trt_seq_fps:.1f} FPS | {trt_seq_gpu_mb:.0f} MB GPU")

    # =========================================================================
    # 3. TRT Pipelined (VE frame N || Dec frame N-1, text cached)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. TRT Pipelined (VE frame N || Dec frame N-1, text cached)")
    print("=" * 70)
    # Reuse same engines
    trt_pipe = TRTPipelinedRunner()
    # Cache text with the first image's input_ids/attention_mask
    _, ids0, mask0 = preprocessed_gpu[0]
    trt_pipe.cache_text(ids0, mask0)
    print(f"  Text features cached. GPU memory: {get_process_gpu_memory_mb():.0f} MB")

    trt_pipe_outputs, trt_pipe_ms, trt_pipe_gpu_mb = benchmark_trt_pipelined(
        trt_pipe, preprocessed_gpu, args.warmup)
    trt_pipe_fps = 1000.0 / trt_pipe_ms
    print(f"  Result: {trt_pipe_ms:.2f} ms/frame | {trt_pipe_fps:.1f} FPS | {trt_pipe_gpu_mb:.0f} MB GPU")

    # Free TRT
    del trt_seq, trt_pipe
    torch.cuda.empty_cache()

    # =========================================================================
    # Fidelity Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("FIDELITY COMPARISON (vs PyTorch BF16 ground truth)")
    print("=" * 70)

    fid_seq = compute_fidelity(pt_outputs, trt_seq_outputs)
    fid_pipe = compute_fidelity(pt_outputs, trt_pipe_outputs)

    print(f"\n  {'Metric':<25} {'TRT Sequential':>16} {'TRT Pipelined':>16}")
    print(f"  {'-'*57}")
    print(f"  {'Avg Cosine Similarity':<25} {fid_seq['avg_cosine_similarity']:>16.6f} {fid_pipe['avg_cosine_similarity']:>16.6f}")
    print(f"  {'Min Cosine Similarity':<25} {fid_seq['min_cosine_similarity']:>16.6f} {fid_pipe['min_cosine_similarity']:>16.6f}")
    print(f"  {'Avg Max Abs Diff':<25} {fid_seq['avg_max_abs_diff']:>16.4f} {fid_pipe['avg_max_abs_diff']:>16.4f}")
    print(f"  {'Avg Mean Abs Diff':<25} {fid_seq['avg_mean_abs_diff']:>16.6f} {fid_pipe['avg_mean_abs_diff']:>16.6f}")

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Method':<35} {'ms/frame':>10} {'FPS':>8} {'GPU MB':>8} {'Cosine':>8}")
    print(f"  {'-'*70}")
    print(f"  {'PyTorch BF16 (ground truth)':<35} {pt_ms:>10.2f} {pt_fps:>8.1f} {pt_gpu_mb:>8.0f} {'1.000000':>8}")
    print(f"  {'TRT Sequential (VE+TE+Dec)':<35} {trt_seq_ms:>10.2f} {trt_seq_fps:>8.1f} {trt_seq_gpu_mb:>8.0f} {fid_seq['avg_cosine_similarity']:>8.6f}")
    print(f"  {'TRT Pipelined (VE||Dec, text cached)':<35} {trt_pipe_ms:>10.2f} {trt_pipe_fps:>8.1f} {trt_pipe_gpu_mb:>8.0f} {fid_pipe['avg_cosine_similarity']:>8.6f}")
    print(f"  {'-'*70}")

    speedup_seq = pt_ms / trt_seq_ms if trt_seq_ms > 0 else 0
    speedup_pipe = pt_ms / trt_pipe_ms if trt_pipe_ms > 0 else 0
    mem_reduction_seq = pt_gpu_mb / trt_seq_gpu_mb if trt_seq_gpu_mb > 0 else 0
    mem_reduction_pipe = pt_gpu_mb / trt_pipe_gpu_mb if trt_pipe_gpu_mb > 0 else 0

    print(f"\n  Speedup (Sequential):  {speedup_seq:.1f}x")
    print(f"  Speedup (Pipelined):   {speedup_pipe:.1f}x")
    print(f"  Memory reduction (Seq): {mem_reduction_seq:.1f}x")
    print(f"  Memory reduction (Pipe): {mem_reduction_pipe:.1f}x")
    print("=" * 70)

    # Save results
    results = {
        "num_images": n,
        "prompt": PROMPT,
        "warmup": args.warmup,
        "pytorch_bf16": {
            "ms_per_frame": pt_ms,
            "fps": pt_fps,
            "gpu_mb": pt_gpu_mb,
        },
        "trt_sequential": {
            "ms_per_frame": trt_seq_ms,
            "fps": trt_seq_fps,
            "gpu_mb": trt_seq_gpu_mb,
            "fidelity": {k: v for k, v in fid_seq.items() if k != "per_image_cosine"},
        },
        "trt_pipelined": {
            "ms_per_frame": trt_pipe_ms,
            "fps": trt_pipe_fps,
            "gpu_mb": trt_pipe_gpu_mb,
            "fidelity": {k: v for k, v in fid_pipe.items() if k != "per_image_cosine"},
        },
        "speedup_sequential": speedup_seq,
        "speedup_pipelined": speedup_pipe,
        "memory_reduction_sequential": mem_reduction_seq,
        "memory_reduction_pipelined": mem_reduction_pipe,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "full_pipeline_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
