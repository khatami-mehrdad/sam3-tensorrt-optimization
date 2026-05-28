#!/usr/bin/env python3
"""Verify output fidelity across optimization levels.

Runs a single image through Level 0 (PyTorch) and each TRT engine,
then compares the output tensors. For lossless optimizations the outputs
should match within FP16 tolerance. Reports max absolute difference,
mean absolute difference, and cosine similarity for each level pair.
"""

import sys
import os
import numpy as np
import torch
from PIL import Image

from bench_common import get_image_paths, ENGINE_DIR, PROMPT

# SAM3 preprocessing constants
IMG_SIZE = 1008
MEAN = 0.5
STD = 0.5

# Tolerances
FP16_ATOL = 1e-2   # absolute tolerance for FP16 rounding
FP16_RTOL = 1e-2   # relative tolerance
COSINE_THRESHOLD = 0.995  # cosine similarity must exceed this for split engines

SPLIT_ENGINE_DIR = ENGINE_DIR / "split"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compare_outputs(name_a: str, out_a: dict, name_b: str, out_b: dict):
    """Compare two sets of outputs and print diagnostics."""
    print(f"\n  Comparing {name_a} vs {name_b}:")
    all_ok = True
    for key in out_a:
        if key not in out_b:
            print(f"    {key}: MISSING in {name_b}")
            all_ok = False
            continue
        a = out_a[key]
        b = out_b[key]
        if a.shape != b.shape:
            print(f"    {key}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            all_ok = False
            continue

        max_abs = np.max(np.abs(a - b))
        mean_abs = np.mean(np.abs(a - b))
        cos_sim = cosine_similarity(a, b)

        status = "OK" if (max_abs < FP16_ATOL or cos_sim > COSINE_THRESHOLD) else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False

        print(f"    {key} {a.shape}:")
        print(f"      max_abs_diff={max_abs:.6e}  mean_abs_diff={mean_abs:.6e}  "
              f"cosine={cos_sim:.6f}  [{status}]")

    return all_ok


def get_level0_output(img_path: str, dtype: str = "fp32") -> dict:
    """Run Level 0 PyTorch inference and return raw output tensors.

    Args:
        img_path: Path to the input image.
        dtype: One of "fp32", "fp16", "bf16". Uses torch.autocast for reduced
               precision (matches how SAM3 is deployed in production).
    """
    from transformers.models.sam3 import Sam3Processor, Sam3Model

    device = "cuda"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    pil_img = Image.open(img_path).convert("RGB")
    inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt").to(device)

    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    autocast_dtype = dtype_map.get(dtype)

    with torch.no_grad():
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    return {
        "instance_masks": outputs.pred_masks.float().cpu().numpy(),
        "semantic_seg": outputs.semantic_seg.float().cpu().numpy(),
    }


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0).astype(np.float32)


def get_trt_output(engine_path: str, img_path: str) -> dict:
    """Run TRT inference and return raw output tensors."""
    import tensorrt as trt

    pil_img = Image.open(img_path).convert("RGB")
    pixel_values = preprocess_image(pil_img)

    # Tokenize prompt
    from transformers.models.sam3 import Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    dummy_img = Image.new("RGB", (224, 224))
    tok = processor(images=dummy_img, text=PROMPT, return_tensors="pt")
    input_ids = tok["input_ids"].numpy().astype(np.int64)
    attention_mask = tok["attention_mask"].numpy().astype(np.int64)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    buffers = {}
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = 1
        for d in shape:
            size *= max(1, d)
        buf = torch.zeros(size, dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype,
                          device="cuda")
        buffers[name] = {"tensor": buf, "shape": tuple(shape), "dtype": dtype}
        context.set_tensor_address(name, buf.data_ptr())
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    def set_input(name, data):
        buf = buffers[name]["tensor"]
        t = torch.from_numpy(data.ravel()).cuda()
        buf[:t.numel()] = t

    set_input("pixel_values", pixel_values)
    set_input("input_ids", input_ids)
    set_input("attention_mask", attention_mask)

    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    results = {}
    for name in output_names:
        shape = buffers[name]["shape"]
        dtype = buffers[name]["dtype"]
        size = 1
        for d in shape:
            size *= max(1, d)
        arr = buffers[name]["tensor"][:size].cpu().numpy().astype(np.float32)
        arr = arr.reshape(shape)
        results[name] = arr

    return results


class TRTModule:
    """Generic TRT engine runner with dynamic shape support."""

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

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                self.output_meta[name] = {"dtype": dtype}

    def infer(self, inputs: dict) -> dict:
        """Run inference with explicit input arrays."""
        buffers = {}
        for name, data in inputs.items():
            t = torch.from_numpy(data).cuda().contiguous()
            buffers[name] = t
            self.context.set_input_shape(name, tuple(data.shape))
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

        results = {}
        for name in self.output_names:
            results[name] = buffers[name].cpu().numpy().astype(np.float32)
        return results


def get_split_trt_output(img_path: str) -> dict:
    """Run split-module TRT pipeline (vision-encoder -> text-encoder -> decoder)."""
    from transformers.models.sam3 import Sam3Processor

    ve_path = SPLIT_ENGINE_DIR / "vision-encoder_fp16_mixed.plan"
    te_path = SPLIT_ENGINE_DIR / "text-encoder_fp16_mixed.plan"
    dec_path = SPLIT_ENGINE_DIR / "decoder_fp16_mixed.plan"

    for p in [ve_path, te_path, dec_path]:
        if not p.exists():
            return None

    # Use processor for preprocessing (matches PyTorch reference)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    pil_img = Image.open(img_path).convert("RGB")
    inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt")
    pixel_values = inputs["pixel_values"].numpy()
    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    attention_mask = inputs["attention_mask"].numpy().astype(np.int64)

    # Vision encoder
    ve = TRTModule(str(ve_path))
    ve_out = ve.infer({"images": pixel_values})

    # Text encoder
    te = TRTModule(str(te_path))
    te_out = te.infer({"input_ids": input_ids, "attention_mask": attention_mask})

    # Decoder
    dec = TRTModule(str(dec_path))
    text_mask = te_out["text_mask"]
    if text_mask.dtype != np.bool_:
        text_mask = text_mask.astype(np.bool_)
    dec_out = dec.infer({
        "fpn_feat_0": ve_out["fpn_feat_0"],
        "fpn_feat_1": ve_out["fpn_feat_1"],
        "fpn_feat_2": ve_out["fpn_feat_2"],
        "fpn_pos_2": ve_out["fpn_pos_2"],
        "text_features": te_out["text_features"],
        "text_mask": text_mask,
    })

    return {"instance_masks": dec_out["pred_masks"]}


def main():
    images = get_image_paths()
    test_img = images[0]
    print(f"Test image: {test_img}")
    print(f"Prompt: '{PROMPT}'")

    engines = {
        "Level 1 (TRT FP16)": ENGINE_DIR / "level1_fp16.plan",
        "Level 2 (+ onnxsim)": ENGINE_DIR / "level2_fp16.plan",
        "Level 3 (+ opt lv5)": ENGINE_DIR / "level3_fp16_opt5.plan",
    }

    # Check which engines exist
    available = {k: v for k, v in engines.items() if v.exists() and v.stat().st_size > 0}
    if not available:
        print("ERROR: No TRT engines found. Build them first.")
        sys.exit(1)

    print(f"\nAvailable engines: {list(available.keys())}")

    # Get PyTorch FP32 reference output
    print("\n--- Running Level 0 (PyTorch FP32 reference) ---")
    ref_output = get_level0_output(test_img, dtype="fp32")
    for k, v in ref_output.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype} "
              f"range=[{v.min():.4f}, {v.max():.4f}]")

    # Get PyTorch BF16 output (production dtype for SAM3)
    print("\n--- Running Level 0 (PyTorch BF16 — production default) ---")
    ref_bf16_output = get_level0_output(test_img, dtype="bf16")
    for k, v in ref_bf16_output.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype} "
              f"range=[{v.min():.4f}, {v.max():.4f}]")

    # Get PyTorch FP16 output
    print("\n--- Running Level 0 (PyTorch FP16) ---")
    ref_fp16_output = get_level0_output(test_img, dtype="fp16")
    for k, v in ref_fp16_output.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype} "
              f"range=[{v.min():.4f}, {v.max():.4f}]")

    all_pass = True

    # Compare PyTorch FP32 vs PyTorch BF16
    ok = compare_outputs("Level 0 (PyTorch FP32)", ref_output,
                         "Level 0 (PyTorch BF16)", ref_bf16_output)
    if not ok:
        all_pass = False

    # Compare PyTorch FP32 vs PyTorch FP16
    ok = compare_outputs("Level 0 (PyTorch FP32)", ref_output,
                         "Level 0 (PyTorch FP16)", ref_fp16_output)
    if not ok:
        all_pass = False

    # Compare PyTorch BF16 vs PyTorch FP16
    ok = compare_outputs("Level 0 (PyTorch BF16)", ref_bf16_output,
                         "Level 0 (PyTorch FP16)", ref_fp16_output)
    if not ok:
        all_pass = False

    # Get TRT outputs and compare
    trt_outputs = {}
    for name, engine_path in available.items():
        print(f"\n--- Running {name} ---")
        print(f"  Engine: {engine_path.name}")
        trt_out = get_trt_output(str(engine_path), test_img)
        trt_outputs[name] = trt_out
        for k, v in trt_out.items():
            print(f"  {k}: shape={v.shape} dtype={v.dtype} "
                  f"range=[{v.min():.4f}, {v.max():.4f}]")

        ok = compare_outputs("Level 0 (PyTorch FP32)", ref_output, name, trt_out)
        if not ok:
            all_pass = False

    # Compare PyTorch BF16 vs TRT engines (most relevant: same-intent precision)
    for name in trt_outputs:
        ok = compare_outputs("Level 0 (PyTorch BF16)", ref_bf16_output,
                             name, trt_outputs[name])
        if not ok:
            all_pass = False

    # Compare PyTorch FP16 vs TRT engines (same precision, different runtime)
    for name in trt_outputs:
        ok = compare_outputs("Level 0 (PyTorch FP16)", ref_fp16_output,
                             name, trt_outputs[name])
        if not ok:
            all_pass = False

    # Compare TRT engines against each other (should be even closer)
    trt_names = list(trt_outputs.keys())
    for i in range(len(trt_names)):
        for j in range(i + 1, len(trt_names)):
            ok = compare_outputs(
                trt_names[i], trt_outputs[trt_names[i]],
                trt_names[j], trt_outputs[trt_names[j]],
            )
            if not ok:
                all_pass = False

    # Split-module TRT pipeline (proven fix for monolithic ONNX divergence)
    print("\n--- Running Split-Module TRT Pipeline (FP16 mixed precision) ---")
    split_output = get_split_trt_output(test_img)
    if split_output is not None:
        for k, v in split_output.items():
            print(f"  {k}: shape={v.shape} dtype={v.dtype} "
                  f"range=[{v.min():.4f}, {v.max():.4f}]")
        ok = compare_outputs("Level 0 (PyTorch FP32)", ref_output,
                             "Split TRT (FP16 mixed)", split_output)
        if not ok:
            all_pass = False
    else:
        print("  SKIP: Split engines not found. Build them with build_split_engines.py --all")

    # Summary
    print(f"\n{'='*60}")
    if all_pass:
        print("RESULT: ALL OUTPUTS MATCH within tolerance.")
    else:
        print("RESULT: SOME OUTPUTS DIFFER beyond tolerance. Check above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
