#!/usr/bin/env python3
"""Verify fidelity of the split-module TRT pipeline vs PyTorch.

Runs vision-encoder -> text-encoder -> decoder as chained TRT inference
and compares the final pred_masks output against PyTorch FP32 reference.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from bench_common import get_image_paths, PROMPT

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SPLIT_ENGINE_DIR = BENCHMARK_DIR / "engines" / "split"

IMG_SIZE = 1008
PATCH_SIZE = 14
FEAT = IMG_SIZE // PATCH_SIZE


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TRTModule:
    """Generic TRT engine runner with dynamic shape support."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.input_names = []
        self.output_names = []
        self.input_meta = {}
        self.output_meta = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_meta[name] = {"shape": shape, "dtype": dtype}
            else:
                self.output_names.append(name)
                self.output_meta[name] = {"shape": shape, "dtype": dtype}

    def infer(self, inputs: dict) -> dict:
        """Run inference with explicit input arrays. Handles dynamic shapes."""
        buffers = {}

        # Set inputs
        for name, data in inputs.items():
            t = torch.from_numpy(data).cuda().contiguous()
            buffers[name] = t
            self.context.set_input_shape(name, tuple(data.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        # Allocate outputs based on resolved shapes
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


def get_pytorch_reference(img_path: str):
    """Run full PyTorch FP32 inference."""
    from transformers.models.sam3 import Sam3Processor, Sam3Model

    device = "cuda"
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    pil_img = Image.open(img_path).convert("RGB")
    inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return {
        "pred_masks": outputs.pred_masks.float().cpu().numpy(),
        "pixel_values": inputs["pixel_values"].cpu().numpy(),
        "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"].cpu().numpy().astype(np.int64),
    }


def run_split_pipeline(pixel_values, input_ids, attention_mask):
    """Run the 3-module split TRT pipeline."""
    ve_path = SPLIT_ENGINE_DIR / "vision-encoder_fp16_mixed.plan"
    te_path = SPLIT_ENGINE_DIR / "text-encoder_fp16_mixed.plan"
    dec_path = SPLIT_ENGINE_DIR / "decoder_fp16_mixed.plan"

    for p in [ve_path, te_path, dec_path]:
        if not p.exists():
            print(f"ERROR: Engine not found: {p}")
            sys.exit(1)

    # Vision encoder
    print("  Running vision-encoder...")
    ve = TRTModule(str(ve_path))
    ve_out = ve.infer({"images": pixel_values})

    # Text encoder
    print("  Running text-encoder...")
    te = TRTModule(str(te_path))
    te_out = te.infer({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })

    # Decoder
    print("  Running decoder...")
    dec = TRTModule(str(dec_path))
    # text_mask needs to be bool
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

    return dec_out


def main():
    images = get_image_paths()
    test_img = images[0]
    print(f"Test image: {test_img}")
    print(f"Prompt: '{PROMPT}'")

    # PyTorch reference
    print("\n--- Running PyTorch FP32 reference ---")
    ref = get_pytorch_reference(test_img)
    pt_masks = ref["pred_masks"]
    print(f"  pred_masks: shape={pt_masks.shape} range=[{pt_masks.min():.4f}, {pt_masks.max():.4f}]")

    # Split TRT pipeline
    print("\n--- Running Split TRT Pipeline (FP16 mixed) ---")
    trt_out = run_split_pipeline(
        ref["pixel_values"], ref["input_ids"], ref["attention_mask"]
    )
    trt_masks = trt_out["pred_masks"]
    print(f"  pred_masks: shape={trt_masks.shape} range=[{trt_masks.min():.4f}, {trt_masks.max():.4f}]")

    # Compare
    print("\n--- Fidelity Comparison ---")
    if pt_masks.shape != trt_masks.shape:
        print(f"  SHAPE MISMATCH: PyTorch={pt_masks.shape} TRT={trt_masks.shape}")
        # Try comparing what we can
        min_shape = tuple(min(a, b) for a, b in zip(pt_masks.shape, trt_masks.shape))
        print(f"  Comparing first {min_shape} elements...")
        pt_sub = pt_masks[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
        trt_sub = trt_masks[:min_shape[0], :min_shape[1], :min_shape[2], :min_shape[3]]
    else:
        pt_sub = pt_masks
        trt_sub = trt_masks

    max_abs = np.abs(pt_sub - trt_sub).max()
    mean_abs = np.abs(pt_sub - trt_sub).mean()
    cos = cosine_similarity(pt_sub, trt_sub)

    print(f"  pred_masks:")
    print(f"    max_abs_diff = {max_abs:.6e}")
    print(f"    mean_abs_diff = {mean_abs:.6e}")
    print(f"    cosine_similarity = {cos:.6f}")

    # Also compare other outputs if available
    for key in ["pred_boxes", "pred_logits", "presence_logits"]:
        if key in trt_out:
            print(f"  {key}: shape={trt_out[key].shape} range=[{trt_out[key].min():.4f}, {trt_out[key].max():.4f}]")

    print(f"\n{'='*60}")
    if cos > 0.999:
        print(f"RESULT: PASS - Split pipeline matches PyTorch (cosine={cos:.6f})")
    elif cos > 0.99:
        print(f"RESULT: GOOD - Minor FP16 divergence acceptable (cosine={cos:.6f})")
    elif cos > 0.95:
        print(f"RESULT: MARGINAL - Some accuracy loss (cosine={cos:.6f})")
    else:
        print(f"RESULT: FAIL - Significant divergence (cosine={cos:.6f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
