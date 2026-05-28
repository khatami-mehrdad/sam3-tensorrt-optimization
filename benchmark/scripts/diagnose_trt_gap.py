#!/usr/bin/env python3
"""Diagnose where the PyTorch vs TRT gap originates.

Tests:
  1. Preprocessing comparison: Sam3Processor vs manual preprocess_image
  2. ONNX Runtime inference (isolates ONNX export from TRT conversion)
  3. TRT with processor-generated inputs (isolates preprocessing from engine)
"""

import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from bench_common import get_image_paths, ENGINE_DIR, ONNX_DIR, PROMPT

IMG_SIZE = 1008
MEAN = 0.5
STD = 0.5


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def manual_preprocess(pil_img: Image.Image) -> np.ndarray:
    """The preprocessing used in bench_level1.py and verify_fidelity.py."""
    img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0).astype(np.float32)


def processor_preprocess(pil_img: Image.Image):
    """The preprocessing used during ONNX export and PyTorch inference."""
    from transformers.models.sam3 import Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt")
    return inputs


def main():
    images = get_image_paths()
    test_img_path = images[0]
    pil_img = Image.open(test_img_path).convert("RGB")
    print(f"Test image: {test_img_path}")
    print(f"Image size: {pil_img.size}")
    print(f"Prompt: '{PROMPT}'")

    # ============================================================
    # TEST 1: Compare preprocessing
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST 1: Preprocessing comparison")
    print(f"{'='*60}")

    manual_pixels = manual_preprocess(pil_img)
    proc_inputs = processor_preprocess(pil_img)
    proc_pixels = proc_inputs["pixel_values"].numpy()

    print(f"\n  Manual preprocess:")
    print(f"    shape={manual_pixels.shape} dtype={manual_pixels.dtype}")
    print(f"    range=[{manual_pixels.min():.4f}, {manual_pixels.max():.4f}]")
    print(f"    mean={manual_pixels.mean():.6f} std={manual_pixels.std():.6f}")

    print(f"\n  Sam3Processor:")
    print(f"    shape={proc_pixels.shape} dtype={proc_pixels.dtype}")
    print(f"    range=[{proc_pixels.min():.4f}, {proc_pixels.max():.4f}]")
    print(f"    mean={proc_pixels.mean():.6f} std={proc_pixels.std():.6f}")

    if manual_pixels.shape != proc_pixels.shape:
        print(f"\n  *** SHAPE MISMATCH: {manual_pixels.shape} vs {proc_pixels.shape} ***")
        print("  This is likely the root cause!")
    else:
        diff = np.abs(manual_pixels - proc_pixels)
        max_diff = diff.max()
        mean_diff = diff.mean()
        cos = cosine_similarity(manual_pixels, proc_pixels)
        print(f"\n  Pixel values difference:")
        print(f"    max_abs_diff={max_diff:.6e}")
        print(f"    mean_abs_diff={mean_diff:.6e}")
        print(f"    cosine={cos:.6f}")
        if max_diff < 1e-5:
            print("    --> IDENTICAL (preprocessing is not the issue)")
        else:
            print("    --> DIFFERENT (preprocessing IS the issue!)")

    # Check tokenization
    print(f"\n  Tokenization check:")
    proc_input_ids = proc_inputs["input_ids"].numpy()
    proc_attn_mask = proc_inputs["attention_mask"].numpy()

    from transformers.models.sam3 import Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    dummy_img = Image.new("RGB", (224, 224))
    dummy_inputs = processor(images=dummy_img, text=PROMPT, return_tensors="pt")
    dummy_input_ids = dummy_inputs["input_ids"].numpy()
    dummy_attn_mask = dummy_inputs["attention_mask"].numpy()

    ids_match = np.array_equal(proc_input_ids, dummy_input_ids)
    mask_match = np.array_equal(proc_attn_mask, dummy_attn_mask)
    print(f"    input_ids match (real img vs dummy img): {ids_match}")
    print(f"    attention_mask match: {mask_match}")
    if not ids_match:
        print(f"    real: shape={proc_input_ids.shape} values={proc_input_ids[:, :10]}")
        print(f"    dummy: shape={dummy_input_ids.shape} values={dummy_input_ids[:, :10]}")

    # ============================================================
    # TEST 2: ONNX Runtime inference (bypasses TRT entirely)
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST 2: ONNX Runtime inference")
    print(f"{'='*60}")

    onnx_path = ONNX_DIR / "level1_sam3.onnx"
    if not onnx_path.exists():
        print(f"  SKIP: {onnx_path} not found")
    else:
        try:
            import onnxruntime as ort

            print(f"  Loading {onnx_path.name}...")
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

            # Run with PROCESSOR inputs (same as PyTorch path)
            ort_inputs = {
                "pixel_values": proc_pixels,
                "input_ids": proc_input_ids.astype(np.int64),
                "attention_mask": proc_attn_mask.astype(np.int64),
            }
            print("  Running ONNX Runtime with processor inputs...")
            ort_outputs = sess.run(None, ort_inputs)
            ort_masks = ort_outputs[0]
            ort_seg = ort_outputs[1]

            print(f"  instance_masks: shape={ort_masks.shape} range=[{ort_masks.min():.4f}, {ort_masks.max():.4f}]")
            print(f"  semantic_seg: shape={ort_seg.shape} range=[{ort_seg.min():.4f}, {ort_seg.max():.4f}]")

            # Compare to PyTorch FP32
            from transformers.models.sam3 import Sam3Model
            device = "cuda"
            model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
            pt_inputs = {k: v.to(device) for k, v in proc_inputs.items()}
            with torch.no_grad():
                pt_out = model(**pt_inputs)
            pt_masks = pt_out.pred_masks.cpu().numpy()
            pt_seg = pt_out.semantic_seg.cpu().numpy()

            print(f"\n  PyTorch FP32 vs ONNX Runtime (same inputs):")
            cos_masks = cosine_similarity(pt_masks, ort_masks)
            cos_seg = cosine_similarity(pt_seg, ort_seg)
            max_diff_masks = np.abs(pt_masks - ort_masks).max()
            max_diff_seg = np.abs(pt_seg - ort_seg).max()
            print(f"    instance_masks: max_diff={max_diff_masks:.6e} cosine={cos_masks:.6f}")
            print(f"    semantic_seg:   max_diff={max_diff_seg:.6e} cosine={cos_seg:.6f}")

            if cos_masks > 0.999:
                print("    --> ONNX export is CORRECT. Problem is in TRT conversion.")
            else:
                print("    --> ONNX export itself is BROKEN!")

            # Also run ORT with MANUAL preprocessing to see if that changes things
            print(f"\n  PyTorch FP32 (processor) vs ONNX Runtime (manual preprocess):")
            ort_inputs_manual = {
                "pixel_values": manual_pixels,
                "input_ids": dummy_input_ids.astype(np.int64),
                "attention_mask": dummy_attn_mask.astype(np.int64),
            }
            ort_out_manual = sess.run(None, ort_inputs_manual)
            cos_masks2 = cosine_similarity(pt_masks, ort_out_manual[0])
            cos_seg2 = cosine_similarity(pt_seg, ort_out_manual[1])
            print(f"    instance_masks: cosine={cos_masks2:.6f}")
            print(f"    semantic_seg:   cosine={cos_seg2:.6f}")

        except ImportError:
            print("  SKIP: onnxruntime not installed (pip install onnxruntime-gpu)")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ============================================================
    # TEST 3: TRT with processor-generated inputs
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST 3: TRT engine with PROCESSOR inputs (vs manual preprocess)")
    print(f"{'='*60}")

    engine_path = ENGINE_DIR / "level1_fp16.plan"
    if not engine_path.exists():
        print(f"  SKIP: {engine_path} not found")
    else:
        import tensorrt as trt

        def run_trt(pixel_values, input_ids, attention_mask):
            logger = trt.Logger(trt.Logger.WARNING)
            with open(str(engine_path), "rb") as f:
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
                size = 1
                for d in shape:
                    size *= max(1, d)
                arr = buffers[name]["tensor"][:size].cpu().numpy().astype(np.float32)
                results[name] = arr.reshape(shape)
            return results

        # TRT with processor inputs
        print("  Running TRT with processor-generated inputs...")
        trt_proc = run_trt(proc_pixels, proc_input_ids.astype(np.int64),
                           proc_attn_mask.astype(np.int64))

        # TRT with manual inputs
        print("  Running TRT with manual-preprocessed inputs...")
        trt_manual = run_trt(manual_pixels, dummy_input_ids.astype(np.int64),
                             dummy_attn_mask.astype(np.int64))

        print(f"\n  TRT (processor inputs) vs TRT (manual inputs):")
        for key in trt_proc:
            cos = cosine_similarity(trt_proc[key], trt_manual[key])
            max_diff = np.abs(trt_proc[key] - trt_manual[key]).max()
            print(f"    {key}: max_diff={max_diff:.6e} cosine={cos:.6f}")

        # Compare TRT processor-inputs vs PyTorch processor-inputs
        print(f"\n  PyTorch FP32 (processor) vs TRT FP16 (processor inputs):")
        try:
            # Reuse pt_masks/pt_seg from above
            cos_masks = cosine_similarity(pt_masks, trt_proc["instance_masks"])
            cos_seg = cosine_similarity(pt_seg, trt_proc["semantic_seg"])
            print(f"    instance_masks: cosine={cos_masks:.6f}")
            print(f"    semantic_seg:   cosine={cos_seg:.6f}")
            if cos_masks > 0.99:
                print("    --> Preprocessing was the issue! TRT engine is fine with correct inputs.")
            else:
                print("    --> Problem persists even with same inputs. TRT engine itself diverges.")
        except NameError:
            print("    (skipped — PyTorch reference not available, run TEST 2 first)")

    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
