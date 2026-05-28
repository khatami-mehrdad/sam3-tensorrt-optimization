#!/usr/bin/env python3
"""Level 1: ONNX export + TensorRT FP16 engine (baseline TRT)."""

import os
import sys

import numpy as np
import tensorrt as trt
import torch
from PIL import Image

from bench_common import (
    benchmark, get_image_paths, save_results,
    ENGINE_DIR, ONNX_DIR, PROMPT,
)

LEVEL = 1
LABEL = "ONNX + TRT FP16"
ENGINE_PATH = ENGINE_DIR / "level1_fp16.plan"
ONNX_PATH = ONNX_DIR / "level1_sam3.onnx"

# SAM3 preprocessing constants (from processor_config.json)
IMG_SIZE = 1008
MEAN = 0.5
STD = 0.5


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize, normalize, transpose to CHW float32."""
    img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, 0).astype(np.float32)  # add batch dim


def get_tokenized_prompt():
    """Tokenize the prompt using the HF processor (done once)."""
    from transformers.models.sam3 import Sam3Processor
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    dummy_img = Image.new("RGB", (224, 224))
    inputs = processor(images=dummy_img, text=PROMPT, return_tensors="pt")
    return inputs["input_ids"].numpy().astype(np.int64), inputs["attention_mask"].numpy().astype(np.int64)


class TRTInference:
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Allocate I/O buffers
        self.buffers = {}
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = 1
            for d in shape:
                size *= max(1, d)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                buf = torch.zeros(size, dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype,
                                  device="cuda")
                self.buffers[name] = {"tensor": buf, "shape": tuple(shape), "dtype": dtype, "mode": "input"}
            else:
                buf = torch.zeros(size, dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype,
                                  device="cuda")
                self.buffers[name] = {"tensor": buf, "shape": tuple(shape), "dtype": dtype, "mode": "output"}
                self.output_names.append(name)
            self.context.set_tensor_address(name, buf.data_ptr())

    def set_input(self, name: str, data: np.ndarray):
        buf = self.buffers[name]["tensor"]
        t = torch.from_numpy(data.ravel()).cuda()
        buf[:t.numel()] = t

    def infer(self):
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()


def main():
    if not ENGINE_PATH.exists():
        print(f"ERROR: Engine not found at {ENGINE_PATH}")
        print(f"Build it first:")
        print(f"  export PATH=/usr/src/tensorrt/bin:$PATH")
        print(f"  trtexec --onnx={ONNX_PATH} --saveEngine={ENGINE_PATH} --fp16 --verbose")
        sys.exit(1)

    print(f"Loading TRT engine from {ENGINE_PATH}...")
    trt_infer = TRTInference(str(ENGINE_PATH))

    # Tokenize prompt once
    input_ids, attention_mask = get_tokenized_prompt()
    trt_infer.set_input("input_ids", input_ids)
    trt_infer.set_input("attention_mask", attention_mask)

    images = get_image_paths()

    # Pre-load PIL images
    pil_images = {}
    for p in images:
        pil_images[p] = Image.open(p).convert("RGB")

    def infer(img_path: str):
        pixel_values = preprocess_image(pil_images[img_path])
        trt_infer.set_input("pixel_values", pixel_values)
        trt_infer.infer()

    results = benchmark(infer, images, label=LABEL)
    results["level"] = LEVEL
    results["engine"] = str(ENGINE_PATH)
    save_results(results, f"level{LEVEL}.json")


if __name__ == "__main__":
    main()
