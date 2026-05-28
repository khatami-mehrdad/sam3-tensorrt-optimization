#!/usr/bin/env python3
"""Export SAM3 to ONNX for the benchmark pipeline.

Adapted from ~/mehrdad/SAM3-TensorRT/python/onnxexport.py.
Outputs to benchmark/onnx/level1_sam3.onnx.
"""

import os
import sys
from pathlib import Path

import torch
from transformers.models.sam3 import Sam3Processor, Sam3Model
from PIL import Image
import requests

ONNX_DIR = Path(__file__).resolve().parent.parent / "onnx"


class Sam3ONNXWrapper(torch.nn.Module):
    def __init__(self, sam3):
        super().__init__()
        self.sam3 = sam3

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.pred_masks, outputs.semantic_seg


def main():
    device = "cpu"  # CPU for maximum ONNX compatibility

    print("Loading Sam3Model...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    prompt = "person"
    image_url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    print(f"Downloading sample image for tracing...")
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"  pixel_values: {pixel_values.shape} {pixel_values.dtype}")
    print(f"  input_ids:    {input_ids.shape} {input_ids.dtype}")
    print(f"  attention_mask: {attention_mask.shape} {attention_mask.dtype}")

    wrapper = Sam3ONNXWrapper(model).to(device).eval()

    os.makedirs(ONNX_DIR, exist_ok=True)
    onnx_path = str(ONNX_DIR / "level1_sam3.onnx")

    print(f"\nExporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        (pixel_values, input_ids, attention_mask),
        onnx_path,
        input_names=["pixel_values", "input_ids", "attention_mask"],
        output_names=["instance_masks", "semantic_seg"],
        dynamo=False,
        opset_version=17,
    )
    print(f"Exported to {onnx_path}")

    # Check file sizes
    for f in ONNX_DIR.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
