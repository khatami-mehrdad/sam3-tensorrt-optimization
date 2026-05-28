#!/usr/bin/env python3
"""Level 0: PyTorch baseline using HuggingFace Sam3Model."""

import torch
from PIL import Image
from transformers.models.sam3 import Sam3Processor, Sam3Model

from bench_common import benchmark, get_image_paths, save_results, PROMPT

LEVEL = 0
LABEL = "PyTorch baseline"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Sam3Model on {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    images = get_image_paths()

    # Pre-load all PIL images into memory so image I/O is not timed
    pil_images = {}
    for p in images:
        pil_images[p] = Image.open(p).convert("RGB")

    def infer(img_path: str):
        pil_img = pil_images[img_path]
        inputs = processor(images=pil_img, text=PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Minimal post-processing to match real usage
        processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

    results = benchmark(infer, images, label=LABEL)
    results["level"] = LEVEL
    save_results(results, f"level{LEVEL}.json")


if __name__ == "__main__":
    main()
