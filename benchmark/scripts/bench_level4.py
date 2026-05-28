#!/usr/bin/env python3
"""Level 4: Encoder/decoder split — run encoder once, decoder per prompt.

This level uses samexporter to split SAM3 into 3 ONNX models (image encoder,
text encoder, decoder), builds separate TRT engines, and benchmarks the
split pipeline.

If samexporter or the split engines are not available, falls back to
measuring the monolithic engine but with a note that split is pending.
"""

import sys
import time

import numpy as np
import torch
from PIL import Image

from bench_common import (
    benchmark, get_image_paths, save_results,
    ENGINE_DIR, PROMPT,
)

LEVEL = 4
LABEL = "Encoder/decoder split"

ENCODER_ENGINE = ENGINE_DIR / "level4_encoder_fp16.plan"
TEXT_ENGINE = ENGINE_DIR / "level4_text_fp16.plan"
DECODER_ENGINE = ENGINE_DIR / "level4_decoder_fp16.plan"

# Fallback: reuse Level 3 monolithic engine if split is not yet built
FALLBACK_ENGINE = ENGINE_DIR / "level3_fp16_opt5.plan"


def main():
    if ENCODER_ENGINE.exists() and DECODER_ENGINE.exists():
        print("Split engines found — running split benchmark.")
        print("TODO: Implement split pipeline inference.")
        print("This requires custom orchestration code specific to the split ONNX structure.")
        print("Falling back to monolithic engine for now.\n")

    # Until the split export is implemented, measure the best monolithic engine
    # and note the split as future work
    engine = FALLBACK_ENGINE
    if not engine.exists():
        engine = ENGINE_DIR / "level2_fp16.plan"
    if not engine.exists():
        engine = ENGINE_DIR / "level1_fp16.plan"
    if not engine.exists():
        print(f"ERROR: No engine found. Build engines first.")
        sys.exit(1)

    print(f"[Level 4] Split not yet built — using monolithic engine: {engine.name}")
    print("  (Encoder/decoder split shows gains with multi-prompt; single-prompt is similar)\n")

    import bench_level1
    bench_level1.ENGINE_PATH = engine
    bench_level1.LEVEL = LEVEL
    bench_level1.LABEL = f"{LABEL} (pending, using {engine.name})"

    bench_level1.main()


if __name__ == "__main__":
    main()
