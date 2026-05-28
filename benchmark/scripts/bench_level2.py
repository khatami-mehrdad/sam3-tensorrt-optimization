#!/usr/bin/env python3
"""Level 2: ONNX graph optimization (onnxsim + polygraphy) + TRT FP16."""

import sys
from bench_common import ENGINE_DIR

# Reuse the Level 1 TRT inference harness with a different engine
ENGINE_PATH = ENGINE_DIR / "level2_fp16.plan"

LEVEL = 2
LABEL = "+ onnxsim + polygraphy"


def main():
    if not ENGINE_PATH.exists():
        print(f"ERROR: Engine not found at {ENGINE_PATH}")
        print("Build it with build_engines.sh first.")
        sys.exit(1)

    # Import and patch bench_level1 to use our engine
    import bench_level1
    bench_level1.ENGINE_PATH = ENGINE_PATH
    bench_level1.LEVEL = LEVEL
    bench_level1.LABEL = LABEL

    bench_level1.main()


if __name__ == "__main__":
    main()
