#!/usr/bin/env python3
"""Level 5: INT8 quantization via NVIDIA Model Optimizer."""

import sys
from bench_common import ENGINE_DIR

ENGINE_PATH = ENGINE_DIR / "level5_int8.plan"

LEVEL = 5
LABEL = "+ INT8 quantization"


def main():
    if not ENGINE_PATH.exists():
        print(f"ERROR: Engine not found at {ENGINE_PATH}")
        print("Build it with build_engines.sh first.")
        sys.exit(1)

    import bench_level1
    bench_level1.ENGINE_PATH = ENGINE_PATH
    bench_level1.LEVEL = LEVEL
    bench_level1.LABEL = LABEL

    bench_level1.main()


if __name__ == "__main__":
    main()
