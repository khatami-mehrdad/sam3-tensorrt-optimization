#!/usr/bin/env python3
"""Apply ONNX graph optimizations: onnxsim + polygraphy constant folding.

Input:  benchmark/onnx/level1_sam3.onnx
Output: benchmark/onnx/level2_optimized.onnx
"""

import os
import sys
from pathlib import Path

ONNX_DIR = Path(__file__).resolve().parent.parent / "onnx"
INPUT_ONNX = ONNX_DIR / "level1_sam3.onnx"
SIMPLIFIED_ONNX = ONNX_DIR / "level2_simplified.onnx"
OUTPUT_ONNX = ONNX_DIR / "level2_optimized.onnx"


def main():
    if not INPUT_ONNX.exists():
        print(f"ERROR: {INPUT_ONNX} not found. Run export_onnx.py first.")
        sys.exit(1)

    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    # Load with external data (model > 2 GB)
    print(f"Loading {INPUT_ONNX} (with external data)...")
    model = onnx.load(str(INPUT_ONNX), load_external_data=True)

    original_nodes = len(model.graph.node)
    print(f"  Original graph: {original_nodes} nodes")

    # Step 1: Try onnxsim (may fail on very large models)
    model_simplified = model
    try:
        from onnxsim import simplify
        print("Running onnxsim (constant folding + dead code elimination)...")
        model_simplified, ok = simplify(model)
        if not ok:
            print("WARNING: onnxsim could not validate the simplified model.")
            model_simplified = model
        simplified_nodes = len(model_simplified.graph.node)
        print(f"  After onnxsim: {simplified_nodes} nodes "
              f"({original_nodes - simplified_nodes} removed)")
    except Exception as e:
        print(f"WARNING: onnxsim failed ({e}), skipping simplification.")
        model_simplified = model

    # Save with external data (required for >2GB models)
    os.makedirs(SIMPLIFIED_ONNX.parent, exist_ok=True)
    convert_model_to_external_data(
        model_simplified,
        all_tensors_to_one_file=True,
        location=SIMPLIFIED_ONNX.name + ".data",
    )
    onnx.save(model_simplified, str(SIMPLIFIED_ONNX))
    print(f"  Saved simplified model to {SIMPLIFIED_ONNX}")

    # Step 2: polygraphy constant folding
    try:
        from polygraphy.backend.onnx import fold_constants
        import onnx_graphsurgeon as gs

        print("Running polygraphy constant folding...")
        graph = gs.import_onnx(model_simplified)
        graph.cleanup()
        onnx_folded = gs.export_onnx(graph)
        onnx_folded = fold_constants(onnx_folded, allow_onnxruntime_shape_inference=True)
        folded_model = onnx.load_from_string(onnx_folded.SerializeToString()) if onnx_folded.ByteSize() < 2_000_000_000 else onnx_folded
        folded_nodes = len(folded_model.graph.node)
        print(f"  After polygraphy: {folded_nodes} nodes")
        convert_model_to_external_data(
            folded_model,
            all_tensors_to_one_file=True,
            location=OUTPUT_ONNX.name + ".data",
        )
        onnx.save(folded_model, str(OUTPUT_ONNX))
    except Exception as e:
        print(f"WARNING: polygraphy step failed: {e}")
        print("Using onnxsim output as level2 ONNX.")
        import shutil
        shutil.copy2(str(SIMPLIFIED_ONNX), str(OUTPUT_ONNX))
        data_file = str(SIMPLIFIED_ONNX) + ".data"
        if os.path.exists(data_file):
            shutil.copy2(data_file, str(OUTPUT_ONNX) + ".data")

    for f in [SIMPLIFIED_ONNX, OUTPUT_ONNX]:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")

    print("Done.")


if __name__ == "__main__":
    main()
