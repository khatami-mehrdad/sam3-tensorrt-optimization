#!/bin/bash
# Run all benchmark levels and produce a summary table.
# Usage: cd benchmark/scripts && bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="/usr/src/tensorrt/bin:$PATH"

echo "============================================"
echo "  SAM3 Optimization Benchmark Suite"
echo "============================================"
echo ""

# Step 0: Download test data
echo "--- Downloading test data ---"
python3 download_coco.py
echo ""

# Step 1: ONNX export (if not done)
if [ ! -f ../onnx/level1_sam3.onnx ]; then
    echo "--- Exporting ONNX (Level 1) ---"
    python3 export_onnx.py
    echo ""
fi

# Step 2: Optimize ONNX (if not done)
if [ ! -f ../onnx/level2_optimized.onnx ]; then
    echo "--- Optimizing ONNX (Level 2) ---"
    python3 optimize_onnx.py
    echo ""
fi

# Step 3: Build all engines
echo "--- Building TensorRT engines ---"
bash build_engines.sh
echo ""

# Step 4: Run benchmarks
echo ""
echo "=========================================="
echo "  Running Benchmarks"
echo "=========================================="
echo ""

echo "--- Level 0: PyTorch baseline ---"
python3 bench_level0.py
echo ""

echo "--- Level 1: ONNX + TRT FP16 ---"
python3 bench_level1.py
echo ""

if [ -f ../engines/level2_fp16.plan ]; then
    echo "--- Level 2: + onnxsim + polygraphy ---"
    python3 bench_level2.py
    echo ""
fi

if [ -f ../engines/level3_fp16_opt5.plan ]; then
    echo "--- Level 3: + builder opt level 5 ---"
    python3 bench_level3.py
    echo ""
fi

echo "--- Level 4: Encoder/decoder split ---"
python3 bench_level4.py
echo ""

if [ -f ../engines/level5_int8.plan ]; then
    echo "--- Level 5: + INT8 quantization ---"
    python3 bench_level5.py
    echo ""
fi

# Step 5: Print summary table
echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
python3 -c "from bench_common import print_summary_table; print_summary_table()"
