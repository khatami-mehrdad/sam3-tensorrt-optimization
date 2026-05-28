#!/bin/bash
# Build all TensorRT engines for the benchmark pipeline.
# Run from the benchmark/ directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"
ONNX_DIR="$BENCHMARK_DIR/onnx"
ENGINE_DIR="$BENCHMARK_DIR/engines"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

mkdir -p "$ENGINE_DIR"

if [ ! -f "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    exit 1
fi

# Level 1: Baseline FP16 from raw ONNX export
ONNX_L1="$ONNX_DIR/level1_sam3.onnx"
ENGINE_L1="$ENGINE_DIR/level1_fp16.plan"
if [ ! -f "$ONNX_L1" ]; then
    echo "ERROR: $ONNX_L1 not found. Run export_onnx.py first."
    exit 1
fi
if [ ! -f "$ENGINE_L1" ]; then
    echo "=== Building Level 1: FP16 from raw ONNX ==="
    "$TRTEXEC" --onnx="$ONNX_L1" --saveEngine="$ENGINE_L1" --fp16 2>&1 | tail -20
    echo ""
else
    echo "Level 1 engine already exists: $ENGINE_L1"
fi

# Level 2: FP16 from optimized ONNX
ONNX_L2="$ONNX_DIR/level2_optimized.onnx"
ENGINE_L2="$ENGINE_DIR/level2_fp16.plan"
if [ ! -f "$ONNX_L2" ]; then
    echo "WARNING: $ONNX_L2 not found. Run optimize_onnx.py first. Skipping Level 2."
else
    if [ ! -f "$ENGINE_L2" ]; then
        echo "=== Building Level 2: FP16 from optimized ONNX ==="
        "$TRTEXEC" --onnx="$ONNX_L2" --saveEngine="$ENGINE_L2" --fp16 2>&1 | tail -20
        echo ""
    else
        echo "Level 2 engine already exists: $ENGINE_L2"
    fi
fi

# Level 3: FP16 from optimized ONNX with builder optimization level 5
ONNX_L3="$ONNX_L2"  # reuse level 2 ONNX
ENGINE_L3="$ENGINE_DIR/level3_fp16_opt5.plan"
if [ ! -f "$ONNX_L3" ]; then
    echo "WARNING: $ONNX_L3 not found. Skipping Level 3."
else
    if [ ! -f "$ENGINE_L3" ]; then
        echo "=== Building Level 3: FP16 + builder optimization level 5 ==="
        "$TRTEXEC" --onnx="$ONNX_L3" --saveEngine="$ENGINE_L3" --fp16 \
            --builderOptimizationLevel=5 \
            --timingCacheFile="$ENGINE_DIR/timing.cache" 2>&1 | tail -20
        echo ""
    else
        echo "Level 3 engine already exists: $ENGINE_L3"
    fi
fi

# Level 5: INT8 from quantized ONNX
ONNX_L5="$ONNX_DIR/level5_int8.onnx"
ENGINE_L5="$ENGINE_DIR/level5_int8.plan"
if [ ! -f "$ONNX_L5" ]; then
    echo "NOTE: $ONNX_L5 not found. INT8 quantization not yet done. Skipping Level 5."
else
    if [ ! -f "$ENGINE_L5" ]; then
        echo "=== Building Level 5: INT8 from quantized ONNX ==="
        "$TRTEXEC" --onnx="$ONNX_L5" --saveEngine="$ENGINE_L5" --int8 --fp16 2>&1 | tail -20
        echo ""
    else
        echo "Level 5 engine already exists: $ENGINE_L5"
    fi
fi

echo ""
echo "=== Engine inventory ==="
ls -lh "$ENGINE_DIR"/*.plan 2>/dev/null || echo "No engines built."
