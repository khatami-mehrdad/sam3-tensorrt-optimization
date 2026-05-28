#!/usr/bin/env python3
"""Build TRT engine via the Python API with per-layer precision control.

This bypasses trtexec and allows forcing Softmax + attention MatMul layers
to FP32 while the rest of the network runs in FP16. This prevents the
_gemm_mha_v2 fused kernel from overflowing in Softmax.

Usage:
    python3 build_trt_python.py                  # FP16 with Softmax/attn in FP32
    python3 build_trt_python.py --fp32           # Pure FP32 (sanity check)
    python3 build_trt_python.py --all-fp16       # Pure FP16 (no constraints)
"""

import argparse
import sys
from pathlib import Path

import tensorrt as trt

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
ONNX_DIR = BENCHMARK_DIR / "onnx"
ENGINE_DIR = BENCHMARK_DIR / "engines"


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 force_softmax_fp32: bool = True, force_attn_matmul_fp32: bool = True,
                 workspace_gb: int = 8):
    """Build a TRT engine with optional per-layer precision constraints."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}")
    success = parser.parse_from_file(onnx_path)
    if not success:
        for i in range(parser.num_errors):
            print(f"  ONNX parse error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")

    print(f"  Network: {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  Precision: FP16 enabled")

    if force_softmax_fp32 or force_attn_matmul_fp32:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        softmax_count = 0
        matmul_count = 0

        for i in range(network.num_layers):
            layer = network.get_layer(i)

            if force_softmax_fp32 and layer.type == trt.LayerType.SOFTMAX:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                softmax_count += 1

            if force_attn_matmul_fp32 and layer.type == trt.LayerType.MATRIX_MULTIPLY:
                if "attn" in layer.name.lower() or "attention" in layer.name.lower():
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
                    matmul_count += 1

        print(f"  Forced FP32: {softmax_count} Softmax layers, {matmul_count} attention MatMul layers")

    print("  Building engine (this may take several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    engine_bytes = bytes(serialized)
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = len(engine_bytes) / (1024 * 1024)
    print(f"  Saved: {engine_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build TRT engine with precision control")
    parser.add_argument("--onnx", type=str, default=str(ONNX_DIR / "level1_sam3.onnx"),
                        help="Path to ONNX model")
    parser.add_argument("--output", type=str, default=None,
                        help="Output engine path (auto-generated if not set)")
    parser.add_argument("--fp32", action="store_true",
                        help="Build pure FP32 engine (sanity check)")
    parser.add_argument("--all-fp16", action="store_true",
                        help="Build pure FP16 engine (no precision constraints)")
    parser.add_argument("--workspace-gb", type=int, default=8,
                        help="Workspace size in GB")
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        print(f"ERROR: ONNX model not found: {args.onnx}")
        sys.exit(1)

    if args.fp32:
        label = "fp32_pyapi"
        fp16 = False
        force_softmax = False
        force_attn = False
    elif args.all_fp16:
        label = "fp16_pyapi_noconstrain"
        fp16 = True
        force_softmax = False
        force_attn = False
    else:
        label = "fp16_mixed_pyapi"
        fp16 = True
        force_softmax = True
        force_attn = True

    output = args.output or str(ENGINE_DIR / f"{label}.plan")

    build_engine(
        onnx_path=args.onnx,
        engine_path=output,
        fp16=fp16,
        force_softmax_fp32=force_softmax,
        force_attn_matmul_fp32=force_attn,
        workspace_gb=args.workspace_gb,
    )
    print("Done.")


if __name__ == "__main__":
    main()
