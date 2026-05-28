#!/usr/bin/env python3
"""Build TRT engines for each split SAM3 module with mixed precision.

Builds FP16 engines with Softmax forced to FP32 to prevent the fused
MHA kernel (_gemm_mha_v2) from overflowing in attention layers.

Usage:
    python3 build_split_engines.py --all
    python3 build_split_engines.py --module vision-encoder
    python3 build_split_engines.py --all --fp32   # Pure FP32 for sanity check
"""

import argparse
import sys
from pathlib import Path

import tensorrt as trt

BENCHMARK_DIR = Path(__file__).resolve().parent.parent
SPLIT_ONNX_DIR = BENCHMARK_DIR / "onnx" / "split"
SPLIT_ENGINE_DIR = BENCHMARK_DIR / "engines" / "split"

MODULES = ["vision-encoder", "text-encoder", "decoder"]


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True,
                 force_softmax_fp32: bool = True, workspace_gb: int = 8):
    """Build a TRT engine with optional per-layer precision constraints."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"  Parsing: {Path(onnx_path).name}")
    success = parser.parse_from_file(onnx_path)
    if not success:
        for i in range(parser.num_errors):
            print(f"    Parse error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse {onnx_path}")

    print(f"    {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Set optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        shape = inp.shape
        min_shape = []
        opt_shape = []
        max_shape = []
        for dim_idx, d in enumerate(shape):
            if d == -1:
                if dim_idx == 0:
                    # Batch dimension: fix to 1
                    min_shape.append(1)
                    opt_shape.append(1)
                    max_shape.append(1)
                else:
                    # Sequence/prompt_len dimension: allow range
                    min_shape.append(1)
                    opt_shape.append(32)
                    max_shape.append(77)
            else:
                min_shape.append(d)
                opt_shape.append(d)
                max_shape.append(d)
        profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
    config.add_optimization_profile(profile)

    if force_softmax_fp32 and fp16:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        softmax_count = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type == trt.LayerType.SOFTMAX:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
                softmax_count += 1
        print(f"    Forced {softmax_count} Softmax layers to FP32")

    print("    Building engine...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")

    engine_bytes = bytes(serialized)
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    size_mb = len(engine_bytes) / (1024 * 1024)
    print(f"    Saved: {engine_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build split-module TRT engines")
    parser.add_argument("--module", type=str, choices=MODULES,
                        help="Build a single module")
    parser.add_argument("--all", action="store_true", help="Build all modules")
    parser.add_argument("--fp32", action="store_true", help="Build in FP32 (sanity check)")
    parser.add_argument("--no-constraints", action="store_true",
                        help="Build FP16 without precision constraints (pure FP16)")
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--onnx-dir", type=str, default=str(SPLIT_ONNX_DIR))
    parser.add_argument("--engine-dir", type=str, default=str(SPLIT_ENGINE_DIR))
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.error("Specify --module or --all")

    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    modules = MODULES if args.all else [args.module]

    precision_label = "fp32" if args.fp32 else ("fp16_pure" if args.no_constraints else "fp16_mixed")

    for module_name in modules:
        onnx_path = onnx_dir / f"{module_name}.onnx"
        if not onnx_path.exists():
            print(f"SKIP: {onnx_path} not found")
            continue

        engine_path = engine_dir / f"{module_name}_{precision_label}.plan"
        print(f"\n--- Building {module_name} ({precision_label}) ---")
        build_engine(
            onnx_path=str(onnx_path),
            engine_path=str(engine_path),
            fp16=not args.fp32,
            force_softmax_fp32=(not args.fp32 and not args.no_constraints),
            workspace_gb=args.workspace_gb,
        )

    print("\nAll engines built.")


if __name__ == "__main__":
    main()
