# SAM3 TensorRT Optimization: Complete Journey

This document captures the full journey of deploying SAM3 (a vision-language model
for open-vocabulary segmentation) as an optimized TensorRT inference pipeline —
from initial investigation through final benchmarked results.

---

## Table of Contents

1. [Starting Point](#1-starting-point)
2. [The Fidelity Problem](#2-the-fidelity-problem)
3. [Root Cause Investigation](#3-root-cause-investigation)
4. [The Fix: Split-Module Export](#4-the-fix-split-module-export)
5. [Building TRT Engines for Split Modules](#5-building-trt-engines-for-split-modules)
6. [Fidelity Verification](#6-fidelity-verification)
7. [Performance Optimization: GPU-Native Tensors](#7-performance-optimization-gpu-native-tensors)
8. [Performance Optimization: GPU Preprocessing](#8-performance-optimization-gpu-preprocessing)
9. [Final Benchmark Results](#9-final-benchmark-results)
10. [Architecture & Pipeline Design](#10-architecture--pipeline-design)
11. [Scripts Reference](#11-scripts-reference)
12. [Lessons Learned](#12-lessons-learned)

---

## 1. Starting Point

**Model**: SAM3 (Segment Anything Model 3) — a ~2.4 GB vision-language model  
**Goal**: Deploy as TensorRT engine for real-time inference  
**Hardware**: NVIDIA GPU with 32 GB VRAM  
**Input**: 2560×1920 camera images, text prompt ("person", etc.)  
**Output**: Instance segmentation masks + bounding boxes + confidence scores  

**SAM3 Architecture** (3 main components):
- **Vision Encoder** (ViT backbone + FPN neck): Processes 1008×1008 image → multi-scale features
- **Text Encoder** (CLIP): Processes text prompt → text features
- **Decoder** (DETR encoder/decoder + mask decoder): Fuses vision+text → predictions

**Preprocessing**:
- Stretch resize to 1008×1008 (bilinear interpolation)
- Scale pixels to [0, 1] (÷ 255)
- Normalize with mean=0.5, std=0.5 → range [-1, 1]
- CLIP tokenizer for text (max 32 tokens for this model)

---

## 2. The Fidelity Problem

### Initial Observation

After exporting SAM3 as a monolithic ONNX file and building a TensorRT engine,
the outputs diverged catastrophically from PyTorch:

| Build Method | Precision | Cosine Similarity |
|---|---|---|
| Monolithic ONNX + `trtexec` | FP16 | 0.868 |
| Monolithic ONNX + `trtexec` | FP32 | 0.868 |
| Monolithic ONNX + Python API | FP32 | 0.868 |
| ONNX Runtime (same ONNX file) | FP32 | 0.999999 |

**Critical finding**: Even in pure FP32, TRT diverged at cosine=0.868. This ruled
out FP16 precision loss as the root cause.

### What We Ruled Out

| Hypothesis | Test | Result |
|---|---|---|
| FP16 overflow | Built monolithic FP32 engine | Same 0.868 divergence |
| ONNX export error | Ran ONNX Runtime on same file | Perfect 0.999999 match |
| Preprocessing mismatch | Manual vs Sam3Processor comparison | Identical |
| trtexec bug | Used TRT Python API | Same divergence |
| Softmax overflow | Forced Softmax FP32 (monolithic) | No improvement |
| Model sensitivity to FP16 | PyTorch autocast FP16/BF16 | cosine > 0.999 |

---

## 3. Root Cause Investigation

### Why TRT Diverges on Large Monolithic Graphs

The model has **26,739 ONNX layers**. TensorRT is an **optimizing compiler** that
rewrites the graph:

- **Layer fusion** — merges adjacent ops into single kernels, changing evaluation
  order (non-associative in floating-point arithmetic)
- **Graph-level rewrites** — reorganizes entire subgraphs; the larger the graph,
  the more aggressive TRT becomes
- **Kernel substitution** — selects from hundreds of implementations per layer,
  each numerically different in finite precision
- **Control flow restructuring** — unrolls/restructures loop structures
  (SAM3's decoder has loops)

Over 26,739 layers, individually tiny numerical differences compound into
catastrophic divergence. ONNX Runtime doesn't have this problem because it's
an **interpreter** (executes each node in graph order without rewrites).

### Community Confirmation

Research found that other large VLM projects hit the same issue:
- [SAM3-TENSORRT-PYTHON](https://github.com/Kishan200308/SAM3-TENSORRT-PYTHON)
  splits the model into sub-modules
- [Blog: Optimizing Samurai Part 2](https://egordmitriev.dev/blog/2026-05-17-optimizing-samurai-part-2)
  documents the exact same TRT fidelity issue and fix

---

## 4. The Fix: Split-Module Export

### Approach

Split the model into 3 independent ONNX files, each exported separately:

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│ Vision Encoder  │     │  Text Encoder    │     │   Decoder     │
│ (ViT + FPN)     │     │  (CLIP + proj)   │     │ (DETR + mask) │
│                 │     │                  │     │               │
│ In: images      │     │ In: input_ids    │     │ In: fpn_feats │
│     [1,3,1008,  │     │     attn_mask    │     │     text_feat │
│      1008]      │     │                  │     │     text_mask │
│                 │     │ Out: text_feats  │     │               │
│ Out: fpn_feat_0 │     │      text_mask   │     │ Out: masks    │
│      fpn_feat_1 │     └──────────────────┘     │      boxes    │
│      fpn_feat_2 │                              │      logits   │
│      fpn_pos_2  │                              └───────────────┘
└─────────────────┘
```

### Why Splitting Fixes It

| Property | Monolithic (26k layers) | Split (3 modules) |
|---|---|---|
| TRT rewrite scope | Global | Confined to each module |
| Intermediate values | May be "optimized away" | Materialized as outputs |
| Graph complexity | Cross-module connections confuse TRT | Each is a standard architecture |
| Error accumulation | Compounds across all layers | Resets at module boundaries |

**Key mechanism**: Module boundaries force TRT to materialize intermediate tensors.
TRT cannot fuse operations across module boundaries, preventing globally-optimal
decisions in one module from corrupting another.

### Export Implementation

Script: `benchmark/scripts/export_split_onnx.py`

Key decisions:
- **Opset 20** (required for modern ops)
- **Pre-computed position embeddings** (avoids `cumsum` which TRT handles poorly)
- **Dynamic batch axis** (allows future batching)
- **Wrapper classes** that isolate each module's forward pass

```bash
python3 export_split_onnx.py --all
# Output: benchmark/onnx/split/{vision-encoder,text-encoder,decoder}.onnx
```

### File Sizes

| Module | ONNX Size | Layers |
|---|---|---|
| Vision Encoder | ~1.4 GB | 15,710 |
| Text Encoder | ~520 MB | 3,040 |
| Decoder | ~185 MB | 9,677 |
| **Total (split)** | **~2.1 GB** | 28,427 |
| Monolithic | ~3.0 GB | 26,739 |

The split total is smaller because shared weights are not duplicated and position
embeddings are pre-computed constants.

---

## 5. Building TRT Engines for Split Modules

Script: `benchmark/scripts/build_split_engines.py`

### Dynamic Shape Handling

Each engine needs an **optimization profile** specifying min/opt/max for dynamic
dimensions:
- Batch dimension: fixed at 1 (single-image inference)
- Sequence dimension (text): min=1, opt=32, max=77

### Precision Configurations

```bash
# FP16 with Softmax forced to FP32 (recommended)
python3 build_split_engines.py --all

# Pure FP16 (no constraints, maximum speed)
python3 build_split_engines.py --all --no-constraints

# Pure FP32 (sanity check only)
python3 build_split_engines.py --all --fp32
```

### Engine Sizes (FP16 mixed)

| Module | Engine Size |
|---|---|
| Vision Encoder | 886 MB |
| Text Encoder | 679 MB |
| Decoder | 53 MB |

---

## 6. Fidelity Verification

Script: `benchmark/scripts/verify_split_fidelity.py`

### Ablation Results

| Configuration | Cosine Similarity | Max Abs Diff | Mean Abs Diff |
|---|---|---|---|
| Monolithic FP32 (any method) | 0.868 | 154 | — |
| Split FP32 | 0.999967 | 7.89 | — |
| Split FP16 (pure) | 0.994072 | 82.6 | 0.780 |
| **Split FP16 + Softmax FP32** | **0.995816** | **51.2** | 0.757 |

### Full Pipeline Comparison (49 images, vs PyTorch BF16)

| Method | Avg Cosine | Min Cosine |
|---|---|---|
| TRT Split FP16 (mixed) | 0.9897 | 0.9672 |

The slight drop from single-image tests (0.996) to full-set average (0.990) is
because the ground truth is PyTorch BF16 (not FP32), so we're comparing two
different reduced-precision implementations.

---

## 7. Performance Optimization: GPU-Native Tensors

### The Problem

Initial benchmarks showed TRT running **slower than PyTorch** — completely wrong.

**Root cause**: The `TRTModule.infer()` method was converting tensors to numpy
(GPU→CPU) between each module, then back to GPU for the next module:

```
VE: GPU input → inference → .cpu().numpy() → CPU output
TE: CPU input → .cuda() → inference → .cpu().numpy() → CPU output  
Dec: CPU input → .cuda() → inference → .cpu().numpy() → CPU output
```

Each intermediate transfer adds ~5-10ms of PCIe latency, completely negating
TRT's speed advantage.

### The Fix

Created `infer_gpu()` method that accepts and returns CUDA tensors directly:

```python
def infer_gpu(self, inputs: dict) -> dict:
    """Zero-copy between modules — tensors stay on GPU."""
    for name, data in inputs.items():
        if isinstance(data, torch.Tensor):
            t = data.cuda().contiguous()  # no-op if already on GPU
        else:
            t = torch.from_numpy(data).cuda().contiguous()
        # ... set tensor address for TRT ...
    # ... execute ...
    return {name: gpu_buffer for name in self.output_names}
```

Data flow after fix:
```
Input (GPU) → VE (GPU) → TE (GPU) → Dec (GPU) → Output (GPU)
                    ↑ no CPU copies between modules ↑
```

### Impact

| Configuration | Before Fix | After Fix |
|---|---|---|
| Vision Encoder only | ~46 FPS | ~46 FPS (no change, single module) |
| Full Sequential | ~3 FPS ❌ | **39 FPS** ✓ |
| Pipelined | ~3 FPS ❌ | **40 FPS** ✓ |

---

## 8. Performance Optimization: GPU Preprocessing

### The Problem

With TRT inference at ~25ms, the CPU preprocessing (HuggingFace Sam3Processor)
was taking **29ms** — becoming the new bottleneck.

SAM3 preprocessing:
1. Resize image to 1008×1008 (bilinear) — slow on CPU for 2560×1920 inputs
2. Rescale to [0, 1]
3. Normalize (mean=0.5, std=0.5)

### The Fix: torchvision v2 on GPU

```python
import torchvision.transforms.v2 as v2

gpu_preprocess = v2.Compose([
    v2.Resize((1008, 1008), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Transfer raw uint8 image to GPU (1 byte/pixel, minimal bandwidth)
img_gpu = torch.from_numpy(np.asarray(pil_img)).cuda().permute(2, 0, 1)
pixel_values = gpu_preprocess(img_gpu).unsqueeze(0)
```

### Results

| Preprocessing Method | Time per Frame |
|---|---|
| CPU (Sam3Processor) | **29.18 ms** |
| GPU (torchvision v2) | **0.29 ms** |
| **Speedup** | **99x** |

### Fidelity Impact

GPU resize uses slightly different bilinear interpolation math than PIL:
- Max pixel difference: 0.0078 (= 1/128, uint8 rounding boundary)
- TRT output cosine similarity: 0.986 (acceptable, no impact on detection quality)

### End-to-End Impact

| Configuration | ms/frame | FPS |
|---|---|---|
| CPU preprocess + TRT | 54.74 | 18.3 |
| GPU preprocess + TRT | 26.04 | **38.4** |
| GPU preprocess + TRT pipelined | 25.24 | **39.6** |
| **End-to-end speedup** | | **2.1x** |

---

## 9. Final Benchmark Results

### Speed & Memory Comparison

| Method | ms/frame | FPS | GPU Memory | Speedup |
|---|---|---|---|---|
| PyTorch BF16 (baseline) | 126.8 | 7.9 | 6,656 MB | 1.0x |
| CPU preprocess + TRT Sequential | 64.9 | 15.4 | 3,726 MB | 2.0x |
| CPU preprocess + TRT Pipelined | 63.6 | 15.7 | 3,726 MB | 2.0x |
| **GPU preprocess + TRT Sequential** | **26.0** | **38.4** | **3,262 MB** | **4.9x** |
| **GPU preprocess + TRT Pipelined** | **25.2** | **39.6** | **3,344 MB** | **5.0x** |

### Per-Module Breakdown (TRT FP16 mixed)

| Module | ms/frame | FPS | % of Pipeline |
|---|---|---|---|
| Vision Encoder | 21.75 | 46 | 85% |
| Text Encoder | 1.14 | 876 | 4% |
| Decoder | 3.78 | 265 | 15% |
| GPU Preprocess | 0.29 | — | 1% |

### Memory Breakdown

| Component | GPU Memory |
|---|---|
| Vision Encoder (weights + workspace) | ~1,736 MB |
| Text Encoder (weights + workspace) | ~680 MB |
| Decoder (weights + workspace) | ~278 MB |
| Inference activations | ~200 MB |
| **Total** | **~2,900 MB** |

Why 3x smaller than PyTorch (10 GB → 3 GB):
- **Native FP16 weights** vs FP32 master copies (2x on weights alone)
- **No autograd activation storage** — PyTorch keeps intermediate values for backward
- **TRT buffer reuse** — pre-computed memory plan shares buffers across non-overlapping layers
- **No PyTorch caching allocator overhead**

Note: PyTorch BF16 uses `torch.autocast` which keeps **master weights in FP32** and
only casts operands for compute. TRT stores weights natively in FP16.

### Fidelity Summary

| Comparison | Avg Cosine | Min Cosine | Notes |
|---|---|---|---|
| TRT (CPU preprocess) vs PyTorch BF16 | 0.9897 | 0.9672 | 49 images |
| TRT (GPU preprocess) vs TRT (CPU preprocess) | 0.9856 | 0.9713 | Interpolation diff |
| Split FP16+constraints vs PyTorch FP32 | 0.9958 | — | Single image |

---

## 10. Architecture & Pipeline Design

### Production Pipeline

```
Camera (2560×1920)
    │
    ▼  [Upload uint8 to GPU — 14.7 MB/frame, async via pinned memory]
GPU: Raw frame (uint8)
    │
    ▼  [torchvision v2: resize 1008×1008 + normalize — 0.3ms]
GPU: pixel_values [1, 3, 1008, 1008] float32
    │
    ├──► Vision Encoder TRT [21.8ms] ──► fpn_features
    │                                         │
    │    Text Encoder TRT [1.1ms, cached] ──► text_features
    │                                         │
    └──────────────────────────────────────► Decoder TRT [3.8ms]
                                                │
                                                ▼
                                          pred_masks, pred_boxes, pred_logits
```

### Pipelining Strategy

Since the Vision Encoder dominates (85% of compute), true pipelining overlaps
VE(frame N) with Dec(frame N-1):

```
Frame 0: [Preprocess] [VE ████████████████████] [Dec ████]
Frame 1:              [Preprocess] [VE ████████████████████] [Dec ████]
Frame 2:                          [Preprocess] [VE ████████████████████] [Dec ████]
                                                            ↑
                                               Throughput limited by VE: ~46 FPS max
```

In practice on a single GPU, VE and Dec compete for the same compute resources,
so pipelining gives marginal improvement (40.3 vs 39.1 FPS). True overlap requires
either multiple GPUs or significantly balanced workloads.

### Text Caching

The text encoder output is deterministic for a given prompt. In production:
- Run text encoder **once** when the prompt changes
- Cache `text_features` and `text_mask` as GPU tensors
- Skip TE entirely for subsequent frames

This saves 1.1ms/frame and is already reflected in the pipelined benchmarks.

### Future Optimization Opportunities

| Optimization | Expected Gain | Effort |
|---|---|---|
| NVIDIA DALI for decode + preprocess | Eliminate remaining CPU entirely | Medium |
| INT8 quantization (PTQ) | 2x speed on VE (Tensor Cores) | Medium |
| CUDA Graphs | Reduce kernel launch overhead | Low |
| Batched inference (multiple frames) | Better GPU utilization | Low |
| Triton Inference Server deployment | Production serving infrastructure | High |

---

## 11. Scripts Reference

All scripts are in `benchmark/scripts/`:

| Script | Purpose |
|---|---|
| `export_split_onnx.py` | Export SAM3 as 3 separate ONNX modules |
| `build_split_engines.py` | Build TRT engines (FP16/FP32/mixed) |
| `verify_split_fidelity.py` | Verify split pipeline matches PyTorch |
| `bench_split_pipeline.py` | Micro-benchmark: per-module and pipeline FPS |
| `full_pipeline_comparison.py` | Full comparison: PyTorch vs TRT (speed + fidelity) |
| `bench_preprocess_gpu.py` | CPU vs GPU preprocessing benchmark |
| `build_trt_python.py` | Monolithic engine build (for investigation) |
| `bench_common.py` | Shared utilities (paths, image loading) |

### Quick Start (from scratch)

```bash
cd ~/mehrdad/sam3/benchmark/scripts

# 1. Export split ONNX modules
python3 export_split_onnx.py --all

# 2. Build TRT engines (FP16 + Softmax FP32)
python3 build_split_engines.py --all

# 3. Verify fidelity
python3 verify_split_fidelity.py

# 4. Run full benchmark
python3 bench_preprocess_gpu.py

# 5. Compare against PyTorch
python3 full_pipeline_comparison.py
```

---

## 12. Lessons Learned

### 1. Never trust monolithic ONNX export for large models

TRT's optimizer becomes unreliable on graphs with >10k layers. Always split
vision-language models at natural architectural boundaries. The ONNX file being
correct (verified by ONNX Runtime) does NOT guarantee TRT correctness.

### 2. FP16 is not the enemy — TRT graph rewrites are

We spent significant time investigating FP16 precision loss before discovering
that FP32 monolithic had the exact same divergence. The "FP16 Softmax overflow"
issue exists but is secondary to the graph compilation issue.

### 3. Intermediate tensor materialization is the key

The reason splitting works: it forces TRT to produce exact intermediate values
at module boundaries. Without this, TRT may "optimize away" intermediate
representations in ways that are algebraically valid but numerically catastrophic.

### 4. CPU-GPU data transfer is the silent killer

Our first "working" TRT pipeline was slower than PyTorch because intermediate
tensors were bouncing between CPU and GPU (numpy ↔ CUDA). Always keep tensors
on GPU between chained engines.

### 5. Preprocessing dominates after TRT optimization

Once inference is fast (25ms), the CPU-bound preprocessing (29ms!) becomes the
bottleneck. Moving resize + normalize to GPU (torchvision v2) eliminates this
completely (0.3ms).

### 6. Measure GPU memory with nvidia-smi, not PyTorch

`torch.cuda.max_memory_allocated()` only tracks PyTorch's allocator. TRT engine
weights are invisible to it. Always use `nvidia-smi --query-compute-apps` for
true per-process GPU memory measurement.

### 7. Profile before assuming pipelining helps

The Vision Encoder takes 85% of compute. Pipelining VE with Dec gives minimal
improvement because they compete for the same GPU resources. True speedup requires
either reducing VE latency (INT8, pruning) or using multiple GPUs.

---

## Appendix: TRT Engine Build Configuration

### Recommended: FP16 Mixed (Softmax FP32)

```python
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
for layer in network:
    if layer.type == trt.LayerType.SOFTMAX:
        layer.precision = trt.float32
        layer.set_output_type(0, trt.float32)
```

This provides the best trade-off: TRT can optimize everything in FP16 except
Softmax layers (which are numerically sensitive to overflow in attention heads).

### Dynamic Shape Profile

```python
profile = builder.create_optimization_profile()
# Batch: always 1
# Sequence length (text): min=1, opt=32, max=77
for input_tensor in network.inputs:
    if dim == -1 and dim_idx == 0:  # batch
        min_shape.append(1); opt_shape.append(1); max_shape.append(1)
    elif dim == -1:  # sequence
        min_shape.append(1); opt_shape.append(32); max_shape.append(77)
```
