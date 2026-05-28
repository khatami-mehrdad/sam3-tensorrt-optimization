# SAM3 to TensorRT: Complete Optimization Guide

> **See also**: [SAM3_TRT_INSIGHTS.md](SAM3_TRT_INSIGHTS.md) — empirical findings from our TRT fidelity investigation (ablation results, root cause analysis, open questions).

This document catalogs every optimization technique for deploying a VLM like
SAM3/SAM3.1 as a TensorRT engine. Techniques are organized by whether they
are **lossless** (bit-exact or numerically equivalent) or **lossy** (trade
accuracy for speed). Lossless optimizations come first.

---

## The Optimization Stack

```
 ┌──────────────────────────────────────────────────────────┐
 │  LOSSLESS (no accuracy impact)                           │
 │                                                          │
 │  1. ONNX Export & Graph Cleanup                          │
 │  2. TensorRT Engine Build (FP32 or matched precision)    │
 │  3. Encoder/Decoder Split                                │
 │  4. CUDA Pre/Post Processing Kernels                     │
 │  5. CUDA Graphs                                          │
 │  6. Memory Optimization (pinned, zero-copy, async)       │
 │  7. Batching & Pipelining                                │
 ├──────────────────────────────────────────────────────────┤
 │  LOSSY (trade accuracy for speed)                        │
 │                                                          │
 │  8. FP16 Precision                                       │
 │  9. INT8 / FP8 Quantization (PTQ)                        │
 │ 10. INT4 / Mixed-Precision Quantization                  │
 │ 11. Quantization-Aware Training (QAT)                    │
 │ 12. Structured Pruning                                   │
 │ 13. Knowledge Distillation                               │
 └──────────────────────────────────────────────────────────┘
```

Note: FP16 is technically lossy but the accuracy impact on vision models is
nearly always negligible. It is the standard deployment precision and the
SAM3-TensorRT repo uses it by default.

---

# Part 1: Lossless Optimizations

These produce numerically identical (or FP32-equivalent) results. There is
no reason not to apply all of them.

---

## 1. ONNX Export & Graph Cleanup

### 1A. Export: Choose the Best Exporter

PyTorch offers two ONNX exporters. The newer dynamo-based exporter produces
cleaner graphs:

| | Legacy (`dynamo=False`) | Dynamo (`dynamo=True`) |
|---|---|---|
| How | Traces forward pass, records ops | torch.export captures ATen graph |
| Graph quality | May have redundant ops | Normalized, cleaner |
| Stability | More stable for complex models | May fail on unsupported ops |
| Default in PyTorch 2.7+ | No | Yes |

```python
# Best practice: try dynamo first, fall back to legacy
torch.onnx.export(
    model, args, "model.onnx",
    dynamo=True,         # cleaner graph
    optimize=True,       # built-in constant folding
    opset_version=17,
)
```

The current SAM3-TensorRT repo uses legacy (`dynamo=False`). Switching to
dynamo is a free improvement if it works.

**Impact**: 5-15% smaller graph, faster TRT build  
**Effort**: Low (change one flag)

### 1B. ONNX Simplifier (onnxsim)

Runs the graph with sample data, replaces subgraphs with their constant
outputs (constant folding), removes dead nodes.

```bash
pip install onnx-simplifier
onnxsim sam3_dynamic.onnx sam3_simplified.onnx
```

Or in Python:
```python
import onnx
from onnxsim import simplify
model = onnx.load("sam3_dynamic.onnx")
model_simplified, ok = simplify(model)
onnx.save(model_simplified, "sam3_simplified.onnx")
```

**Impact**: 5-20% fewer nodes  
**Effort**: One command

### 1C. Polygraphy Graph Surgery

NVIDIA's tool for deeper graph cleanup -- constant folding, dead code
elimination, shape inference. Catches patterns onnxsim misses.

```bash
pip install polygraphy onnx_graphsurgeon
polygraphy surgeon sanitize sam3_dynamic.onnx \
  --fold-constants -o sam3_folded.onnx
```

Or programmatically (as used by NVIDIA's FasterViT):
```python
from polygraphy.backend.onnx import fold_constants
import onnx_graphsurgeon as gs

graph = gs.import_onnx(onnx.load("model.onnx"))
graph.cleanup()
onnx_model = gs.export_onnx(graph)
onnx_model = fold_constants(onnx_model, allow_onnxruntime_shape_inference=True)
onnx.save(onnx_model, "model_optimized.onnx")
```

**Impact**: Complementary to onnxsim  
**Effort**: Low

### 1D. ONNX Runtime Transformer Fusions

ORT has transformer-specific graph rewrites: fuse multi-head attention,
LayerNorm, GELU, Bias+Skip patterns into single optimized ops.

| Level | What |
|-------|------|
| O1 | Basic: constant folding, dead code elimination |
| O2 | Extended: attention fusion, LayerNorm fusion, GELU fusion |
| O3 | O2 + GELU approximation (this one is lossy) |

```python
from optimum.onnxruntime import ORTOptimizer, AutoOptimizationConfig
optimizer = ORTOptimizer.from_pretrained(model_path)
optimizer.optimize(save_dir="optimized/",
                   optimization_config=AutoOptimizationConfig.O2())
```

Note: O3's GELU approximation is slightly lossy. Stick to O2 for lossless.

**Impact**: 10-30% for transformer-heavy models  
**Effort**: Low

---

## 2. TensorRT Engine Build

These optimizations happen automatically inside TensorRT when you build
the engine. You just need to provide a clean ONNX graph (from step 1).

### 2A. Layer Fusion (Automatic)

TensorRT fuses adjacent operations into single GPU kernels:
- Conv + BatchNorm + ReLU -> 1 kernel
- MatMul + Add + GELU -> 1 kernel
- LayerNorm components -> 1 kernel

Fewer kernels = less launch overhead + fewer intermediate memory buffers.
This is fully automatic. The quality depends on how clean your ONNX is.

### 2B. Multi-Head Attention Fusion (Automatic)

TensorRT can fuse the full Q/K/V -> attention -> output into a single
optimized kernel. This reduces memory from O(S^2) to O(S).

Requirements (TRT 10.x):
- Q, K, V clearly separable in the graph
- Mask must use pointwise ops feeding a single Add before Softmax
- Must be Where, Add, or Subtract

If it fails, check `trtexec --verbose` output. May need attention graph
restructuring (see Section 3C).

### 2C. Kernel Auto-Tuning

TensorRT benchmarks many kernel implementations per layer and picks the
fastest for your specific GPU. Higher optimization levels try more options.

```bash
# Default (level 3)
trtexec --onnx=model.onnx --saveEngine=engine.plan --fp16

# Maximum (level 5) -- slower build, potentially faster engine
trtexec --onnx=model.onnx --saveEngine=engine.plan --fp16 \
  --builderOptimizationLevel=5

# Cache results to speed up rebuilds
trtexec --onnx=model.onnx --saveEngine=engine.plan --fp16 \
  --timingCacheFile=timing.cache
```

**Impact**: 5-15% (level 5 vs default)  
**Effort**: One flag

---

## 3. Encoder/Decoder Split

### Can the Prompt Be Entangled with the Image Encoder?

**Yes, in some VLM architectures -- but NOT in SAM3.**

Here is SAM3's data flow, verified from the source code in
`sam3/model/vl_combiner.py`:

```
                    Image                          Text prompt
                      |                                |
                      v                                v
              ViT Backbone (32 layers)         Text Encoder (CLIP, 24 layers)
                      |                                |
                      v                                v
              FPN Neck (multi-scale)           Text embeddings (256-d)
                      |                                |
                      +----------- INDEPENDENT --------+
                      |                                |
                      v                                v
              vision_features                  language_features
                      |                                |
                      +---> Transformer Encoder (fusion happens HERE)
                                      |
                                      v
                              Transformer Decoder (200 queries)
                                      |
                                      v
                              pred_masks, semantic_seg
```

The key code is `SAM3VLBackbone.forward()`:

```python
def forward(self, samples, captions, ...):
    output = self.forward_image(samples)        # vision ONLY
    output.update(self.forward_text(captions))   # text ONLY
    return output
```

`forward_image` and `forward_text` are completely independent. The ViT
backbone never sees the text prompt. They are combined only later in
the Transformer Encoder via cross-attention.

**This means SAM3 splits cleanly into 3 engines:**

```
[Engine 1: Image Encoder]     ViT + FPN         ~90% of compute
[Engine 2: Text Encoder]      CLIP text encoder  ~2% of compute
[Engine 3: Decoder]           Fusion + decoder   ~8% of compute
```

### When Is Splitting NOT Possible?

Some VLM architectures fuse text into the vision backbone early:

| Architecture | Entangled? | Why |
|---|---|---|
| SAM3 / SAM3.1 | **No** | Vision and text are independent until fusion transformer |
| CLIP (dual encoder) | **No** | Separate image/text towers |
| Florence-2 | **Yes** | DaViT backbone has text-conditioned layers |
| CoCa | **Partially** | Shared decoder attends to image during captioning |
| Flamingo | **Yes** | Gated cross-attention layers interleave with vision blocks |
| PaLI | **Yes** | ViT output is projected into the language model's input space, but ViT itself is independent. So ViT splits cleanly; the rest does not. |

**Rule of thumb**: If the vision backbone's `forward()` takes only image
tensors and no text, it can be split. If it takes both, it cannot.

### Why Split?

- **Run encoder once per image, decoder once per prompt.** For N prompts,
  this is Nx faster than running the full model N times.
- **Cache encoder features.** In video, consecutive frames can share encoder
  results.
- **Independent optimization.** Encoder (compute-heavy) and decoder
  (memory-bound) may benefit from different quantization strategies.

The `samexporter` tool already supports SAM3 three-way split:
```bash
pip install samexporter
# Produces: image_encoder.onnx, language_encoder.onnx, decoder.onnx
```

The "Detect Anything in Real Time" paper (arxiv 2603.11441, 2026) uses
this exact split for SAM3 and reports 5.6x speedup at 3 classes, scaling
to 25x at 80 classes.

**Impact**: 1x (single prompt) to 25x+ (many prompts)  
**Effort**: Medium

---

## 4. CUDA Pre/Post Processing Kernels

Move image preprocessing and mask postprocessing from CPU to GPU to avoid
CPU<->GPU round-trips.

### What Moves to CUDA

| Operation | CPU version | CUDA kernel |
|-----------|------------|-------------|
| Resize | `cv2.resize()` or PIL | Nearest-neighbor in custom kernel |
| Normalize | `(img/255 - 0.5) / 0.5` | Fused with resize |
| HWC -> CHW | `np.transpose()` | Fused with resize |
| Sigmoid | `scipy.special.expit()` | `1/(1+exp(-x))` on GPU |
| Threshold | `mask > 0.5` | Fused with sigmoid |
| Overlay | `cv2.addWeighted()` | Alpha-blend in kernel |

The SAM3-TensorRT repo already does all of this. The preprocess kernel
combines resize + scale + normalize + transpose into a single launch with
thread coarsening (each thread handles a 2x2 pixel block).

**Impact**: 1.2-1.5x vs CPU pre/post  
**Effort**: Medium (custom CUDA kernels)

---

## 5. CUDA Graphs

Record the entire inference sequence (preprocess -> TRT inference ->
postprocess) as a CUDA graph. Replay it as a single CPU submission.

```cpp
// Record once
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
pre_process_kernel<<<...>>>(input, processed);
trt_ctx->enqueueV3(stream);
post_process_kernel<<<...>>>(output, result);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, 0);

// Replay every frame -- single CPU call
cudaGraphLaunch(graphExec, stream);
```

Or with Torch-TensorRT:
```python
compiled = torch_tensorrt.compile(model, ..., use_cuda_graph=True)
```

**Requirement**: Static input shapes, stable buffer addresses.

**Impact**: 10-30% latency reduction (bigger gains for small batch sizes
where CPU dispatch overhead dominates)  
**Effort**: Medium

---

## 6. Memory Optimization

### 6A. Pinned (Page-Locked) Memory

Allows DMA transfers from CPU to GPU without an extra copy through a
staging buffer.

```cpp
cudaHostRegister(image.data, bytes, cudaHostRegisterDefault);
cudaMemcpyAsync(gpu_buf, image.data, bytes, cudaMemcpyHostToDevice, stream);
```

### 6B. Zero-Copy (Integrated GPUs Only)

On Jetson/DGX Spark, CPU and GPU share physical memory. Skip the copy
entirely:

```cpp
cudaHostAlloc(&buf, bytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&gpu_ptr, buf, 0);
// GPU reads directly from buf -- no transfer
```

### 6C. Async Transfers

Overlap data transfer with computation:

```cpp
cudaMemcpyAsync(gpu_input, host_input, bytes, cudaMemcpyHostToDevice, stream);
// GPU starts preprocessing while the transfer completes
pre_process_kernel<<<..., stream>>>(...);
```

**Impact**: 5-15% combined  
**Effort**: Low

---

## 7. Batching & Pipelining

### 7A. Batch Processing

Process multiple images per engine invocation:

```bash
trtexec --onnx=model.onnx --saveEngine=engine.plan --fp16 \
  --minShapes=pixel_values:1x3x1008x1008 \
  --optShapes=pixel_values:4x3x1008x1008 \
  --maxShapes=pixel_values:8x3x1008x1008
```

Requires dynamic batch in the ONNX export (add `dynamic_axes` for batch dim).

**Impact**: Near-linear throughput scaling  
**Effort**: Medium (need dynamic shape export)

### 7B. Double-Buffer Pipeline

While GPU processes frame N, CPU decodes frame N+1:

```
CPU:  [decode 1] [decode 2] [decode 3] ...
GPU:          [infer 1] [infer 2] [infer 3] ...
```

**Impact**: Up to 2x throughput for I/O-bound pipelines  
**Effort**: Medium

---

# Part 2: Lossy Optimizations

These trade some accuracy for speed. Apply after exhausting all lossless
options. Ordered from least lossy to most lossy.

---

## 8. FP16 Precision

Convert from FP32 to FP16 (half precision). This is the standard deployment
baseline and almost universally used.

```bash
trtexec --onnx=model.onnx --saveEngine=engine_fp16.plan --fp16
```

**What TRT does**: Converts eligible layers to FP16. Some layers (softmax
normalization, loss computation) stay FP32 for numerical stability. TRT
decides this automatically.

### Known Issue: Monolithic ONNX Export Divergence

**Problem**: When SAM3 is exported as a single monolithic ONNX file and
compiled by TensorRT (via either `trtexec` or the Python API), the resulting
engine diverges significantly from PyTorch — even in FP32 mode:

| Build Method | Precision | Cosine Similarity |
|---|---|---|
| Monolithic ONNX + `trtexec` | FP16 | 0.868 |
| Monolithic ONNX + `trtexec` | FP32 | 0.868 |
| Monolithic ONNX + Python API | FP32 | 0.868 |
| ONNX Runtime (same ONNX) | FP32 | 0.999999 |

The ONNX model itself is correct (ORT matches PyTorch perfectly). The issue
is that TRT's graph optimizer reorganizes computation in the large fused
graph in ways that accumulate numerical error across 26,000+ layers.

**Root Cause**: TRT's fused multi-head attention kernel (`_gemm_mha_v2`)
causes Softmax overflow in FP16. But even in FP32, the monolithic graph's
complexity causes TRT to make suboptimal fusion/rewrite decisions that
diverge from the reference computation order.

### Solution: Split-Module Export + Mixed Precision

The proven fix (documented by the
[SAM3-TENSORRT-PYTHON](https://github.com/Kishan200308/SAM3-TENSORRT-PYTHON)
project and [this optimization blog post](https://egordmitriev.dev/blog/2026-05-17-optimizing-samurai-part-2))
is to:

1. **Split the model** into independent sub-modules (vision-encoder,
   text-encoder, decoder) that TRT can compile correctly in isolation.
2. **Force Softmax layers to FP32** via `OBEY_PRECISION_CONSTRAINTS` to
   prevent the fused MHA kernel from overflowing.

Results with the split-module approach:

| Configuration | Cosine Similarity | Max Abs Diff | Mean Abs Diff |
|---|---|---|---|
| Split FP16 + Softmax FP32 | **0.9958** | 51.2 | 0.76 |
| Split FP16 + Softmax + attn MatMul FP32 | 0.9955 | 87.2 | 0.72 |

The Softmax-only constraint is recommended (better cosine, smaller engines).

### Build Commands

```bash
# Step 1: Export split ONNX modules
python3 benchmark/scripts/export_split_onnx.py --all

# Step 2: Build engines with mixed precision
python3 benchmark/scripts/build_split_engines.py --all

# Step 3: Verify fidelity
python3 benchmark/scripts/verify_split_fidelity.py
```

### Why Not Just Use `--precisionConstraints` on the Monolithic ONNX?

We tested this approach (Phase 1 of our investigation). Even with Softmax
and attention MatMul layers forced to FP32 on the monolithic ONNX, TRT still
diverges (~0.87 cosine). The issue is not just FP16 overflow — it's that
TRT's graph rewrites on a 26,739-layer monolithic graph are fundamentally
different from the reference computation order. Splitting the model into
smaller sub-graphs limits TRT's rewrite scope to operations that are safe
to reorganize.

For the full ablation study and root cause analysis, see
[SAM3_TRT_INSIGHTS.md](SAM3_TRT_INSIGHTS.md).

**Impact**: ~2x throughput over FP32, half the memory  
**Accuracy loss**: Negligible (cosine > 0.995 vs FP32 PyTorch)  
**Effort**: Medium (split export + engine build scripts provided)

---

## 9. INT8 / FP8 Post-Training Quantization (PTQ)

Calibrate quantization scales using a representative dataset, then embed
quantize/dequantize (Q/DQ) nodes into the graph.

### Format Comparison

| Format | Bits | What's Quantized | Typical Accuracy Loss | GPU Requirement |
|--------|------|------------------|-----------------------|-----------------|
| FP16 | 16 | Weights + activations | ~0% | Any NVIDIA |
| INT8 (W8A8) | 8 | Weights + activations | 0.1-0.5% | Any TRT GPU |
| FP8 (W8A8) | 8 | Weights + activations | 0.05-0.3% | Hopper+ (H100) |
| INT8 weight-only | 8 | Weights only | <0.1% | Any TRT GPU |

### ONNX PTQ Path (NVIDIA Model Optimizer)

```bash
pip install nvidia-modelopt

# Prepare calibration data: 128-512 representative images
# as a numpy array of shape [N, 3, 1008, 1008]

python -m modelopt.onnx.quantization \
  --onnx_path=sam3_simplified.onnx \
  --quantize_mode=int8 \
  --calibration_data=calib.npy \
  --calibration_method=entropy \
  --output_path=sam3_int8.onnx

trtexec --onnx=sam3_int8.onnx --saveEngine=engine_int8.plan --int8 --fp16
```

### PyTorch PTQ Path

```python
import modelopt.torch.quantization as mtq

def calibrate(model):
    for images in calibration_loader:  # 128-512 samples
        model(images)

mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)
torch.onnx.export(model, ...)  # Q/DQ nodes are embedded
```

Calibration methods:
- `max`: Min/max of observed values. Fast, simple.
- `entropy`: Minimizes KL divergence. Better accuracy, slower.

**Impact**: 1.5-2x over FP16  
**Accuracy loss**: 0.1-0.5% (model-dependent)  
**Effort**: Medium (need representative calibration data)

---

## 10. INT4 / Mixed-Precision Quantization

More aggressive compression. Weights go to 4-bit; activations stay at
higher precision.

| Format | Description | Accuracy Loss |
|--------|-------------|---------------|
| INT4 AWQ (W4A16) | 4-bit weights, FP16 activations | 0.5-2% |
| W4A8 AWQ | 4-bit weights, INT8 activations | 1-3% |
| NVFP4 | NVIDIA's 4-bit FP with dynamic block quantization | 0.5-1.5% |

```bash
python -m modelopt.onnx.quantization \
  --onnx_path=sam3.onnx \
  --quantize_mode=int4 \
  --calibration_data=calib.npy \
  --calibration_method=awq_clip \
  --output_path=sam3_int4.onnx
```

NVIDIA Model Optimizer's `auto_quantize` can search per-layer for the best
precision mix:

```python
mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantize_cfg=[mtq.FP8_DEFAULT_CFG, mtq.INT4_AWQ_CFG],
    forward_loop=calibrate,
)
```

**Impact**: 2-3x over FP16 (mainly memory bandwidth)  
**Accuracy loss**: 0.5-3%  
**Effort**: Medium-High  
**GPU notes**: NVFP4 requires Blackwell (B200+). INT4 AWQ works on any GPU.

---

## 11. Quantization-Aware Training (QAT)

Fine-tune the model with fake quantization nodes inserted, so weights learn
to tolerate quantization noise. Used when PTQ degrades accuracy too much.

```python
mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=calibrate)

for epoch in range(5, 20):  # short fine-tune
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**Impact**: Recovers 1-3% accuracy over PTQ at same bit-width  
**Effort**: High (needs training data and compute)

---

## 12. Structured Pruning

Remove entire dimensions (attention heads, MLP hidden units, ViT blocks)
from the model.

State of the art:
- **CORP** (2025): One-shot, no retraining. 128-512 calibration images,
  ~20 min on one GPU. DeiT-Huge: 50% pruning, 83.3% accuracy, 1.85x speed.
- **SERo** (2025): 69% FLOPs reduction, 2.4x speedup, 1.55% accuracy drop.

Can combine with TRT sparsity exploitation:
```bash
trtexec --onnx=pruned_model.onnx --fp16 --sparsity=enable
```

**Impact**: 1.5-2.5x throughput  
**Accuracy loss**: 1-3%  
**Effort**: Medium (CORP) to High (methods requiring fine-tuning)

---

## 13. Knowledge Distillation

Train a smaller student model to mimic the SAM3 teacher.

The optimal compression pipeline ordering (per "Prune-Quantize-Distill",
2025) is: **Prune -> Quantize (QAT) -> Distill**. Distillation recovers
accuracy after the model is already in its constrained form.

**Impact**: Recovers 1-3% accuracy after pruning + quantization  
**Effort**: High (needs full training pipeline)

---

# Implementation Roadmap

### Phase 1: Baseline FP16 (Lossless + FP16)

```bash
# Install deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.0.0rc1 accelerate onnx onnx-simplifier
export PATH=/usr/src/tensorrt/bin:$PATH

# Free GPU memory
nvidia-smi  # kill large processes

# Export ONNX
cd ~/mehrdad/SAM3-TensorRT
python python/onnxexport.py

# Simplify graph
onnxsim onnx_weights/sam3_dynamic.onnx onnx_weights/sam3_simplified.onnx

# Build FP16 engine
trtexec --onnx=onnx_weights/sam3_simplified.onnx \
  --saveEngine=sam3_fp16.plan --fp16 --verbose

# Build and run C++ app
cd cpp && mkdir build && cd build && cmake .. && make
./sam3_pcs_app /path/to/images ../../sam3_fp16.plan
```

### Phase 2: Graph Optimization

- Try dynamo export (`dynamo=True` in onnxexport.py)
- Apply Polygraphy constant folding
- Apply ORT O2 transformer fusions
- Rebuild engine with `--builderOptimizationLevel=5`
- Add CUDA graph capture to C++ inference loop

### Phase 3: Encoder/Decoder Split

- Use `samexporter` or write custom split export
- Build separate TRT engines for image encoder, text encoder, decoder
- Modify C++ app to cache encoder output across prompts
- Profile to confirm compute distribution

### Phase 4: INT8 Quantization (First Lossy Step)

- Prepare calibration dataset (128-512 images)
- Run ONNX PTQ with entropy calibration
- Build INT8 engine, validate accuracy against FP16 baseline
- If accuracy drops too much, try mixed-precision per-layer

### Phase 5: Advanced (Pruning + QAT)

- Apply CORP structured pruning to ViT backbone
- Fine-tune with QAT to recover accuracy
- Export pruned+quantized model
- Build TRT engine with `--sparsity=enable`

---

# Tool Reference

| Tool | Purpose | Install |
|------|---------|---------|
| `trtexec` | Build TRT engines | `/usr/src/tensorrt/bin/trtexec` (pre-installed) |
| `onnxsim` | Simplify ONNX graphs | `pip install onnx-simplifier` |
| `polygraphy` | ONNX graph surgery | `pip install polygraphy` |
| `onnx_graphsurgeon` | Low-level ONNX manipulation | `pip install onnx_graphsurgeon` |
| `nvidia-modelopt` | PTQ/QAT quantization | `pip install nvidia-modelopt` |
| `samexporter` | SAM3 encoder/decoder split | `pip install samexporter` |
| `nsys` (Nsight Systems) | GPU timeline profiling | Pre-installed with CUDA |

---

# Further Reading

- [TensorRT: Working with Transformers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-transformers.html) -- MHA fusion requirements
- [TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html) -- Q/DQ workflow
- [NVIDIA Model Optimizer: ONNX PTQ](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/onnx_ptq/README.md) -- Quantization CLI
- [NVIDIA Model Optimizer: VLM PTQ](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/vlm_ptq/README.md) -- VLM recipes
- [Torch-TensorRT: CUDA Graphs](https://docs.pytorch.org/TensorRT/tutorials/runtime_opt/cuda_graphs.html)
- [Detect Anything in Real Time (arxiv 2603.11441)](https://arxiv.org/html/2603.11441v1) -- SAM3 split + TRT FP16
- [CORP Structured Pruning (arxiv 2602.05243)](https://arxiv.org/html/2602.05243v2) -- One-shot ViT pruning
- [From Checkpoint to ONNX Runtime (May 2026)](https://egordmitriev.dev/blog/2026-05-16-optimizing-samurai-part-1) -- SAM2 optimization walkthrough
- [samexporter](https://github.com/vietanhdev/samexporter) -- SAM3 split export tool
