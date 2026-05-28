# SAM3 TensorRT: Investigation Insights

Empirical findings from debugging the TRT fidelity divergence. This
companion document to [SAM3_TO_TENSORRT.md](SAM3_TO_TENSORRT.md) records
what was tried, what failed, and why — preserving institutional knowledge
for future TRT deployments of large VLMs.

---

## 1. Problem Statement

When SAM3 (a ~3 GB vision-language model with 26,739 ONNX layers) is exported
as a monolithic ONNX file and compiled by TensorRT, the resulting engine
produces outputs that diverge catastrophically from PyTorch — **even in pure
FP32 mode**.

The ONNX file itself is correct: ONNX Runtime reproduces PyTorch output with
near-perfect fidelity (cosine similarity = 0.999999). The divergence is
entirely a TRT compilation artifact.

---

## 2. Key Finding: Splitting Is the Fix

The complete ablation:

| Configuration | Splitting? | Precision Constraints? | Cosine Similarity | Max Abs Diff |
|---|---|---|---|---|
| Monolithic FP32 (`trtexec`) | No | N/A | 0.868 | 154 |
| Monolithic FP32 (Python API) | No | N/A | 0.868 | 154 |
| Monolithic FP16 + constraints | No | Softmax FP32 | 0.868 | 154 |
| ONNX Runtime FP32 (same ONNX) | No | N/A | 0.999999 | <0.001 |
| **Split FP32** | **Yes** | **N/A** | **0.999967** | **7.89** |
| **Split FP16 pure** | **Yes** | **No** | **0.994072** | **82.6** |
| **Split FP16 + Softmax FP32** | **Yes** | **Yes** | **0.995816** | **51.2** |

**Conclusion**: Splitting the model into sub-modules is the fix. It accounts
for the jump from 0.868 to 0.994+. Precision constraints provide marginal
additional improvement (0.994 → 0.996).

---

## 3. Why ONNX Runtime Works but TRT Doesn't

ONNX Runtime is an **interpreter**: it executes each ONNX node in graph order,
allocating intermediate tensors exactly as specified. The computation follows
the exact same sequence as PyTorch's eager execution.

TensorRT is an **optimizing compiler**: it rewrites, fuses, and reorders the
graph before generating GPU kernels. For a 26,739-layer graph, this means:

- **Layer fusion** — Adjacent ops merged into single kernels. Changes
  evaluation order, which in IEEE 754 arithmetic is non-associative:
  `(a+b)+c ≠ a+(b+c)` due to rounding.

- **Graph-level rewrites** — Entire subgraphs reorganized. The larger the
  graph, the more aggressive and "creative" TRT gets.

- **Kernel substitution** — TRT selects from hundreds of kernel
  implementations per layer (different tiling, Winograd, etc.), each
  mathematically equivalent in infinite precision but numerically different
  in finite precision.

- **Control flow restructuring** — SAM3's decoder has loop structures. TRT
  unrolls/restructures these differently when embedded in a larger graph
  context.

Over 26,739 layers, individually tiny numerical differences compound into
catastrophic divergence.

---

## 4. Why Splitting Fixes It

Splitting the model into 3 sub-modules (vision-encoder: 15,710 layers,
text-encoder: 3,040 layers, decoder: 9,677 layers) constrains TRT in
several ways:

| Property | Monolithic (26k layers) | Split (3 modules) |
|---|---|---|
| TRT rewrite scope | Global — can reorder across entire model | Local — confined to each module |
| Intermediate values | May be "optimized away" by fusion | Explicitly materialized as tensor outputs |
| Graph complexity | Unusual cross-connections confuse TRT | Each module is a standard architecture (ViT, CLIP, DETR) that TRT handles well |
| Error accumulation | Compounds across all 26k layers in one pass | Resets at module boundaries |

The key mechanism: **module boundaries force TRT to materialize intermediate
tensors**. TRT cannot fuse operations across module boundaries, so it cannot
make a "globally optimal" decision in the vision encoder that accidentally
corrupts the signal path for the decoder.

---

## 5. Precision Constraints: Nice-to-Have, Not Essential

Since the monolithic FP32 (no FP16 at all) already diverged at 0.868,
FP16 overflow was never the root cause. The precision constraints provide a
small quality improvement on the split modules:

| Split Config | Cosine | Max Diff | Mean Diff |
|---|---|---|---|
| FP16 pure (no constraints) | 0.994072 | 82.6 | 0.780 |
| FP16 + Softmax FP32 | 0.995816 | 51.2 | 0.757 |
| FP16 + Softmax + attn MatMul FP32 | 0.995530 | 87.2 | 0.718 |

The Softmax-only constraint reduces max outlier error (82 → 51) without
hurting overall cosine. Adding attention MatMul constraints doesn't help
further — it slightly worsens cosine while marginally improving mean error.

**Recommendation**: Use `--no-constraints` for maximum performance. Use the
default (Softmax FP32) if you need tighter worst-case error bounds.

---

## 6. The Softmax Overflow Bug (Secondary Issue)

TRT's fused multi-head attention kernel (`_gemm_mha_v2`) computes Q*K^T and
Softmax in a single kernel. In FP16, attention scores can exceed the
representable range (~65504) before Softmax normalizes them, producing
Inf/NaN outputs.

This is a well-documented issue:
- [SAM3-TENSORRT-PYTHON](https://github.com/Kishan200308/SAM3-TENSORRT-PYTHON)
  forces Softmax to FP32 in all sub-modules
- [Blog: Optimizing Samurai Part 2](https://egordmitriev.dev/blog/2026-05-17-optimizing-samurai-part-2)
  documents the exact bug and kernel name

However, this is **not** the root cause of our 0.868 divergence (since FP32
also failed). It's a separate, additive source of error that only manifests
in FP16 mode on the split modules.

---

## 7. PyTorch FP16/BF16 Comparison

To rule out inherent FP16 sensitivity in the model itself:

| PyTorch Mode | Cosine vs FP32 |
|---|---|
| FP16 (`torch.autocast`) | 0.999+ |
| BF16 (`torch.autocast`) | 0.998+ |

PyTorch's autocast keeps sensitive operations (LayerNorm, Softmax) in FP32
automatically, using FP32 accumulation for matmuls. The model itself is
FP16-safe when the runtime handles precision correctly.

---

## 8. What We Ruled Out

| Hypothesis | Test | Result |
|---|---|---|
| Preprocessing mismatch | Compared manual vs Sam3Processor | Identical (cosine=1.000) |
| ONNX export error | Compared ORT vs PyTorch | Perfect match (cosine=0.999999) |
| `trtexec` bug | Used TRT Python API instead | Same divergence |
| FP16 precision loss | Tested monolithic FP32 | Same divergence |
| Softmax overflow | Forced Softmax FP32 on monolithic | No improvement |
| Attention accumulation | Forced attn MatMul FP32 on monolithic | No improvement |

---

## 9. Open Questions

- **What specific TRT graph rewrite causes the monolithic divergence?**
  Polygraphy's layer-by-layer comparison (`--onnxrt-outputs mark all`)
  failed because TRT cannot mark outputs inside loop structures. A manual
  binary search (bisecting the graph) could pinpoint the first layer where
  divergence appears.

- **Would a different ONNX opset or exporter help?** We used legacy
  (`dynamo=False`) opset 17 for monolithic, opset 20 for split. The opset
  difference was not tested in isolation.

- **Is this specific to SAM3 or a general issue with large VLMs?** The
  SAM3-TENSORRT-PYTHON project and the optimization blog both hit the same
  issue independently, suggesting it's a pattern for models with >10k layers.

---

## 10. Reproduction

```bash
# Full pipeline from scratch
cd ~/mehrdad/sam3/benchmark/scripts

# Export split ONNX modules
python3 export_split_onnx.py --all

# Build all engine variants
python3 build_split_engines.py --all                # FP16 + Softmax FP32
python3 build_split_engines.py --all --no-constraints  # Pure FP16
python3 build_split_engines.py --all --fp32         # Pure FP32

# Verify fidelity
python3 verify_split_fidelity.py
```
