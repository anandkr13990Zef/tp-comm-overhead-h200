# Tensor-Parallel Communication Overhead: Real 2×H200 vs Single-GPU Stream Proxy

## Objective

Measure the communication overhead of 2-way tensor-parallel (TP) prefill for
LLaMA-2 13B and 70B on a 2×H200 node, and compare it against the single-GPU,
multi-stream proxy used in the earlier capstone, in order to quantify how far
the proxy departs from real-hardware behaviour.

## Setup

| Item | Value |
|---|---|
| GPUs | 2 × NVIDIA H200, 143771 MiB each (~141 GB usable) |
| Interconnect | NV18 (18 bonded NVLinks), nominal ~450 GB/s per direction |
| Inter-GPU topology | Direct NVLink (PCIe/NIC paths not used by the 2-GPU TP run) |
| PyTorch | 2.11.0+cu130 |
| NCCL | 2.28.9 |
| Precision | bfloat16 |
| Launcher | `torchrun --nproc_per_node=2` (real), single process (sim) |

## Method

A single LLaMA-2 decoder layer is built at the **exact** model dimensions and
timed during a forward (prefill) pass. Per-layer times are scaled by the model's
layer count for a full-model estimate.

| Model | hidden | n_heads | n_kv_heads | intermediate | n_layers |
|---|---|---|---|---|---|
| 13B | 5120 | 40 | 40 | 13824 | 40 |
| 70B | 8192 | 64 | 8 | 28672 | 80 |

Each layer performs two all-reduces under TP: after `o_proj` (attention output)
and after `down_proj` (MLP output). These are timed with CUDA events on the
default stream; compute time is `total − comm`. In the real run, per-iteration
`total` and `comm` are reduced with `MAX` across ranks so the slower rank gates
the reported number. Each measurement is the median of 20 timed iterations after
5 warm-up iterations.

**Real mode**: weights sharded across the 2 GPUs; the two all-reduces are NCCL
collectives over NVLink.

**Sim mode (proxy)**: full layer on one GPU; the contraction dimension of each
row-parallel projection is split across 2 CUDA streams, and the "communication"
is the local element-wise sum of the partial outputs in HBM. No data crosses any
interconnect and there is no cross-device barrier.

### What is synthetic, and why the timings are still valid

Weights are random-initialised. This does not affect any timing: collective cost
is a function of `(batch, seq, hidden, dtype)` only, and dense GEMM / FlashAttention
runtime is set by tensor shapes, not values (no sparsity, no value-dependent
branching). Because the layers are built at exact LLaMA-2 dimensions, the tensors
crossing NVLink are byte-identical in size and dtype to the real models, so the
all-reduce being timed is the same operation the real model performs at that shape.
Loading real weights would enable correctness/perplexity checks but would not
change any latency reported here.

The structural approximation is the single-layer × `n_layers` scaling. It assumes
all decoder layers are identical (true for LLaMA) and omits the embedding lookup
(no comm) and the vocab-parallel `lm_head` collective (one all-gather/all-reduce
per forward). The omitted output-layer collective is amortised over many layers in
prefill but would be more significant in decode.

## NVLink all-reduce bandwidth

Measured with `all_reduce` on bf16 buffers, median of 20 iterations. For a 2-GPU
all-reduce, busbw equals algbw.

| Size | Latency | algbw = busbw |
|---|---|---|
| 1 MB | 0.040 ms | 26.3 GB/s |
| 4 MB | 0.053 ms | 79.8 GB/s |
| 16 MB | 0.093 ms | 181.2 GB/s |
| 64 MB | 0.263 ms | 255.1 GB/s |
| 256 MB | 0.906 ms | 296.3 GB/s |
| 1024 MB | 3.283 ms | 327.0 GB/s |

Bandwidth rises with message size and plateaus near 327 GB/s at 1 GB. Small
messages are latency-bound (a 1 MB transfer reaches only ~26 GB/s), which is the
regime the per-layer all-reduces fall into at small batch and sequence length.

## Results

### Real, 2-way TP over NVLink

| model | batch | seq | layer_total (ms) | comm (ms) | compute (ms) | comm % | full (ms) | tok/s |
|---|---|---|---|---|---|---|---|---|
| 13b | 1 | 512 | 1.0757 | 0.3841 | 0.6916 | 35.71 | 43.0 | 11899 |
| 13b | 1 | 2048 | 1.6883 | 0.1910 | 1.4973 | 11.31 | 67.5 | 30326 |
| 13b | 4 | 512 | 1.6596 | 0.1910 | 1.4686 | 11.51 | 66.4 | 30851 |
| 13b | 4 | 2048 | 6.2099 | 0.6212 | 5.5887 | 10.00 | 248.4 | 32980 |
| 13b | 16 | 512 | 6.1274 | 0.6170 | 5.5104 | 10.07 | 245.1 | 33423 |
| 13b | 16 | 2048 | 24.5236 | 2.4670 | 22.0566 | 10.06 | 980.9 | 33405 |
| 13b | 64 | 512 | 24.6854 | 2.6561 | 22.0292 | 10.76 | 987.4 | 33186 |
| 13b | 64 | 2048 | 101.3439 | 10.8838 | 90.4601 | 10.74 | 4053.8 | 32334 |
| 70b | 1 | 512 | 1.3128 | 0.1930 | 1.1198 | 14.70 | 105.0 | 4875 |
| 70b | 1 | 2048 | 3.4552 | 0.2842 | 3.1711 | 8.22 | 276.4 | 7409 |
| 70b | 4 | 512 | 3.5105 | 0.3321 | 3.1784 | 9.46 | 280.8 | 7292 |
| 70b | 4 | 2048 | 14.2146 | 1.4099 | 12.8047 | 9.92 | 1137.2 | 7204 |
| 70b | 16 | 512 | 13.8980 | 1.1831 | 12.7149 | 8.51 | 1111.8 | 7368 |
| 70b | 16 | 2048 | 58.7521 | 6.6015 | 52.1507 | 11.24 | 4700.2 | 6972 |
| 70b | 64 | 512 | 57.7104 | 5.6050 | 52.1054 | 9.71 | 4616.8 | 7098 |
| 70b | 64 | 2048 | 230.5551 | 19.2991 | 211.2559 | 8.37 | 18444.4 | 7106 |

### Simulated stream proxy, single GPU

| model | batch | seq | layer_total (ms) | comm (ms) | comm % | tok/s |
|---|---|---|---|---|---|---|
| 13b | 1 | 512 | 0.952 | 0.027 | 2.87 | 13444 |
| 13b | 1 | 2048 | 2.591 | 0.038 | 1.49 | 19764 |
| 13b | 4 | 512 | 2.552 | 0.039 | 1.51 | 20059 |
| 13b | 4 | 2048 | 10.436 | 0.122 | 1.17 | 19624 |
| 13b | 16 | 512 | 10.267 | 0.121 | 1.18 | 19947 |
| 13b | 16 | 2048 | 42.261 | 0.462 | 1.09 | 19385 |
| 13b | 64 | 512 | 41.513 | 0.461 | 1.11 | 19733 |
| 13b | 64 | 2048 | 171.612 | 1.838 | 1.07 | 19094 |
| 70b | 1 | 512 | 1.621 | 0.020 | 1.25 | 3948 |
| 70b | 1 | 2048 | 6.268 | 0.055 | 0.88 | 4085 |
| 70b | 4 | 512 | 6.138 | 0.055 | 0.90 | 4171 |
| 70b | 4 | 2048 | 25.160 | 0.188 | 0.75 | 4070 |
| 70b | 16 | 512 | 25.230 | 0.189 | 0.75 | 4059 |
| 70b | 16 | 2048 | 102.476 | 0.747 | 0.73 | 3997 |
| 70b | 64 | 512 | 101.085 | 0.747 | 0.74 | 4052 |
| 70b | 64 | 2048 | 406.111 | 2.994 | 0.74 | 4034 |

Note: sim per-layer compute is roughly 2× the real per-rank compute because one
GPU runs the full, unsharded layer. Sim throughput is therefore lower; this is
expected and not the comparison of interest. The comparison is the comm fraction.

### Comm fraction: real vs proxy

`gap_pp` = real comm % − proxy comm %.

| model | batch | seq | comm % real | comm % proxy | gap (pp) |
|---|---|---|---|---|---|
| 13b | 1 | 512 | 35.71 | 2.87 | 32.84 |
| 13b | 1 | 2048 | 11.31 | 1.49 | 9.82 |
| 13b | 4 | 512 | 11.51 | 1.51 | 10.00 |
| 13b | 4 | 2048 | 10.00 | 1.17 | 8.83 |
| 13b | 16 | 512 | 10.07 | 1.18 | 8.89 |
| 13b | 16 | 2048 | 10.06 | 1.09 | 8.97 |
| 13b | 64 | 512 | 10.76 | 1.11 | 9.65 |
| 13b | 64 | 2048 | 10.74 | 1.07 | 9.67 |
| 70b | 1 | 512 | 14.70 | 1.25 | 13.45 |
| 70b | 1 | 2048 | 8.22 | 0.88 | 7.34 |
| 70b | 4 | 512 | 9.46 | 0.90 | 8.56 |
| 70b | 4 | 2048 | 9.92 | 0.75 | 9.17 |
| 70b | 16 | 512 | 8.51 | 0.75 | 7.76 |
| 70b | 16 | 2048 | 11.24 | 0.73 | 10.51 |
| 70b | 64 | 512 | 9.71 | 0.74 | 8.97 |
| 70b | 64 | 2048 | 8.37 | 0.74 | 7.63 |

## Findings

**1. The proxy and the real run measure different operations.** The real comm is
an NCCL all-reduce over NVLink, which involves data transfer, a cross-device
synchronization barrier, and collective launch overhead. The proxy's "comm" is a
local element-wise add in HBM with none of those components. The proxy is not a
scaled-down version of the all-reduce; it lacks the two terms (barrier and launch
latency) that dominate the real cost at small message sizes.

**2. The proxy reports comm of 0.7–2.9%; the real fabric is 8–36%.** In the
compute-heavy region this is roughly 10% vs 1% (about a 10× difference). At the
smallest workload measured (13B, batch 1, seq 512) it is 35.71% vs 2.87%. The
discrepancy is largest in the regime where communication matters most.

**3. Small all-reduces are latency-bound, not bandwidth-bound.** For 13B at
batch 1, the activation is ~5 MB at seq 512 and ~20 MB at seq 2048, yet the
measured comm time is higher at seq 512 (0.384 ms) than at seq 2048 (0.191 ms).
At these sub-millisecond scales the all-reduce time is governed by fixed launch
and barrier cost plus run-to-run jitter rather than payload size. The bandwidth
table corroborates this: small transfers achieve only a fraction of peak
bandwidth. The single 35.71% figure should be read as the latency-bound corner
and is inherently noisier than the larger-workload points.

**4. The comm fraction stabilises near 10% (13B) once past the latency floor.**
For batch ≥ 4, both compute (GEMM FLOPs ∝ tokens) and comm (all-reduce bytes ∝
tokens × hidden) scale with token count, so their ratio converges to a roughly
constant value set by arithmetic intensity and bandwidth. This ~10% is the
structural TP tax at production batch sizes for this model on this hardware.

**5. 70B shows a somewhat lower comm fraction than 13B.** Comm bytes scale
linearly with hidden (8192/5120 ≈ 1.6×), while per-layer compute scales with
hidden × (hidden + intermediate), which grows faster (MLP work ratio ≈ 3.3×).
Wider layers amortise the all-reduce over more FLOPs, so the comm fraction is
lower. The observed drop (≈10% → ≈8–9%) is modest and partly obscured by the
fixed per-all-reduce cost still present at these sizes; the direction is
consistent with the mechanism but the magnitude should not be over-read from this
data alone.

**6. The proxy cannot be calibrated to recover the real number.** Because it has
no barrier term and no launch-latency term, no single multiplier maps its output
onto the real curve: the required correction at the latency-bound corner (~12×)
differs from the compute-bound region (~10×) and would differ again in decode.

## Practical implication

Using the single-GPU stream proxy to estimate TP communication cost would
under-report it by roughly 10× in the typical case and ~12× at small batch. A
roofline estimate is more appropriate: per-layer comm ≈ `2 × max(L, bytes / B)`,
where `L` is the measured launch+barrier latency floor, `bytes = batch × seq ×
hidden × 2` (two all-reduces, bf16), and `B` is the busbw at that message size
from the bandwidth table. The proxy has neither the `L` term nor the correct `B`.

## Limitations

- **Prefill only.** Decode (seq 1 with KV cache) has memory-bound GEMVs while the
  two all-reduces keep their latency floor, so the comm fraction in decode would
  be higher than any value here. These results do not characterise decode.
- **Exposed communication.** No compute/comm overlap is applied. Production
  inference stacks overlap collectives with compute, which would reduce the
  effective comm fraction below the real values reported here. The real column is
  therefore an upper bound on the un-overlapped tax.
- **Single layer × n_layers.** Omits the vocab-parallel `lm_head` collective
  (negligible in prefill, more relevant in decode) and any cross-layer scheduling.
- **Intra-node NVLink only.** Cross-node TP over the NIC path would incur
  substantially higher comm cost.
- **Random weights.** Valid for timing (shape/dtype-determined) but not used to
  verify numerical correctness or perplexity.
- Per-point figures are medians of 20 iterations; sub-millisecond points carry
  proportionally more measurement noise.

## Reproduction

```bash
# NVLink all-reduce bandwidth
torchrun --nproc_per_node=2 nccl_bw.py | tee bw.txt

# Real 2-way TP overhead
torchrun --nproc_per_node=2 tp_overhead.py --mode real \
    --models 13b,70b --batch 1,4,16,64 --seq 512,2048 --out real.csv

# Simulated single-GPU stream proxy
python tp_overhead.py --mode sim --sim-partitions 2 \
    --models 13b,70b --batch 1,4,16,64 --seq 512,2048 --out sim.csv
```

Environment: 2×H200, PyTorch 2.11.0+cu130, NCCL 2.28.9, bfloat16.
