# tp-comm-overhead-h200
Measures real 2-way tensor-parallel communication overhead for LLaMA-2 13B/70B prefill on 2×H200 over NVLink, and contrasts it against a single-GPU multi-stream proxy. The proxy reports ~1% comm; real NVLink all-reduce is 8–36%, peaking at small batch. Includes NCCL bandwidth probe and per-layer comm/compute attribution via CUDA events.
