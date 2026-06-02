import argparse, csv, os, statistics, sys
import torch, torch.distributed as dist, torch.nn.functional as F
from torch import nn

CONFIGS = {
    "13b": dict(hidden=5120, n_heads=40, n_kv_heads=40, intermediate=13824, n_layers=40),
    "70b": dict(hidden=8192, n_heads=64, n_kv_heads=8, intermediate=28672, n_layers=80),
}
_COMM = []

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

def build_rope(seq, hd, device, dtype, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(seq, device=device).float(), inv)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

class Col(nn.Module):
    def __init__(self, i, o, div):
        super().__init__(); assert o % div == 0
        self.w = nn.Parameter(torch.empty(o // div, i)); nn.init.normal_(self.w, std=0.02)
    def forward(self, x): return F.linear(x, self.w)

class Row(nn.Module):
    def __init__(self, i, o, ws, mode, parts):
        super().__init__(); self.mode, self.parts = mode, parts
        if mode == "real":
            assert i % ws == 0; self.w = nn.Parameter(torch.empty(o, i // ws))
        else:
            assert i % parts == 0; self.w = nn.Parameter(torch.empty(o, i))
            self.st = [torch.cuda.Stream() for _ in range(parts)]
        nn.init.normal_(self.w, std=0.02)
    def forward(self, x):
        if self.mode == "real":
            y = F.linear(x, self.w)
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); dist.all_reduce(y, op=dist.ReduceOp.SUM); e.record()
            _COMM.append((s, e)); return y
        p = self.parts; ch = self.w.shape[1] // p
        xp, wp = x.split(ch, dim=-1), self.w.split(ch, dim=1)
        out = [None] * p; cur = torch.cuda.current_stream()
        for i in range(p):
            self.st[i].wait_stream(cur)
            with torch.cuda.stream(self.st[i]): out[i] = F.linear(xp[i], wp[i])
        for st in self.st: cur.wait_stream(st)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); y = out[0]
        for i in range(1, p): y = y + out[i]
        e.record(); _COMM.append((s, e)); return y

class RMS(nn.Module):
    def __init__(self, h, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(h)); self.eps = eps
    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(v + self.eps)).to(x.dtype) * self.w

class Layer(nn.Module):
    def __init__(self, c, ws, mode, parts):
        super().__init__()
        self.hd = c["hidden"] // c["n_heads"]; div = ws if mode == "real" else 1
        self.lh = c["n_heads"] // div; self.lkv = c["n_kv_heads"] // div; self.rep = self.lh // self.lkv
        self.ln1 = RMS(c["hidden"])
        self.q = Col(c["hidden"], c["n_heads"] * self.hd, div)
        self.k = Col(c["hidden"], c["n_kv_heads"] * self.hd, div)
        self.v = Col(c["hidden"], c["n_kv_heads"] * self.hd, div)
        self.o = Row(c["n_heads"] * self.hd, c["hidden"], ws, mode, parts)
        self.ln2 = RMS(c["hidden"])
        self.gate = Col(c["hidden"], c["intermediate"], div)
        self.up = Col(c["hidden"], c["intermediate"], div)
        self.down = Row(c["intermediate"], c["hidden"], ws, mode, parts)
    def forward(self, x, cos, sin):
        b, s, _ = x.shape; h = self.ln1(x)
        q = self.q(h).view(b, s, self.lh, self.hd).transpose(1, 2)
        k = self.k(h).view(b, s, self.lkv, self.hd).transpose(1, 2)
        v = self.v(h).view(b, s, self.lkv, self.hd).transpose(1, 2)
        q = q * cos[None, None] + rotate_half(q) * sin[None, None]
        k = k * cos[None, None] + rotate_half(k) * sin[None, None]
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1); v = v.repeat_interleave(self.rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(b, s, self.lh * self.hd)
        x = x + self.o(a); h = self.ln2(x)
        g = F.silu(self.gate(h)) * self.up(h)
        return x + self.down(g)

def measure(layer, b, s, c, dev, dt, mode, ws, iters, warm):
    cos, sin = build_rope(s, c["hidden"] // c["n_heads"], dev, dt)
    x = torch.randn(b, s, c["hidden"], device=dev, dtype=dt); tot, com = [], []
    for it in range(warm + iters):
        if mode == "real": dist.barrier()
        torch.cuda.synchronize(); _COMM.clear()
        a = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        a.record()
        with torch.no_grad(): layer(x, cos, sin)
        e.record(); torch.cuda.synchronize()
        if it < warm: continue
        t = a.elapsed_time(e); cm = sum(p.elapsed_time(q) for p, q in _COMM)
        if mode == "real":
            r = torch.tensor([t, cm], device=dev); dist.all_reduce(r, op=dist.ReduceOp.MAX)
            t, cm = r[0].item(), r[1].item()
        tot.append(t); com.append(cm)
    return statistics.median(tot), statistics.median(com)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "sim"], required=True)
    ap.add_argument("--models", default="13b,70b"); ap.add_argument("--batch", default="1,4,16,64")
    ap.add_argument("--seq", default="512,2048"); ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--sim-partitions", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dt = torch.bfloat16
    if a.mode == "real":
        if int(os.environ.get("WORLD_SIZE", "1")) < 2: sys.exit("use torchrun --nproc_per_node=2")
        dist.init_process_group("nccl"); rank = dist.get_rank(); ws = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank)); torch.cuda.set_device(local)
        dev = torch.device(f"cuda:{local}")
    else:
        rank, ws = 0, 1; dev = torch.device("cuda:0")
    rows = []
    for m in a.models.split(","):
        c = CONFIGS[m]
        if a.mode == "real": assert c["n_heads"] % ws == 0 and c["n_kv_heads"] % ws == 0
        layer = Layer(c, ws, a.mode, a.sim_partitions).to(dev, dt)
        par = ws if a.mode == "real" else a.sim_partitions
        for b in [int(x) for x in a.batch.split(",")]:
            for s in [int(x) for x in a.seq.split(",")]:
                lt, lc = measure(layer, b, s, c, dev, dt, a.mode, ws, a.iters, a.warmup)
                full = lt * c["n_layers"]
                rows.append(dict(mode=a.mode, model=m, parallel=par, batch=b, seq=s,
                    n_layers=c["n_layers"], layer_total_ms=round(lt, 4), layer_comm_ms=round(lc, 4),
                    layer_compute_ms=round(lt - lc, 4), comm_pct=round(100 * lc / lt, 2) if lt else 0,
                    full_total_ms=round(full, 3), full_tokens_per_s=round(b * s / (full / 1000), 1) if full else 0))
                if rank == 0:
                    print(f"{m:>4} b={b:<3} s={s:<5} layer={lt:7.3f}ms comm={lc:6.3f}ms "
                          f"({rows[-1]['comm_pct']:5.2f}%) full={full:8.1f}ms {rows[-1]['full_tokens_per_s']:>10.1f} tok/s", flush=True)
        del layer; torch.cuda.empty_cache()
    if rank == 0 and a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {a.out}", flush=True)
    if a.mode == "real": dist.destroy_process_group()

if __name__ == "__main__":
    main()
