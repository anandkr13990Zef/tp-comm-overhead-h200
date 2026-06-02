import csv, datetime, subprocess
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"<unavailable: {e}>"
def tinfo():
    import torch
    g = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return (f"torch {torch.__version__} | cuda {torch.version.cuda} | "
            f"nccl {'.'.join(map(str, torch.cuda.nccl.version()))} | gpus {g}")
def rows(p):
    try:
        with open(p) as f: return list(csv.DictReader(f))
    except FileNotFoundError: return []
real, sim = rows("real.csv"), rows("sim.csv")
k = lambda r: (r["model"], r["batch"], r["seq"])
sm = {k(r): r for r in sim}; m = []
for r in real:
    s = sm.get(k(r))
    if not s: continue
    cr, cs = float(r["comm_pct"]), float(s["comm_pct"])
    m.append((r["model"], int(r["batch"]), int(r["seq"]), cr, cs, round(cr - cs, 2),
              float(r["full_tokens_per_s"])))
m.sort(key=lambda x: (x[0], x[1], x[2]))
try: bw = open("bw.txt").read().strip()
except FileNotFoundError: bw = "<no bw.txt>"
sep = "=" * 78
L = [sep, "TENSOR-PARALLEL OVERHEAD: real 2x H200 NVLink vs single-GPU stream proxy",
     f"generated {datetime.datetime.now().isoformat(timespec='seconds')}", sep, "",
     "[SETUP]", tinfo(), "",
     sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"), "",
     "[TOPOLOGY] NV18 = 18 bonded NVLinks (~900 GB/s bidir)", sh("nvidia-smi topo -m"), "",
     "[NVLINK ALL-REDUCE BANDWIDTH]", bw, "",
     "[RESULTS] comm% = per-layer all-reduce / layer total. gap_pp = real - sim.", "",
     f"{'model':>5} {'batch':>5} {'seq':>5} | {'comm%_real':>10} {'comm%_sim':>9} {'gap_pp':>7} | {'tok/s_real':>11}",
     "-" * 70]
for r in m:
    L.append(f"{r[0]:>5} {r[1]:>5} {r[2]:>5} | {r[3]:>10.2f} {r[4]:>9.2f} {r[5]:>7.2f} | {r[6]:>11.1f}")
L += ["", "[NOTE] single decoder layer at exact LLaMA-2 dims x n_layers; random weights",
      "(latency is shape/dtype-determined, exact for timing). gap_pp peaks at the",
      "comm-bound corner (small batch/seq) - the regime the proxy most misrepresents.", sep]
t = "\n".join(L) + "\n"
open("report.txt", "w").write(t); print(t); print("wrote -> report.txt")
