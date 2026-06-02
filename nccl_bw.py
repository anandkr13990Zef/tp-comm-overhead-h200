import os, statistics, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", rank)); torch.cuda.set_device(local)
dev = torch.device(f"cuda:{local}")
if rank == 0:
    print(f"# world_size={ws} dtype=bf16 nccl={torch.cuda.nccl.version()}")
for mb in [1, 4, 16, 64, 256, 1024]:
    n = mb * 1024 * 1024 // 2
    x = torch.ones(n, dtype=torch.bfloat16, device=dev)
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); ts = []
    for _ in range(20):
        dist.barrier(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); dist.all_reduce(x); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1000.0)
    t = statistics.median(ts); algbw = n * 2 / t / 1e9; busbw = algbw * 2 * (ws - 1) / ws
    if rank == 0:
        print(f"  {mb:>6} MB  {t*1e3:8.3f}ms  algbw {algbw:7.1f} GB/s  busbw {busbw:7.1f} GB/s", flush=True)
dist.destroy_process_group()
