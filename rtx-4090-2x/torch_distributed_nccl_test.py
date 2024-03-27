import os
import time
import torch
import torch.distributed


if __name__ == '__main__':
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    N = 50
    if local_rank == 0:
        a = torch.randn(1024, 1024, 1024, device=device, dtype=torch.float32)
        torch.distributed.send(a, dst=1)    # warmup / sync
        torch.cuda.synchronize(device)
        start  = time.monotonic()
        for i in range(N):
            torch.distributed.send(a, dst=1)
        torch.cuda.synchronize(device)
        end = time.monotonic()
        elapsed = end - start
        storage_size = a.untyped_storage().size()
        print(f"rank={local_rank} send N={N}, elapsed_time={elapsed:.4f}s, {storage_size*N/elapsed/2**30:.3f} GB/s, sum={a.sum()} ({a.device})")
    else:
        a = torch.empty(1024, 1024, 1024, device=device, dtype=torch.float32)
        torch.distributed.recv(a, src=0)    # warmup / sync
        torch.cuda.synchronize(device)
        start  = time.monotonic()
        for i in range(N):
            torch.distributed.recv(a, src=0)
        torch.cuda.synchronize(device)
        end  = time.monotonic()
        elapsed = end - start
        storage_size = a.untyped_storage().size()
        print(f"rank={local_rank} recv N={N}, elapsed_time={elapsed:.4f}s, {storage_size*N/elapsed/2**30:.3f} GB/s, sum={a.sum()} ({a.device})")
        

    torch.cuda.synchronize(device)
    start  = time.monotonic()
    for i in range(N):
        torch.distributed.broadcast(a, src=0)
    torch.cuda.synchronize(device)
    end  = time.monotonic()
    elapsed = end - start
    print(f"rank={local_rank} broadcast(a, src=0) N={N}, elapsed_time={elapsed:.4f}s, {storage_size*N/elapsed/2**30:.3f} GB/s, sum={a.sum()} ({a.device})")

    torch.cuda.synchronize(device)
    start  = time.monotonic()
    for i in range(N):
        torch.distributed.broadcast(a, src=1)
    torch.cuda.synchronize(device)
    end  = time.monotonic()
    elapsed = end - start
    print(f"rank={local_rank} broadcast(a, src=1) N={N}, elapsed_time={elapsed:.4f}s, {storage_size*N/elapsed/2**30:.3f} GB/s, sum={a.sum()} ({a.device})")
