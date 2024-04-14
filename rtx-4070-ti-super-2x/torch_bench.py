import torch
import torch.utils.benchmark as benchmark

device0 = torch.device("cuda", 0)
device1 = torch.device("cuda", 1)

x0 = torch.randn(1024, 1024, 512, dtype=torch.float32, device=device0)
x1 = torch.randint(0,100, (1024, 1024, 512), dtype=torch.long, device=device1)

def copy_tensor(x, dest_device):
    y = x.to(dest_device, non_blocking=False, copy=False)
    return y

t0 = benchmark.Timer(
    stmt='copy_tensor(x0, device1)',
    setup='from __main__ import copy_tensor',
    globals={'x0': x0, 'device1': device1},
    num_threads=1)

t1 = benchmark.Timer(
    stmt='copy_tensor(x1, device1)',
    setup='from __main__ import copy_tensor',
    globals={'x1': x1, 'device1': device0},
    num_threads=1)

# sanity check
s0 = x0.sum().cpu()
y1 = copy_tensor(x0, device1)
s1 = y1.sum().cpu()
assert torch.abs(s0-s1) < 1e-5

m0 = t0.timeit(100)
storage_size0 = x0.untyped_storage().size()
print(f"{device0}->{device1}: {storage_size0/m0.mean/2**30:.3f} GB/s")

m1 = t1.timeit(100)
storage_size1 = x1.untyped_storage().size()
print(f"{device1}->{device0}: {storage_size1/m1.mean/2**30:.3f} GB/s")
