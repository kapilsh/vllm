# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for BootstrapProvider with real NCCL groups.

Launch with torchrun:
    torchrun --nproc-per-node=2 \
        tests/distributed/test_bootstrap_integration.py
"""

import os
import sys

import torch
import torch.distributed as dist

from vllm.distributed.bootstrap import (
    BootstrapInfo,
    ProcessGroupBootstrap,
)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"[rank {rank}] world_size={world_size}, local_rank={local_rank}")

    # --- Test 1: Single group with all ranks ---
    bootstrap = ProcessGroupBootstrap()
    all_ranks = list(range(world_size))
    info = bootstrap.create_group(
        group_ranks=[all_ranks],
        global_rank=rank,
        backend="nccl",
    )

    assert info.rank == rank
    assert info.ranks == all_ranks
    assert info.world_size == world_size
    assert info.rank_in_group == rank
    assert info.cpu_group is not None
    assert info.device_group is not None

    # Verify the NCCL group actually works: all_reduce
    t = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(t, group=info.device_group)
    expected = sum(r + 1.0 for r in range(world_size))
    assert t.item() == expected, f"all_reduce failed: {t.item()} != {expected}"

    # Verify the gloo CPU group works
    t_cpu = torch.tensor([rank * 10.0])
    dist.all_reduce(t_cpu, group=info.cpu_group)
    expected_cpu = sum(r * 10.0 for r in range(world_size))
    assert t_cpu.item() == expected_cpu, (
        f"cpu all_reduce failed: {t_cpu.item()} != {expected_cpu}"
    )

    print(f"[rank {rank}] Test 1 PASSED: single group all_reduce")

    # --- Test 2: Multiple sub-groups (pairs) ---
    if world_size >= 2:
        # Split into pairs: [0,1], [2,3], ...
        group_ranks = []
        for i in range(0, world_size, 2):
            if i + 1 < world_size:
                group_ranks.append([i, i + 1])
            else:
                group_ranks.append([i])

        bootstrap2 = ProcessGroupBootstrap()
        info2 = bootstrap2.create_group(
            group_ranks=group_ranks,
            global_rank=rank,
            backend="nccl",
        )

        assert rank in info2.ranks
        assert info2.world_size == len(info2.ranks)
        assert info2.rank_in_group == info2.ranks.index(rank)

        # Verify sub-group all_reduce
        t2 = torch.tensor([1.0], device="cuda")
        dist.all_reduce(t2, group=info2.device_group)
        assert t2.item() == float(info2.world_size), (
            f"sub-group all_reduce: {t2.item()} != {info2.world_size}"
        )

        print(f"[rank {rank}] Test 2 PASSED: sub-group all_reduce")

    dist.barrier()
    if rank == 0:
        print("ALL TESTS PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
