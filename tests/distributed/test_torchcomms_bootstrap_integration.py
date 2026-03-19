# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for TorchcommsBootstrap (hybrid PG + TorchComm).

Launch with:
    torchrun --nproc-per-node=2 \
        tests/distributed/test_torchcomms_bootstrap_integration.py

Requirements:
    - torchcomms nightly (0.1.0.dev+cu130 or later)
    - CUDA GPUs (at least 2)
    - torch nightly (2.12+)

This test calls torch.distributed.init_process_group() since the hybrid
TorchcommsBootstrap needs ProcessGroups for metadata communication.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Skip gracefully if torchcomms is not installed.
    try:
        import torchcomms  # noqa: F401
    except ImportError:
        print("SKIP: torchcomms not installed", file=sys.stderr)
        sys.exit(0)

    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", file=sys.stderr)
        sys.exit(0)

    gpu_count = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if gpu_count < world_size:
        print(
            f"SKIP: need {world_size} GPUs but only {gpu_count} available",
            file=sys.stderr,
        )
        sys.exit(0)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Initialize torch.distributed (required for hybrid bootstrap).
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    from vllm.distributed.bootstrap import TorchcommsBootstrap

    all_ranks = list(range(world_size))

    bootstrap = TorchcommsBootstrap()
    info = bootstrap.create_group(
        group_ranks=[all_ranks],
        global_rank=rank,
        backend="nccl",
    )

    # Verify basic info.
    assert info.rank == rank
    assert info.ranks == all_ranks
    assert info.world_size == world_size
    assert info.rank_in_group == rank

    # Hybrid: both ProcessGroups AND TorchComm objects should be present.
    assert info.cpu_group is not None, "cpu_group should be set"
    assert info.device_group is not None, "device_group should be set"
    assert info.device_comm is not None, "device_comm should be set"
    assert info.cpu_comm is not None, "cpu_comm should be set"

    # Verify all_reduce works on the TorchComm device comm.
    device = torch.device("cuda", local_rank)
    tensor = torch.ones(4, device=device) * (rank + 1)
    info.device_comm.all_reduce(
        tensor, torchcomms.ReduceOp.SUM, async_op=False
    )

    expected = sum(r + 1 for r in range(world_size))
    assert torch.allclose(
        tensor, torch.full((4,), expected, device=device, dtype=tensor.dtype)
    ), f"Rank {rank}: expected all {expected}, got {tensor}"

    # Verify all_reduce works on the TorchComm CPU comm.
    cpu_tensor = torch.ones(4) * (rank + 1)
    info.cpu_comm.all_reduce(
        cpu_tensor, torchcomms.ReduceOp.SUM, async_op=False
    )
    assert torch.allclose(
        cpu_tensor, torch.full((4,), expected, dtype=cpu_tensor.dtype)
    ), f"Rank {rank}: CPU expected all {expected}, got {cpu_tensor}"

    # Verify ProcessGroup all_reduce also works (sanity check).
    pg_tensor = torch.ones(4, device=device) * (rank + 1)
    torch.distributed.all_reduce(pg_tensor, group=info.device_group)
    assert torch.allclose(
        pg_tensor,
        torch.full((4,), expected, device=device, dtype=pg_tensor.dtype),
    ), f"Rank {rank}: PG all_reduce expected {expected}, got {pg_tensor}"

    # Test TorchCommDeviceCommunicator wrapping.
    from unittest.mock import patch

    from vllm.distributed.device_communicators.torchcomm_communicator import (
        TorchCommDeviceCommunicator,
    )

    with patch(
        "vllm.config.get_current_vllm_config_or_none", return_value=None
    ):
        comm = TorchCommDeviceCommunicator(
            cpu_group=info.cpu_group,
            device=device,
            device_group=info.device_group,
            device_comm=info.device_comm,
            unique_name="test:0",
        )

    # Test all_reduce through the communicator.
    ar_tensor = torch.ones(4, device=device) * (rank + 1)
    result = comm.all_reduce(ar_tensor)
    assert torch.allclose(
        result,
        torch.full((4,), expected, device=device, dtype=result.dtype),
    ), f"Rank {rank}: comm.all_reduce expected {expected}, got {result}"

    # Test broadcast through the communicator.
    bc_tensor = torch.ones(4, device=device) * (rank + 1)
    result = comm.broadcast(bc_tensor, src=0)
    assert torch.allclose(
        result, torch.full((4,), 1.0, device=device, dtype=result.dtype)
    ), f"Rank {rank}: comm.broadcast expected 1.0, got {result}"

    torch.distributed.destroy_process_group()
    print(f"Rank {rank}: PASSED")


if __name__ == "__main__":
    main()
