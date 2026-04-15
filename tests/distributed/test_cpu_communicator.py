# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CpuCommunicator with and without torchcomms shim.

Uses gloo backend on CPU — no GPU required.

Run:
  pytest tests/distributed/test_cpu_communicator.py -v
"""

import os

import multiprocess as mp
import pytest
import torch

from tests.utils import ensure_current_vllm_config
from vllm.distributed.device_communicators.cpu_communicator import (
    CpuCommunicator,
)
from vllm.distributed.parallel_state import (
    get_world_group,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


def distributed_run(fn, world_size, use_torchcomms=False):
    processes: list[mp.Process] = []
    for i in range(world_size):
        env: dict[str, str] = {
            "RANK": str(i),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12399",
        }
        if use_torchcomms:
            env["VLLM_DISTRIBUTED_USE_TORCHCOMMS"] = "1"
        p = mp.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def cpu_worker_wrapper(fn):
    def wrapped_fn(env):
        update_environment_variables(env)
        with ensure_current_vllm_config():
            init_distributed_environment(backend="gloo")
            fn()

    return wrapped_fn


@cpu_worker_wrapper
def all_reduce_worker():
    group = get_world_group()
    comm = CpuCommunicator(
        cpu_group=group.cpu_group,
        device=torch.device("cpu"),
        device_group=group.device_group,
        unique_name="test",
    )
    tensor = torch.ones(64, dtype=torch.float32) * (comm.rank + 1)
    result = comm.all_reduce(tensor)
    expected = sum(range(1, comm.world_size + 1))
    assert torch.all(result == expected).item()


@cpu_worker_wrapper
def all_gather_worker():
    group = get_world_group()
    comm = CpuCommunicator(
        cpu_group=group.cpu_group,
        device=torch.device("cpu"),
        device_group=group.device_group,
        unique_name="test",
    )
    tensor = torch.ones(8, dtype=torch.float32) * (comm.rank + 1)
    result = comm.all_gather(tensor, dim=0)
    assert result.shape == (8 * comm.world_size,)
    for r in range(comm.world_size):
        chunk = result[r * 8 : (r + 1) * 8]
        assert torch.all(chunk == r + 1).item()


@cpu_worker_wrapper
def gather_worker():
    group = get_world_group()
    comm = CpuCommunicator(
        cpu_group=group.cpu_group,
        device=torch.device("cpu"),
        device_group=group.device_group,
        unique_name="test",
    )
    tensor = torch.ones(8, dtype=torch.float32) * (comm.rank + 1)
    result = comm.gather(tensor, dst=0, dim=0)
    if comm.rank_in_group == 0:
        assert result is not None
        assert result.shape == (8 * comm.world_size,)
        for r in range(comm.world_size):
            chunk = result[r * 8 : (r + 1) * 8]
            assert torch.all(chunk == r + 1).item()
    else:
        assert result is None


@pytest.mark.parametrize("use_torchcomms", [False, True],
                         ids=["standard", "torchcomms"])
def test_cpu_all_reduce(use_torchcomms):
    distributed_run(all_reduce_worker, 2, use_torchcomms=use_torchcomms)


@pytest.mark.parametrize("use_torchcomms", [False, True],
                         ids=["standard", "torchcomms"])
def test_cpu_all_gather(use_torchcomms):
    distributed_run(all_gather_worker, 2, use_torchcomms=use_torchcomms)


@pytest.mark.parametrize("use_torchcomms", [False, True],
                         ids=["standard", "torchcomms"])
def test_cpu_gather(use_torchcomms):
    distributed_run(gather_worker, 2, use_torchcomms=use_torchcomms)
