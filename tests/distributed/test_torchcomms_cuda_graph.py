# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph capture + replay tests for TorchComm communicators.

Validates that torchcomms collectives (all_reduce, all_gather, broadcast,
reduce_scatter, send/recv) can be captured inside a torch.cuda.CUDAGraph
and replayed with correct results.

Run with torchrun (requires 2+ GPUs):
    torchrun --nproc-per-node=2 tests/distributed/test_torchcomms_cuda_graph.py
    torchrun --nproc-per-node=4 tests/distributed/test_torchcomms_cuda_graph.py
"""

from __future__ import annotations

import os
import sys

import torch
import torchcomms


def _setup() -> tuple[int, int, torch.device, torchcomms.TorchComm]:
    """Create a TorchComm from env vars (no torch.distributed needed)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    store = torch.distributed.TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]),
        world_size=world_size,
        is_master=(rank == 0),
    )

    # Create world comm, then split (mirrors TorchcommsBootstrap)
    device_store = torch.distributed.PrefixStore("test/device", store)
    world_comm = torchcomms.new_comm(
        "nccl", device, name="world_device", store=device_store,
    )
    comm = world_comm.split(
        list(range(world_size)), name="tp_split",
    )
    return rank, world_size, device, comm


def _capture_graph(
    device: torch.device,
    fn,
) -> torch.cuda.CUDAGraph:
    """Warmup then capture fn() into a CUDAGraph."""
    # Warmup
    fn()

    stream = torch.cuda.Stream(device=device)
    torch.cuda.current_stream(device).synchronize()
    stream.wait_stream(torch.cuda.current_stream(device))

    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            fn()
    stream.synchronize()
    return g


def test_all_reduce_eager(rank, world_size, device, comm):
    x = torch.ones(8, device=device) * (rank + 1)
    comm.all_reduce(x, torchcomms.ReduceOp.SUM, async_op=False)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(x, torch.full_like(x, expected)), \
        f"Eager all_reduce: got {x[0].item()}, expected {expected}"


def test_all_reduce_graph(rank, world_size, device, comm):
    expected = sum(range(1, world_size + 1))
    y = torch.ones(8, device=device) * (rank + 1)

    g = _capture_graph(device, lambda: comm.all_reduce(
        y, torchcomms.ReduceOp.SUM, async_op=False))

    for i in range(5):
        y.fill_(rank + 1)
        g.replay()
        torch.cuda.synchronize(device)
        assert torch.allclose(y, torch.full_like(y, expected)), \
            f"Graph replay {i}: got {y[0].item()}, expected {expected}"


def test_all_gather_eager(rank, world_size, device, comm):
    inp = torch.ones(4, device=device) * (rank + 1)
    out = torch.empty(4 * world_size, device=device)
    comm.all_gather_single(out, inp, async_op=False)
    for r in range(world_size):
        chunk = out[r * 4 : (r + 1) * 4]
        assert torch.allclose(chunk, torch.full_like(chunk, r + 1)), \
            f"Eager all_gather chunk {r}: {chunk[0].item()} != {r + 1}"


def test_all_gather_graph(rank, world_size, device, comm):
    inp = torch.ones(4, device=device) * (rank + 1)
    out = torch.empty(4 * world_size, device=device)

    g = _capture_graph(device, lambda: comm.all_gather_single(
        out, inp, async_op=False))

    for i in range(3):
        inp.fill_(rank + 1)
        out.zero_()
        g.replay()
        torch.cuda.synchronize(device)
        for r in range(world_size):
            chunk = out[r * 4 : (r + 1) * 4]
            assert torch.allclose(chunk, torch.full_like(chunk, r + 1)), \
                f"Graph replay {i}, chunk {r}: {chunk[0].item()} != {r + 1}"


def test_broadcast_eager(rank, world_size, device, comm):
    x = torch.ones(8, device=device) * (42 if rank == 0 else 0)
    comm.broadcast(x, 0, async_op=False)
    assert torch.allclose(x, torch.full_like(x, 42)), \
        f"Eager broadcast: got {x[0].item()}, expected 42"


def test_broadcast_graph(rank, world_size, device, comm):
    x = torch.ones(8, device=device) * (42 if rank == 0 else 0)

    g = _capture_graph(device, lambda: comm.broadcast(
        x, 0, async_op=False))

    for i in range(3):
        x.fill_(42 if rank == 0 else 0)
        g.replay()
        torch.cuda.synchronize(device)
        assert torch.allclose(x, torch.full_like(x, 42)), \
            f"Graph replay {i}: got {x[0].item()}, expected 42"


def test_reduce_scatter_eager(rank, world_size, device, comm):
    # Each rank contributes a full tensor, gets back 1/world_size chunk
    inp = torch.ones(8 * world_size, device=device) * (rank + 1)
    out = torch.empty(8, device=device)
    comm.reduce_scatter_single(out, inp, torchcomms.ReduceOp.SUM, async_op=False)
    expected = sum(range(1, world_size + 1))
    assert torch.allclose(out, torch.full_like(out, expected)), \
        f"Eager reduce_scatter: got {out[0].item()}, expected {expected}"


def test_reduce_scatter_graph(rank, world_size, device, comm):
    expected = sum(range(1, world_size + 1))
    inp = torch.ones(8 * world_size, device=device) * (rank + 1)
    out = torch.empty(8, device=device)

    g = _capture_graph(device, lambda: comm.reduce_scatter_single(
        out, inp, torchcomms.ReduceOp.SUM, async_op=False))

    for i in range(3):
        inp.fill_(rank + 1)
        out.zero_()
        g.replay()
        torch.cuda.synchronize(device)
        assert torch.allclose(out, torch.full_like(out, expected)), \
            f"Graph replay {i}: got {out[0].item()}, expected {expected}"


def test_send_recv_eager(rank, world_size, device, comm):
    if world_size < 2:
        return
    if rank == 0:
        t = torch.full((4,), 99.0, device=device)
        comm.send(t, 1, async_op=False)
    elif rank == 1:
        t = torch.empty(4, device=device)
        comm.recv(t, 0, async_op=False)
        assert torch.allclose(t, torch.full_like(t, 99.0)), \
            f"Eager recv: got {t[0].item()}, expected 99"


def main():
    rank, world_size, device, comm = _setup()

    tests = [
        ("all_reduce_eager", test_all_reduce_eager),
        ("all_reduce_graph", test_all_reduce_graph),
        ("all_gather_eager", test_all_gather_eager),
        ("all_gather_graph", test_all_gather_graph),
        ("broadcast_eager", test_broadcast_eager),
        ("broadcast_graph", test_broadcast_graph),
        ("reduce_scatter_eager", test_reduce_scatter_eager),
        ("reduce_scatter_graph", test_reduce_scatter_graph),
        ("send_recv_eager", test_send_recv_eager),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn(rank, world_size, device, comm)
            if rank == 0:
                print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            if rank == 0:
                print(f"  FAIL  {name}: {e}")
            failed += 1

    if rank == 0:
        print(f"\n{passed}/{passed + failed} tests passed "
              f"(world_size={world_size})")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
