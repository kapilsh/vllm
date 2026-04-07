# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Coordination backend — abstracts metadata exchange for Custom AllReduce
# ---------------------------------------------------------------------------


class CoordinationBackend(ABC):
    """Abstraction for CPU-side metadata exchange used by Custom AllReduce.

    Custom AllReduce needs to exchange IPC handles, device IDs, and graph
    buffer metadata across ranks.  This can be done via either a
    ``torch.distributed`` ProcessGroup or a TorchComm communicator.
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        """Return this process's rank within the group."""
        ...

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Return the number of ranks in the group."""
        ...

    @property
    @abstractmethod
    def ranks(self) -> list[int]:
        """Return sorted global ranks in the group."""
        ...

    @abstractmethod
    def all_gather_object(self, obj: Any) -> list[Any]:
        """All-gather a picklable Python object from every rank."""
        ...

    @abstractmethod
    def broadcast_object_list(
        self, obj_list: list[Any], src: int
    ) -> None:
        """Broadcast an object list from ``src`` to all ranks."""
        ...

    @abstractmethod
    def in_same_node(self) -> bool:
        """Return True if all ranks are on the same node."""
        ...


class ProcessGroupCoordination(CoordinationBackend):
    """Coordination via a ``torch.distributed`` ProcessGroup (gloo)."""

    def __init__(self, group: ProcessGroup) -> None:
        self._group = group
        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

    @property
    def rank(self) -> int:
        return dist.get_rank(group=self._group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(group=self._group)

    @property
    def ranks(self) -> list[int]:
        return sorted(dist.get_process_group_ranks(group=self._group))

    def all_gather_object(self, obj: Any) -> list[Any]:
        result: list[Any] = [None] * self.world_size
        dist.all_gather_object(result, obj, group=self._group)
        return result

    def broadcast_object_list(
        self, obj_list: list[Any], src: int
    ) -> None:
        dist.broadcast_object_list(
            obj_list, src=src, group=self._group, device="cpu"
        )

    def in_same_node(self) -> bool:
        return all(in_the_same_node_as(self._group, source_rank=0))


class TorchCommCoordination(CoordinationBackend):
    """Coordination via a TorchComm communicator (no ProcessGroup needed)."""

    def __init__(
        self, cpu_comm: Any, rank: int, world_size: int
    ) -> None:
        self._comm = cpu_comm
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def ranks(self) -> list[int]:
        return list(range(self._world_size))

    def all_gather_object(self, obj: Any) -> list[Any]:
        import torchcomms.objcol
        result: list[Any] = [None] * self._world_size
        torchcomms.objcol.all_gather_object(
            self._comm, result, obj, weights_only=False
        )
        return result

    def broadcast_object_list(
        self, obj_list: list[Any], src: int
    ) -> None:
        import torchcomms.objcol
        torchcomms.objcol.broadcast_object_list(
            self._comm, obj_list, root=src, weights_only=False
        )

    def in_same_node(self) -> bool:
        return all(in_the_same_node_as(self._comm, source_rank=0))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


# ---------------------------------------------------------------------------
# CustomAllreduce
# ---------------------------------------------------------------------------


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup | None = None,
        device: int | str | torch.device = "cuda:0",
        max_size: int = 8192 * 1024,
        symm_mem_enabled: bool = False,
        cpu_comm: Any | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        """
        Args:
            group: ProcessGroup for coordination (standard path).
            device: the device to bind the CustomAllreduce to.
            cpu_comm: TorchComm for coordination (torchcomms path).
                Exactly one of ``group`` or ``cpu_comm`` must be provided.
            rank: explicit rank (required when using cpu_comm).
            world_size: explicit world_size (required when using cpu_comm).
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not custom_ar:
            logger.info(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        # Build the coordination backend.
        if cpu_comm is not None:
            assert rank is not None and world_size is not None, (
                "rank and world_size required with cpu_comm"
            )
            self._coord = TorchCommCoordination(cpu_comm, rank, world_size)
        elif group is not None:
            self._coord = ProcessGroupCoordination(group)
        else:
            raise ValueError("Either group or cpu_comm must be provided")

        if not self._coord.in_same_node():
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = self._coord.rank
        world_size = self._coord.world_size
        self.rank = rank
        if world_size == 1:
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES:
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(current_platform.device_count()))

        physical_device_id = device_ids[device.index]
        physical_device_ids = self._coord.all_gather_object(physical_device_id)

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert current_platform.is_cuda_alike()
        fully_connected = current_platform.is_fully_connected(physical_device_ids)
        if world_size > 2 and not fully_connected:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self._create_shared_buffer(
            ops.meta_size() + max_size, rank, world_size
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self._create_shared_buffer(max_size, rank, world_size)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(self.world_size)]
        all_data[self.rank] = [handle, offset]
        ranks = self._coord.ranks
        for i, rank in enumerate(ranks):
            self._coord.broadcast_object_list(all_data[i], src=rank)
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    # ------------------------------------------------------------------
    # AllReduce
    # ------------------------------------------------------------------

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.fully_connected:
            return inp_size < self.max_size
        return False

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    # Shared buffer management
    # ------------------------------------------------------------------

    def _create_shared_buffer(
        self,
        size_in_bytes: int,
        rank: int,
        world_size: int,
    ) -> list[int]:
        """Allocate an IPC buffer and exchange handles via the coord backend."""
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)
        handles = self._coord.all_gather_object(handle)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        """Legacy static method — uses torch.distributed directly."""
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])
