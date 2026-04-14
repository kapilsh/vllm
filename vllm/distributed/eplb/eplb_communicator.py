# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
EPLB communicator implementations and factory.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch.distributed import (
    P2POp,
    ProcessGroup,
    batch_isend_irecv,
)

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import (
    ncclDataTypeEnum,
)
from vllm.distributed.parallel_state import GroupCoordinator, is_local_first_rank
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import init_logger

logger = init_logger(__name__)


class EplbCommunicator(ABC):
    """Abstract EPLB communicator for expert weight transfers."""

    @abstractmethod
    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        pass

    @abstractmethod
    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

    def set_stream(self, cuda_stream: torch.cuda.Stream | None) -> None:
        self._cuda_stream = cuda_stream

    def _log_initialized(self) -> None:
        if is_local_first_rank():
            logger.info("Initialized EPLB communicator: %s.", self.__class__.__name__)


class TorchDistNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by torch.distributed isend/irecv."""

    def __init__(
        self,
        ep_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._ep_group = ep_group
        self._cuda_stream = cuda_stream
        self._p2p_ops: list[P2POp] = []
        self._log_initialized()

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._p2p_ops.append(
            P2POp(
                torch.distributed.isend,
                tensor,
                dst_rank,
                self._ep_group,
            )
        )

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._p2p_ops.append(
            P2POp(
                torch.distributed.irecv,
                tensor,
                src_rank,
                self._ep_group,
            )
        )

    def execute(self) -> None:
        if not self._p2p_ops:
            return
        try:
            with torch.cuda.stream(self._cuda_stream):
                reqs = batch_isend_irecv(self._p2p_ops)
                for req in reqs:
                    req.wait()
        finally:
            self._p2p_ops.clear()


class TorchDistGlooStagedEplbCommunicator(EplbCommunicator):
    """EPLB communicator using gloo P2P with CPU staging."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._cpu_group = cpu_group
        self._cuda_stream = cuda_stream
        self._ops: list[tuple[str, torch.Tensor, int]] = []
        self._log_initialized()

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._ops.append(("send", tensor, dst_rank))

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._ops.append(("recv", tensor, src_rank))

    def execute(self) -> None:
        if not self._ops:
            return

        p2p_ops: list[P2POp] = []
        recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []

        def build_ops() -> None:
            for op, tensor, peer_rank in self._ops:
                if op == "send":
                    cpu_tensor = tensor.to(device="cpu", non_blocking=True)
                    p2p_ops.append(
                        P2POp(
                            torch.distributed.isend,
                            cpu_tensor,
                            peer_rank,
                            self._cpu_group,
                        )
                    )
                    continue
                cpu_tensor = torch.empty_like(tensor, device="cpu")
                p2p_ops.append(
                    P2POp(
                        torch.distributed.irecv,
                        cpu_tensor,
                        peer_rank,
                        self._cpu_group,
                    )
                )
                recv_staging.append((tensor, cpu_tensor))

        try:
            with torch.cuda.stream(self._cuda_stream):
                build_ops()
        finally:
            self._ops.clear()

        # Wait for all D2H copies to finish
        # before issuing gloo batch_isend_irecv operations.
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()
        else:
            torch.cuda.current_stream().synchronize()

        reqs = batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

        if not recv_staging:
            return
        with torch.cuda.stream(self._cuda_stream):
            for dst_tensor, cpu_tensor in recv_staging:
                dst_tensor.copy_(cpu_tensor, non_blocking=True)


class TorchCommEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by a TorchComm object.

    Uses ``batch_op_create()`` to collect send/recv ops, then issues
    them all at once with ``issue()``.  This is the torchcomms equivalent
    of ``torch.distributed.batch_isend_irecv``.

    Note: Currently validated for EP groups of size 2 (the common case
    for single-node TP+EP). Larger EP groups may require torchcomms
    batch_op_create fixes for partial-participation P2P patterns.
    """

    def __init__(
        self,
        device_comm: object,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._comm = device_comm
        self._cuda_stream = cuda_stream
        self._batch = None
        self._log_initialized()

    def _ensure_batch(self) -> None:
        if self._batch is None:
            self._batch = self._comm.batch_op_create()

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._ensure_batch()
        self._batch.send(tensor, dst_rank)

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._ensure_batch()
        self._batch.recv(tensor, src_rank)

    def execute(self) -> None:
        if self._batch is None:
            return
        try:
            with torch.cuda.stream(self._cuda_stream):
                work = self._batch.issue(async_op=False)
        finally:
            self._batch = None


class PyNcclEplbCommunicator(EplbCommunicator):
    """EPLB communicator backed by PyNcclCommunicator using ncclSend/ncclRecv."""

    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        cuda_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._pynccl_comm = pynccl_comm
        self._cuda_stream = cuda_stream
        self._group_started = False
        self._log_initialized()

    def _ensure_group_started(self) -> None:
        if not self._group_started:
            self._pynccl_comm.group_start()
            self._group_started = True

    def add_send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        self._ensure_group_started()
        self._pynccl_comm.send(tensor, dst_rank, stream=self._cuda_stream)

    def add_recv(self, tensor: torch.Tensor, src_rank: int) -> None:
        self._ensure_group_started()
        self._pynccl_comm.recv(tensor, src_rank, stream=self._cuda_stream)

    def execute(self) -> None:
        if self._group_started:
            self._pynccl_comm.group_end()
            self._group_started = False


# ---------------------------------------------------------------------------
# EplbGroupContext — abstraction for group-level operations in EPLB
# ---------------------------------------------------------------------------


class EplbGroupContext(ABC):
    """Thin abstraction over the group-level operations EPLB needs.

    EPLB uses rank/size queries and collectives (all_reduce, all_gather,
    barrier) that previously went through a raw ``ProcessGroup``.  This
    interface lets the same code work with either a PG or a TorchComm
    communicator.
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        """Rank within the EP group."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of ranks in the EP group."""
        ...

    @abstractmethod
    def all_reduce_sum(self, tensor: torch.Tensor) -> None:
        """In-place SUM all-reduce across the group."""
        ...

    @abstractmethod
    def all_reduce_max(self, tensor: torch.Tensor) -> None:
        """In-place MAX all-reduce across the group."""
        ...

    @abstractmethod
    def all_gather(
        self, output_list: list[torch.Tensor], input_tensor: torch.Tensor
    ) -> None:
        """All-gather tensors into *output_list*."""
        ...

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all ranks."""
        ...


class ProcessGroupEplbContext(EplbGroupContext):
    """Backed by a raw ``torch.distributed`` ProcessGroup."""

    def __init__(self, pg: ProcessGroup) -> None:
        self._pg = pg

    @property
    def rank(self) -> int:
        return self._pg.rank()

    @property
    def size(self) -> int:
        return self._pg.size()

    def all_reduce_sum(self, tensor: torch.Tensor) -> None:
        torch.distributed.all_reduce(tensor, group=self._pg)

    def all_reduce_max(self, tensor: torch.Tensor) -> None:
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=self._pg
        )

    def all_gather(
        self, output_list: list[torch.Tensor], input_tensor: torch.Tensor
    ) -> None:
        torch.distributed.all_gather(output_list, input_tensor, group=self._pg)

    def barrier(self) -> None:
        torch.distributed.barrier(group=self._pg)


class TorchCommEplbContext(EplbGroupContext):
    """Backed by a TorchComm device communicator (no ProcessGroup needed)."""

    def __init__(self, device_comm: Any, rank: int, size: int) -> None:
        self._comm = device_comm
        self._rank = rank
        self._size = size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    def all_reduce_sum(self, tensor: torch.Tensor) -> None:
        import torchcomms

        self._comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

    def all_reduce_max(self, tensor: torch.Tensor) -> None:
        import torchcomms

        self._comm.all_reduce(tensor, torchcomms.ReduceOp.MAX, async_op=False)

    def all_gather(
        self, output_list: list[torch.Tensor], input_tensor: torch.Tensor
    ) -> None:
        output_tensor = torch.cat(
            [torch.empty_like(input_tensor) for _ in output_list], dim=0
        )
        self._comm.all_gather_single(
            output_tensor, input_tensor, async_op=False
        )
        chunks = output_tensor.chunk(self._size, dim=0)
        for i, chunk in enumerate(chunks):
            output_list[i].copy_(chunk)

    def barrier(self) -> None:
        self._comm.barrier(async_op=False)


def create_eplb_group_context(
    group_coordinator: GroupCoordinator,
) -> EplbGroupContext:
    """Build the right ``EplbGroupContext`` for *group_coordinator*."""
    from vllm.distributed.device_communicators.torchcomm_communicator import (
        TorchCommDeviceCommunicator,
    )

    device_comm = group_coordinator.device_communicator
    if isinstance(device_comm, TorchCommDeviceCommunicator):
        return TorchCommEplbContext(
            device_comm=device_comm.comm,
            rank=group_coordinator.rank_in_group,
            size=group_coordinator.world_size,
        )
    pg = group_coordinator.device_group
    if pg is None:
        raise RuntimeError(
            "EPLB requires either a TorchComm device communicator or a "
            "device ProcessGroup, but neither is available."
        )
    return ProcessGroupEplbContext(pg)


def create_eplb_communicator(
    group_coordinator: GroupCoordinator,
    backend: str | None,
    expert_weights: Sequence[torch.Tensor],
) -> EplbCommunicator:
    # Keep a safe default for callers that have not resolved communicator yet.
    if backend is None:
        backend = "torch_nccl"

    tensor_device_type = expert_weights[0].device.type if expert_weights else "cpu"
    torch_group = (
        group_coordinator.cpu_group
        if tensor_device_type == "cpu"
        else group_coordinator.device_group
    )

    def _create_pynccl() -> EplbCommunicator:
        if tensor_device_type == "cpu":
            raise RuntimeError(
                "EPLB communicator 'pynccl' supports only cuda-like devices "
                f"(got {tensor_device_type})."
            )
        unsupported_dtypes = sorted(
            {
                tensor.dtype
                for tensor in expert_weights
                if not ncclDataTypeEnum.supports_torch_dtype(tensor.dtype)
            },
            key=str,
        )
        if unsupported_dtypes:
            raise RuntimeError(
                "EPLB communicator 'pynccl' requested but expert weights contain "
                "unsupported dtypes: "
                f"({', '.join(str(dtype) for dtype in unsupported_dtypes)})."
            )

        device_comm = group_coordinator.device_communicator
        pynccl_comm = (
            getattr(device_comm, "pynccl_comm", None)
            if device_comm is not None
            else None
        )
        if pynccl_comm is None or pynccl_comm.disabled or not pynccl_comm.available:
            raise RuntimeError("EPLB communicator 'pynccl' requested but unavailable.")
        try:
            return PyNcclEplbCommunicator(pynccl_comm=pynccl_comm)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize PyNcclEplbCommunicator ({exc})."
            ) from exc

    # Auto-detect torchcomms: if the device communicator is a
    # TorchCommDeviceCommunicator and no explicit backend was requested
    # (or torch_nccl was requested), use TorchComm for EPLB.
    device_comm = group_coordinator.device_communicator
    from vllm.distributed.device_communicators.torchcomm_communicator import (
        TorchCommDeviceCommunicator,
    )
    if isinstance(device_comm, TorchCommDeviceCommunicator) and backend in (
        "torch_nccl", None
    ):
        return TorchCommEplbCommunicator(
            device_comm=device_comm.comm,
        )

    is_stateless = isinstance(group_coordinator, StatelessGroupCoordinator)
    if is_stateless:
        if backend not in ("torch_nccl", "pynccl"):
            raise ValueError(
                f"Elastic EP requires 'torch_nccl' or 'pynccl' EPLB communicator "
                f"(got '{backend}'). torch_gloo is not supported with stateless groups."
            )
        if backend == "torch_nccl":
            logger.warning(
                "Stateless elastic EP requires PyNCCL backend. "
                "Forcing EPLB communicator to 'pynccl'."
            )
            backend = "pynccl"
        return _create_pynccl()

    if backend == "torch_gloo":
        return TorchDistGlooStagedEplbCommunicator(
            cpu_group=group_coordinator.cpu_group,
        )
    elif backend == "torch_nccl":
        return TorchDistNcclEplbCommunicator(ep_group=torch_group)
    elif backend == "pynccl":
        return _create_pynccl()
    raise ValueError(f"Unknown EPLB communicator backend: {backend}")
