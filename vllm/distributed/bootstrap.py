# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap abstractions for distributed group creation.

A BootstrapProvider encapsulates rank discovery, initialization, and group
creation — separating it from the communication layer
(DeviceCommunicator, MessageQueue) that lives in GroupCoordinator.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


@dataclass
class BootstrapInfo:
    """Result of bootstrapping a single distributed group."""

    rank: int  # global rank of this process
    ranks: list[int]  # global ranks participating in the group
    world_size: int  # number of ranks in the group
    rank_in_group: int  # index of this rank within *ranks*
    cpu_group: ProcessGroup | None = None  # gloo group for CPU coordination
    device_group: ProcessGroup | None = None  # device (NCCL) group
    device_comm: Any | None = None  # TorchComm for device communication
    cpu_comm: Any | None = None  # TorchComm for CPU (gloo) communication


class BootstrapProvider(ABC):
    """Abstract interface for distributed initialization and group creation.

    Implementations decide *how* the distributed environment is set up
    and how groups are formed (e.g. via ``torch.distributed`` or a custom
    store-based handshake).
    """

    @abstractmethod
    def init(
        self,
        rank: int,
        world_size: int,
        backend: str,
        init_method: str | None = None,
        timeout: timedelta | None = None,
    ) -> None:
        """Initialize the distributed environment."""
        ...

    @abstractmethod
    def is_initialized(self) -> bool:
        """Return True if the distributed environment has been initialized."""
        ...

    @abstractmethod
    def get_rank(self) -> int:
        """Return the global rank of this process."""
        ...

    @abstractmethod
    def get_world_size(self) -> int:
        """Return the total number of processes."""
        ...

    @abstractmethod
    def get_backend(self) -> str:
        """Return the backend string (e.g. 'nccl', 'gloo')."""
        ...

    @abstractmethod
    def get_bootstrap_info(
        self,
        group_ranks: list[list[int]],
        global_rank: int,
        backend: str,
    ) -> BootstrapInfo:
        """Create device and CPU groups for every rank-list and return info
        for the group that *global_rank* belongs to.

        Args:
            group_ranks: One or more lists of global ranks.  Exactly one list
                must contain *global_rank*.
            global_rank: The calling process's global rank.
            backend: Backend string for device communication (e.g. ``"nccl"``).

        Returns:
            A :class:`BootstrapInfo` for the group containing *global_rank*.
        """
        ...

    @abstractmethod
    def destroy(self) -> None:
        """Tear down the distributed environment."""
        ...


class ProcessGroupBootstrap(BootstrapProvider):
    """Default bootstrap using ``torch.distributed``.

    This reproduces the original ``GroupCoordinator.__init__`` logic with
    zero behavioral change.
    """

    def init(
        self,
        rank: int,
        world_size: int,
        backend: str,
        init_method: str | None = None,
        timeout: timedelta | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "backend": backend,
            "world_size": world_size,
            "rank": rank,
        }
        if init_method is not None:
            kwargs["init_method"] = init_method
        if timeout is not None:
            kwargs["timeout"] = timeout
        # When TORCH_DISTRIBUTED_USE_TORCHCOMMS=1, the TorchComm backend
        # ignores the rank/world_size kwargs and reads env vars instead.
        # Also needs MASTER_ADDR/MASTER_PORT for its StoreManager.
        # Propagate them so vLLM's multiproc-spawned workers can init.
        if os.environ.get("TORCH_DISTRIBUTED_USE_TORCHCOMMS") == "1":
            os.environ["TORCHCOMM_RANK"] = str(rank)
            os.environ["TORCHCOMM_SIZE"] = str(world_size)
            if init_method and init_method.startswith("tcp://"):
                # Parse "tcp://host:port" into MASTER_ADDR and MASTER_PORT
                addr_port = init_method[len("tcp://"):]
                host, port = addr_port.rsplit(":", 1)
                os.environ.setdefault("MASTER_ADDR", host)
                os.environ.setdefault("MASTER_PORT", port)
        torch.distributed.init_process_group(**kwargs)

    def is_initialized(self) -> bool:
        return torch.distributed.is_initialized()

    def get_rank(self) -> int:
        return torch.distributed.get_rank()

    def get_world_size(self) -> int:
        return torch.distributed.get_world_size()

    def get_backend(self) -> str:
        return str(torch.distributed.get_backend())

    def get_bootstrap_info(
        self,
        group_ranks: list[list[int]],
        global_rank: int,
        backend: str,
    ) -> BootstrapInfo:
        # Deferred import: suppress_stdout lives in system_utils which
        # transitively imports current_platform → cuda.py → vllm._C.
        # Importing at call time (rather than module level) keeps
        # bootstrap.py importable without compiled extensions.
        from vllm.utils.system_utils import suppress_stdout

        result_device_group = None
        result_cpu_group = None
        result_ranks: list[int] | None = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=backend
            )
            # A group with gloo backend to allow direct coordination
            # between processes through the CPU.
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(
                    ranks, backend="gloo"
                )
            if global_rank in ranks:
                result_ranks = ranks
                result_device_group = device_group
                result_cpu_group = cpu_group

        assert result_ranks is not None, (
            f"global_rank {global_rank} not found in any group_ranks"
        )
        assert result_cpu_group is not None
        assert result_device_group is not None

        return BootstrapInfo(
            rank=global_rank,
            ranks=result_ranks,
            world_size=len(result_ranks),
            rank_in_group=result_ranks.index(global_rank),
            cpu_group=result_cpu_group,
            device_group=result_device_group,
        )

    def destroy(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


