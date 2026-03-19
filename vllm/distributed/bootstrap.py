# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap abstractions for distributed group creation.

A BootstrapProvider encapsulates rank discovery and ProcessGroup creation,
separating it from the communication layer (DeviceCommunicator, MessageQueue)
that lives in GroupCoordinator.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed
from torch.distributed import ProcessGroup


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
    """Abstract interface for creating distributed groups.

    Implementations decide *how* groups are formed (e.g. via
    ``torch.distributed.new_group`` or a custom store-based handshake).
    """

    @abstractmethod
    def create_group(
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


class ProcessGroupBootstrap(BootstrapProvider):
    """Default bootstrap using ``torch.distributed.new_group()``.

    This reproduces the original ``GroupCoordinator.__init__`` logic with
    zero behavioral change.
    """

    def create_group(
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


class TorchcommsBootstrap(BootstrapProvider):
    """Hybrid bootstrap: ProcessGroups + TorchComm communicators.

    Creates standard ``torch.distributed`` ProcessGroups (needed by
    ``GroupCoordinator`` for object-level communication such as
    ``broadcast_object``, ``send_object``, ``recv_object``, and
    ``MessageQueue``) **and** TorchComm communicators for device-level
    collectives (``all_reduce``, ``all_gather``, etc.).

    Precondition: ``torch.distributed.init_process_group()`` must have
    been called before using this provider (same as
    ``ProcessGroupBootstrap``).
    """

    def __init__(
        self,
        store: torch.distributed.Store | None = None,
        device: torch.device | None = None,
        timeout: timedelta | None = None,
        group_name: str | None = None,
    ) -> None:
        self._store = store
        self._device = device
        self._timeout = timeout or timedelta(seconds=300)
        self._group_name = group_name or "vllm"
        # Lazily-created world-level communicators.
        self._world_device_comm: Any | None = None
        self._world_cpu_comm: Any | None = None
        # Counter to generate unique sub-comm names across split() calls.
        self._split_counter: int = 0
        # Delegate for ProcessGroup creation.
        self._pg_bootstrap = ProcessGroupBootstrap()

    def _get_store(self) -> torch.distributed.Store:
        """Return the provided store or create a TCPStore from env vars."""
        if self._store is not None:
            return self._store
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return torch.distributed.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=self._timeout,
        )

    def _ensure_world_comms(self, backend: str) -> None:
        """Create world-level device and CPU comms if not already created."""
        if self._world_device_comm is not None:
            return

        import torchcomms

        store = self._get_store()
        device = self._device or torch.device("cuda", int(
            os.environ.get("LOCAL_RANK", "0")
        ))

        device_store = torch.distributed.PrefixStore(
            f"{self._group_name}/device", store
        )
        self._world_device_comm = torchcomms.new_comm(
            backend,
            device,
            name=f"{self._group_name}_world_device",
            store=device_store,
            timeout=self._timeout,
        )

        cpu_store = torch.distributed.PrefixStore(
            f"{self._group_name}/cpu", store
        )
        cpu_device = torch.device("cpu")
        self._world_cpu_comm = torchcomms.new_comm(
            "gloo",
            cpu_device,
            name=f"{self._group_name}_world_cpu",
            store=cpu_store,
            timeout=self._timeout,
        )

    def create_group(
        self,
        group_ranks: list[list[int]],
        global_rank: int,
        backend: str,
    ) -> BootstrapInfo:
        # 1. Create ProcessGroups via the standard path.
        pg_info = self._pg_bootstrap.create_group(
            group_ranks, global_rank, backend
        )

        # 2. Create TorchComm communicators.
        self._ensure_world_comms(backend)

        split_id = self._split_counter
        self._split_counter += 1

        device_sub = self._world_device_comm.split(
            group_ranks,
            name=f"{self._group_name}_device_split{split_id}",
            timeout=self._timeout,
        )
        cpu_sub = self._world_cpu_comm.split(
            group_ranks,
            name=f"{self._group_name}_cpu_split{split_id}",
            timeout=self._timeout,
        )

        # 3. Return combined info: PGs + TorchComm objects.
        return BootstrapInfo(
            rank=pg_info.rank,
            ranks=pg_info.ranks,
            world_size=pg_info.world_size,
            rank_in_group=pg_info.rank_in_group,
            cpu_group=pg_info.cpu_group,
            device_group=pg_info.device_group,
            device_comm=device_sub,
            cpu_comm=cpu_sub,
        )
