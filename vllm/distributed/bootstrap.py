# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap abstractions for distributed group creation.

A BootstrapProvider encapsulates rank discovery and ProcessGroup creation,
separating it from the communication layer (DeviceCommunicator, MessageQueue)
that lives in GroupCoordinator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    cpu_group: ProcessGroup  # gloo group for CPU coordination
    device_group: ProcessGroup  # group for device (e.g. NCCL) communication


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
