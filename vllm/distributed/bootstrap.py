# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap abstractions for distributed group creation.

A BootstrapProvider encapsulates rank discovery and ProcessGroup creation,
separating it from the communication layer (DeviceCommunicator, MessageQueue)
that lives in GroupCoordinator.
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
        pg_free: bool = False,
    ) -> None:
        self._store = store
        self._device = device
        self._timeout = timeout or timedelta(seconds=300)
        self._group_name = group_name or "vllm"
        self._pg_free = pg_free
        # Lazily-created world-level communicators.
        self._world_device_comm: Any | None = None
        self._world_cpu_comm: Any | None = None
        # Counter to generate unique sub-comm names across split() calls.
        self._split_counter: int = 0
        # Delegate for ProcessGroup creation.
        self._pg_bootstrap = ProcessGroupBootstrap()

        logger.info("[torchcomms] TorchcommsBootstrap created: "
                    "group_name=%s, timeout=%s, store=%s, device=%s, "
                    "pg_free=%s",
                    self._group_name, self._timeout,
                    type(store).__name__ if store else "None",
                    device, pg_free)

    def _get_store(self) -> torch.distributed.Store:
        """Return the provided store or get it from the default process group.

        When vLLM uses multiproc executor, it initializes torch.distributed
        with ``init_method=tcp://host:port`` rather than ``env://``, so
        MASTER_ADDR/MASTER_PORT env vars are not set.  We fall back to the
        store that ``init_process_group`` already created.
        """
        if self._store is not None:
            logger.info("[torchcomms] Using user-provided store: %s",
                        type(self._store).__name__)
            return self._store
        from torch.distributed.distributed_c10d import _get_default_store
        store = _get_default_store()
        logger.info("[torchcomms] Using default store from "
                    "torch.distributed: %s", type(store).__name__)
        return store

    def _ensure_world_comms(self, backend: str) -> None:
        """Create world-level device and CPU comms if not already created."""
        if self._world_device_comm is not None:
            logger.debug("[torchcomms] World comms already initialized, "
                         "skipping")
            return

        import torchcomms

        logger.info("[torchcomms] Initializing world-level communicators "
                    "(backend=%s, group=%s, timeout=%s)",
                    backend, self._group_name, self._timeout)

        store = self._get_store()
        device = self._device or torch.device("cuda", torch.cuda.current_device())
        logger.info("[torchcomms] Device for world comms: %s", device)

        # torchcomms.new_comm() and its backends read RANK, WORLD_SIZE,
        # LOCAL_RANK, MASTER_ADDR, and MASTER_PORT from env vars.  vLLM's
        # multiproc executor doesn't set these (it uses tcp:// init_method
        # instead of env://), so we populate them from the already-initialized
        # torch.distributed and the underlying TCPStore.
        env_vars_set = []
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(torch.distributed.get_rank())
            env_vars_set.append(f"RANK={os.environ['RANK']}")
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(torch.distributed.get_world_size())
            env_vars_set.append(f"WORLD_SIZE={os.environ['WORLD_SIZE']}")
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(device.index)
            env_vars_set.append(f"LOCAL_RANK={os.environ['LOCAL_RANK']}")
        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            # Unwrap PrefixStore layers to find the underlying TCPStore
            underlying = store
            prefix_layers = []
            while isinstance(underlying, torch.distributed.PrefixStore):
                prefix_layers.append(type(underlying).__name__)
                underlying = underlying.underlying_store
            if prefix_layers:
                logger.info("[torchcomms] Unwrapped %d PrefixStore layers "
                            "to reach %s",
                            len(prefix_layers), type(underlying).__name__)
            if isinstance(underlying, torch.distributed.TCPStore):
                os.environ.setdefault("MASTER_ADDR", underlying.host)
                os.environ.setdefault("MASTER_PORT", str(underlying.port))
                env_vars_set.append(f"MASTER_ADDR={underlying.host}")
                env_vars_set.append(f"MASTER_PORT={underlying.port}")
            else:
                # Fallback for local single-node usage
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29500")
                env_vars_set.append("MASTER_ADDR=127.0.0.1 (fallback)")
                env_vars_set.append("MASTER_PORT=29500 (fallback)")

        if env_vars_set:
            logger.info("[torchcomms] Populated env vars for torchcomms: %s",
                        ", ".join(env_vars_set))
        else:
            logger.info("[torchcomms] All env vars already set "
                        "(RANK=%s, WORLD_SIZE=%s, LOCAL_RANK=%s, "
                        "MASTER_ADDR=%s, MASTER_PORT=%s)",
                        os.environ.get("RANK"), os.environ.get("WORLD_SIZE"),
                        os.environ.get("LOCAL_RANK"),
                        os.environ.get("MASTER_ADDR"),
                        os.environ.get("MASTER_PORT"))

        device_store = torch.distributed.PrefixStore(
            f"{self._group_name}/device", store
        )
        logger.info("[torchcomms] Creating world device comm "
                    "(name=%s_world_device, backend=%s, device=%s)",
                    self._group_name, backend, device)
        self._world_device_comm = torchcomms.new_comm(
            backend,
            device,
            name=f"{self._group_name}_world_device",
            store=device_store,
            timeout=self._timeout,
        )
        logger.info("[torchcomms] World device comm created: %s",
                    self._world_device_comm)

        cpu_store = torch.distributed.PrefixStore(
            f"{self._group_name}/cpu", store
        )
        cpu_device = torch.device("cpu")
        logger.info("[torchcomms] Creating world CPU comm "
                    "(name=%s_world_cpu, backend=gloo, device=%s)",
                    self._group_name, cpu_device)
        self._world_cpu_comm = torchcomms.new_comm(
            "gloo",
            cpu_device,
            name=f"{self._group_name}_world_cpu",
            store=cpu_store,
            timeout=self._timeout,
        )
        logger.info("[torchcomms] World CPU comm created: %s",
                    self._world_cpu_comm)
        logger.info("[torchcomms] World-level communicator init complete")

    def create_group(
        self,
        group_ranks: list[list[int]],
        global_rank: int,
        backend: str,
    ) -> BootstrapInfo:
        logger.info("[torchcomms] create_group called: "
                    "group_ranks=%s, global_rank=%d, backend=%s, "
                    "pg_free=%s",
                    group_ranks, global_rank, backend, self._pg_free)

        # 1. Create ProcessGroups (unless pg_free mode).
        if self._pg_free:
            logger.info("[torchcomms] Step 1/3: Skipping ProcessGroup "
                        "creation (pg_free=True)")
            # Compute rank metadata locally instead of via PG bootstrap.
            result_ranks: list[int] | None = None
            for ranks in group_ranks:
                if global_rank in ranks:
                    result_ranks = ranks
                    break
            assert result_ranks is not None, (
                f"global_rank {global_rank} not found in any group_ranks"
            )
            pg_rank = global_rank
            pg_ranks = result_ranks
            pg_world_size = len(result_ranks)
            pg_rank_in_group = result_ranks.index(global_rank)
            pg_cpu_group = None
            pg_device_group = None
        else:
            logger.info("[torchcomms] Step 1/3: Creating ProcessGroups "
                        "via standard torch.distributed path")
            pg_info = self._pg_bootstrap.create_group(
                group_ranks, global_rank, backend
            )
            logger.info("[torchcomms] ProcessGroups created: "
                        "rank=%d, ranks=%s, world_size=%d, "
                        "rank_in_group=%d, "
                        "device_group=%s, cpu_group=%s",
                        pg_info.rank, pg_info.ranks, pg_info.world_size,
                        pg_info.rank_in_group, pg_info.device_group,
                        pg_info.cpu_group)
            pg_rank = pg_info.rank
            pg_ranks = pg_info.ranks
            pg_world_size = pg_info.world_size
            pg_rank_in_group = pg_info.rank_in_group
            pg_cpu_group = pg_info.cpu_group
            pg_device_group = pg_info.device_group

        # 2. Create TorchComm communicators.
        logger.info("[torchcomms] Step 2/3: Ensuring world-level "
                    "TorchComm communicators")
        self._ensure_world_comms(backend)

        split_id = self._split_counter
        self._split_counter += 1

        # group_ranks is a list of rank-groups (e.g. [[0,1], [2,3]]).
        # torchcomms.split() expects the flat rank list for the subgroup
        # that this process belongs to.
        my_ranks = None
        for ranks in group_ranks:
            if global_rank in ranks:
                my_ranks = ranks
                break
        if my_ranks is None:
            my_ranks = group_ranks[0]
            logger.warning("[torchcomms] global_rank %d not found in any "
                           "group_ranks, defaulting to first group: %s",
                           global_rank, my_ranks)

        logger.info("[torchcomms] Step 3/3: Splitting world comms for "
                    "subgroup (split_id=%d, my_ranks=%s)",
                    split_id, my_ranks)

        device_sub = self._world_device_comm.split(
            my_ranks,
            name=f"{self._group_name}_device_split{split_id}",
            timeout=self._timeout,
        )
        logger.info("[torchcomms] Device sub-comm created: "
                    "name=%s_device_split%d, comm=%s",
                    self._group_name, split_id, device_sub)

        cpu_sub = self._world_cpu_comm.split(
            my_ranks,
            name=f"{self._group_name}_cpu_split{split_id}",
            timeout=self._timeout,
        )
        logger.info("[torchcomms] CPU sub-comm created: "
                    "name=%s_cpu_split%d, comm=%s",
                    self._group_name, split_id, cpu_sub)

        # 3. Return combined info: PGs (if created) + TorchComm objects.
        logger.info("[torchcomms] create_group complete for "
                    "global_rank=%d: split_id=%d, "
                    "world_size=%d, rank_in_group=%d, pg_free=%s",
                    global_rank, split_id,
                    pg_world_size, pg_rank_in_group, self._pg_free)
        return BootstrapInfo(
            rank=pg_rank,
            ranks=pg_ranks,
            world_size=pg_world_size,
            rank_in_group=pg_rank_in_group,
            cpu_group=pg_cpu_group,
            device_group=pg_device_group,
            device_comm=device_sub,
            cpu_comm=cpu_sub,
        )
