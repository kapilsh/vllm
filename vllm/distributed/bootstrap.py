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


class TorchcommsBootstrap(BootstrapProvider):
    """ProcessGroup-free bootstrap using TorchComm communicators.

    Creates TorchComm communicators for all communication — both
    device-level collectives (``all_reduce``, ``all_gather``, etc.)
    and object-level operations (via ``torchcomms.objcol``).

    No ``torch.distributed.new_group()`` calls are made.  Rank metadata
    is computed locally from ``group_ranks``.

    Unlike ``ProcessGroupBootstrap``, this provider does NOT call
    ``torch.distributed.init_process_group()``.  Instead, ``init()``
    stores rank/world_size locally and populates env vars so that
    torchcomms can self-bootstrap.
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
        # Distributed state — populated by init().
        self._rank: int = -1
        self._world_size: int = -1
        self._backend: str = ""
        self._initialized: bool = False

        logger.info("[torchcomms] TorchcommsBootstrap created: "
                    "group_name=%s, timeout=%s, store=%s, device=%s",
                    self._group_name, self._timeout,
                    type(store).__name__ if store else "None",
                    device)

    def init(
        self,
        rank: int,
        world_size: int,
        backend: str,
        init_method: str | None = None,
        timeout: timedelta | None = None,
    ) -> None:
        self._rank = rank
        self._world_size = world_size
        self._backend = backend
        self._initialized = True
        if timeout is not None:
            self._timeout = timeout

        logger.info("[torchcomms] init called: rank=%d, world_size=%d, "
                    "backend=%s, init_method=%s",
                    rank, world_size, backend, init_method)

        # Populate env vars for torchcomms backends.
        env_vars_set = []
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(rank)
            env_vars_set.append(f"RANK={rank}")
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(world_size)
            env_vars_set.append(f"WORLD_SIZE={world_size}")
        if "LOCAL_RANK" not in os.environ:
            device = self._device or torch.device(
                "cuda", torch.cuda.current_device()
                if torch.cuda.is_available() else 0
            )
            os.environ["LOCAL_RANK"] = str(device.index or 0)
            env_vars_set.append(f"LOCAL_RANK={os.environ['LOCAL_RANK']}")

        # Parse init_method for master_addr/port if provided.
        if init_method and init_method.startswith("tcp://"):
            # Format: tcp://host:port
            addr_part = init_method[len("tcp://"):]
            if ":" in addr_part:
                host, port_str = addr_part.rsplit(":", 1)
                os.environ.setdefault("MASTER_ADDR", host)
                os.environ.setdefault("MASTER_PORT", port_str)
                if "MASTER_ADDR" not in [v.split("=")[0]
                                          for v in env_vars_set]:
                    env_vars_set.append(f"MASTER_ADDR={host}")
                    env_vars_set.append(f"MASTER_PORT={port_str}")
        elif "MASTER_ADDR" not in os.environ:
            # Fallback for single-node usage.
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            env_vars_set.append("MASTER_ADDR=127.0.0.1 (fallback)")
            env_vars_set.append("MASTER_PORT=29500 (fallback)")

        if env_vars_set:
            logger.info("[torchcomms] Populated env vars: %s",
                        ", ".join(env_vars_set))
        else:
            logger.info("[torchcomms] All env vars already set "
                        "(RANK=%s, WORLD_SIZE=%s, LOCAL_RANK=%s, "
                        "MASTER_ADDR=%s, MASTER_PORT=%s)",
                        os.environ.get("RANK"), os.environ.get("WORLD_SIZE"),
                        os.environ.get("LOCAL_RANK"),
                        os.environ.get("MASTER_ADDR"),
                        os.environ.get("MASTER_PORT"))

    def is_initialized(self) -> bool:
        return self._initialized

    def get_rank(self) -> int:
        return self._rank

    def get_world_size(self) -> int:
        return self._world_size

    def get_backend(self) -> str:
        return self._backend

    def _get_store(self) -> torch.distributed.Store:
        """Return the provided store or create one from env vars.

        When a store was provided at construction time, use it directly.
        Otherwise, create a TCPStore from MASTER_ADDR/MASTER_PORT env vars
        (which init() will have populated).
        """
        if self._store is not None:
            logger.info("[torchcomms] Using user-provided store: %s",
                        type(self._store).__name__)
            return self._store
        # Create a TCPStore from env vars — no torch.distributed dependency.
        host = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = int(os.environ.get("MASTER_PORT", "29500"))
        is_master = (self._rank == 0)
        store = torch.distributed.TCPStore(
            host_name=host,
            port=port,
            world_size=self._world_size,
            is_master=is_master,
            timeout=self._timeout,
        )
        logger.info("[torchcomms] Created TCPStore from env vars: "
                    "host=%s, port=%d, is_master=%s",
                    host, port, is_master)
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
        device = self._device or torch.device(
            "cuda", torch.cuda.current_device()
        )
        logger.info("[torchcomms] Device for world comms: %s", device)

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

    def get_bootstrap_info(
        self,
        group_ranks: list[list[int]],
        global_rank: int,
        backend: str,
    ) -> BootstrapInfo:
        logger.info("[torchcomms] get_bootstrap_info called: "
                    "group_ranks=%s, global_rank=%d, backend=%s",
                    group_ranks, global_rank, backend)

        # 1. Compute rank metadata locally (no ProcessGroup creation).
        logger.info("[torchcomms] Step 1/3: Computing rank metadata "
                    "(no ProcessGroup creation)")
        result_ranks: list[int] | None = None
        for ranks in group_ranks:
            if global_rank in ranks:
                result_ranks = ranks
                break
        assert result_ranks is not None, (
            f"global_rank {global_rank} not found in any group_ranks"
        )

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

        # 3. Return info with TorchComm objects only (no ProcessGroups).
        logger.info("[torchcomms] get_bootstrap_info complete for "
                    "global_rank=%d: split_id=%d, "
                    "world_size=%d, rank_in_group=%d",
                    global_rank, split_id,
                    len(result_ranks), result_ranks.index(global_rank))
        return BootstrapInfo(
            rank=global_rank,
            ranks=result_ranks,
            world_size=len(result_ranks),
            rank_in_group=result_ranks.index(global_rank),
            cpu_group=None,
            device_group=None,
            device_comm=device_sub,
            cpu_comm=cpu_sub,
        )

    def destroy(self) -> None:
        import ctypes

        logger.info("[torchcomms] Destroying TorchcommsBootstrap")
        self._initialized = False
        # TorchCommNCCL's C++ destructor calls ncclCommAbort() which
        # blocks indefinitely on a thread join waiting for peers.
        # Prevent the destructor from ever running by permanently
        # incrementing the refcount.  The OS cleans up on process exit.
        if self._world_device_comm is not None:
            ctypes.pythonapi.Py_IncRef(
                ctypes.py_object(self._world_device_comm))
            self._world_device_comm = None
        if self._world_cpu_comm is not None:
            ctypes.pythonapi.Py_IncRef(
                ctypes.py_object(self._world_cpu_comm))
            self._world_cpu_comm = None
