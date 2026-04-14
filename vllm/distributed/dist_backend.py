# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Drop-in distributed backend shim.

When torchcomms is installed, this module re-exports from its compatibility
layer which provides the same API as ``torch.distributed`` but can optionally
route collectives through the torchcomms backend (enabled via
``init_process_group(use_torchcomms=True)``).

When torchcomms is **not** installed, everything falls back to plain
``torch.distributed`` and behaviour is identical to upstream vLLM.
"""

from __future__ import annotations

import types

import torch.distributed as _torch_dist

from vllm.logger import init_logger

logger = init_logger(__name__)

TORCHCOMMS_AVAILABLE: bool

try:
    from torchcomms import distwrap as _distwrap  # type: ignore[import-untyped]

    TORCHCOMMS_AVAILABLE = True
    logger.info(
        "torchcomms is available. "
        "Use --use-torchcomms to enable torchcomms routing."
    )
except ImportError:
    _distwrap = None  # type: ignore[assignment]
    TORCHCOMMS_AVAILABLE = False
    logger.info(
        "torchcomms is not installed. "
        "Using torch.distributed as the distributed backend."
    )


class _DistProxy(types.ModuleType):
    """Module proxy that forwards to torchcomms when available, else torch.distributed.

    For attributes that torchcomms doesn't explicitly define (e.g. ``get_backend``,
    ``is_backend_available``), the proxy falls through to ``torch.distributed``.
    """

    def __init__(self) -> None:
        super().__init__("vllm.distributed.dist_backend._dist_proxy")
        self._primary = _distwrap if _distwrap is not None else _torch_dist
        self._fallback = _torch_dist

    def __getattr__(self, name: str) -> object:
        try:
            return getattr(self._primary, name)
        except AttributeError:
            return getattr(self._fallback, name)


dist: types.ModuleType = _DistProxy() if TORCHCOMMS_AVAILABLE else _torch_dist

# Re-export commonly used symbols so callers can write:
#   from vllm.distributed.dist_backend import dist, ProcessGroup
# These are identical objects on both torch.distributed and torchcomms.
ProcessGroup = _torch_dist.ProcessGroup
ReduceOp = _torch_dist.ReduceOp
Store = _torch_dist.Store
P2POp = _torch_dist.P2POp


def is_torchcomms_enabled() -> bool:
    """Check if torchcomms routing is currently active.

    This is True only when torchcomms is installed AND was enabled via
    ``init_process_group(use_torchcomms=True)``.
    """
    if not TORCHCOMMS_AVAILABLE:
        return False
    from torchcomms.distwrap.utils import torchcomms_is_enabled
    return torchcomms_is_enabled()


__all__ = ["dist", "TORCHCOMMS_AVAILABLE", "is_torchcomms_enabled"]
