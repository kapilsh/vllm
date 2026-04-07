# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device communicator backed by a TorchComm object from torchcomms.

This replaces the NCCL/platform-specific collective path with calls
through a pre-created ``TorchComm`` communicator while keeping
ProcessGroups for metadata/object-level communication.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.distributed import ProcessGroup

from .base_device_communicator import DeviceCommunicatorBase

logger = logging.getLogger(__name__)


class TorchCommDeviceCommunicator(DeviceCommunicatorBase):
    """Uses a ``TorchComm`` object for device-level collectives.

    Optionally integrates Custom AllReduce (CUDA IPC) for small tensors
    where it outperforms NCCL.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup | None = None,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        device_comm: Any = None,
        unique_name: str = "",
        bootstrap_info: Any = None,
        use_custom_allreduce: bool = False,
        cpu_comm: Any = None,
    ):
        super().__init__(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
            bootstrap_info=bootstrap_info,
        )
        assert device_comm is not None, (
            "TorchCommDeviceCommunicator requires a TorchComm object"
        )
        self.comm = device_comm

        # Optional Custom AllReduce for small tensors (CUDA IPC).
        self.ca_comm = None
        if use_custom_allreduce and self.world_size > 1:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )
            self.ca_comm = CustomAllreduce(
                group=cpu_group,
                device=device,
                cpu_comm=cpu_comm,
                rank=self.rank,
                world_size=self.world_size,
            )
            if self.ca_comm.disabled:
                self.ca_comm = None
            else:
                logger.info(
                    "[torchcomms] Custom AllReduce enabled for %s",
                    unique_name,
                )

        logger.info("[torchcomms] TorchCommDeviceCommunicator initialized: "
                     "unique_name=%s, device=%s, rank=%d, world_size=%d, "
                     "comm=%s, ca_comm=%s",
                     unique_name, device, self.rank, self.world_size,
                     device_comm, self.ca_comm is not None)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Try Custom AllReduce first (fast IPC path for small tensors).
        if self.ca_comm is not None:
            out = self.ca_comm.custom_all_reduce(input_)
            if out is not None:
                return out

        import torchcomms

        # Must be out-of-place: the vllm::all_reduce custom op's fake
        # returns torch.empty_like(input_), so torch.compile assumes the
        # output is a distinct tensor.  In-place would corrupt CUDA graph
        # replay at TP>2.
        out = torch.empty_like(input_)
        out.copy_(input_)
        self.comm.all_reduce(out, torchcomms.ReduceOp.SUM, async_op=False)
        return out

    def all_gather(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()

        # Allocate flat output for all_gather_single.
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )

        self.comm.all_gather_single(output_tensor, input_, async_op=False)

        # Reshape to match DeviceCommunicatorBase convention.
        output_tensor = output_tensor.reshape(
            (self.world_size,) + input_size
        )
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1:]
        )
        return output_tensor

    def reduce_scatter(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_

        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        import torchcomms

        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % self.world_size == 0
        chunk_size = input_tensor.shape[0] // self.world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]
        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        self.comm.reduce_scatter_single(
            output_tensor, input_tensor,
            torchcomms.ReduceOp.SUM, async_op=False,
        )

        return output_tensor.movedim(0, dim).contiguous()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        self.comm.broadcast(tensor, src, async_op=False)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        self.comm.send(tensor, dst, async_op=False)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        self.comm.recv(tensor, src, async_op=False)
        return tensor

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            dim += input_.dim()

        if self.rank_in_group == dst:
            gather_list = [
                torch.empty_like(input_) for _ in range(self.world_size)
            ]
        else:
            gather_list = None

        self.comm.gather(gather_list, input_, dst, async_op=False)

        if self.rank_in_group == dst:
            return torch.cat(gather_list, dim=dim)
        return None

    def batch_isend_irecv(self, p2p_ops: list) -> None:
        """Batched P2P using torchcomms batch_op_create().

        Accepts a list of ``torch.distributed.P2POp`` objects (same
        interface as ``torch.distributed.batch_isend_irecv``) and
        executes them via the TorchComm batch API.
        """
        batch = self.comm.batch_op_create()
        for op in p2p_ops:
            op_name = getattr(op.op, "__name__", "")
            if op_name == "isend":
                batch.send(op.tensor, op.group_peer)
            elif op_name == "irecv":
                batch.recv(op.tensor, op.group_peer)
            else:
                raise ValueError(f"Unsupported P2P op: {op.op}")
        batch.issue(async_op=False)

    def destroy(self):
        if self.ca_comm is not None:
            self.ca_comm.close()
            self.ca_comm = None
