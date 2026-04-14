# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import torch

from vllm.distributed.parallel_state import get_pp_group


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> None:
    pp = get_pp_group()
    assert pp.is_last_rank

    assert sampled_token_ids.dtype == torch.int64
    pp.broadcast(sampled_token_ids.contiguous(), src=pp.rank_in_group)

    combined = torch.stack((num_sampled, num_rejected), dim=0)
    pp.broadcast(combined, src=pp.rank_in_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pp = get_pp_group()
    assert not pp.is_last_rank

    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    pp.broadcast(sampled_tokens, src=pp.world_size - 1)

    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    pp.broadcast(combined, src=pp.world_size - 1)
    num_sampled, num_rejected = combined.unbind(dim=0)
    return sampled_tokens, num_sampled, num_rejected
