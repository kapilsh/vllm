# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the BootstrapProvider abstraction.

Run `pytest tests/distributed/test_bootstrap.py`.
"""

import unittest
from unittest.mock import MagicMock, patch

from vllm.distributed.bootstrap import (
    BootstrapInfo,
    ProcessGroupBootstrap,
)


class BootstrapInfoTest(unittest.TestCase):
    """Unit tests for BootstrapInfo dataclass construction."""

    def test_construction(self):
        cpu_group = MagicMock()
        device_group = MagicMock()
        info = BootstrapInfo(
            rank=1,
            ranks=[0, 1, 2, 3],
            world_size=4,
            rank_in_group=1,
            cpu_group=cpu_group,
            device_group=device_group,
        )
        self.assertEqual(info.rank, 1)
        self.assertEqual(info.ranks, [0, 1, 2, 3])
        self.assertEqual(info.world_size, 4)
        self.assertEqual(info.rank_in_group, 1)
        self.assertIs(info.cpu_group, cpu_group)
        self.assertIs(info.device_group, device_group)

    def test_rank_in_group_matches_index(self):
        info = BootstrapInfo(
            rank=2,
            ranks=[0, 2, 4],
            world_size=3,
            rank_in_group=1,
            cpu_group=MagicMock(),
            device_group=MagicMock(),
        )
        self.assertEqual(info.rank_in_group, info.ranks.index(info.rank))


class ProcessGroupBootstrapTest(unittest.TestCase):
    """Tests for ProcessGroupBootstrap using mocked torch.distributed."""

    @patch("vllm.utils.system_utils.suppress_stdout")
    @patch("vllm.distributed.bootstrap.torch.distributed.new_group")
    def test_single_group(self, mock_new_group, mock_suppress):
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)

        device_pg = MagicMock(name="device_pg")
        cpu_pg = MagicMock(name="cpu_pg")
        mock_new_group.side_effect = [device_pg, cpu_pg]

        bootstrap = ProcessGroupBootstrap()
        info = bootstrap.create_group(
            group_ranks=[[0, 1]],
            global_rank=0,
            backend="nccl",
        )

        self.assertEqual(info.rank, 0)
        self.assertEqual(info.ranks, [0, 1])
        self.assertEqual(info.world_size, 2)
        self.assertEqual(info.rank_in_group, 0)
        self.assertIs(info.cpu_group, cpu_pg)
        self.assertIs(info.device_group, device_pg)

        self.assertEqual(mock_new_group.call_count, 2)
        mock_new_group.assert_any_call([0, 1], backend="nccl")
        mock_new_group.assert_any_call([0, 1], backend="gloo")

    @patch("vllm.utils.system_utils.suppress_stdout")
    @patch("vllm.distributed.bootstrap.torch.distributed.new_group")
    def test_multiple_groups_selects_correct(self, mock_new_group, mock_suppress):
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)

        device_pg_01 = MagicMock(name="device_pg_01")
        cpu_pg_01 = MagicMock(name="cpu_pg_01")
        device_pg_23 = MagicMock(name="device_pg_23")
        cpu_pg_23 = MagicMock(name="cpu_pg_23")
        mock_new_group.side_effect = [
            device_pg_01, cpu_pg_01,
            device_pg_23, cpu_pg_23,
        ]

        bootstrap = ProcessGroupBootstrap()
        info = bootstrap.create_group(
            group_ranks=[[0, 1], [2, 3]],
            global_rank=3,
            backend="nccl",
        )

        self.assertEqual(info.rank, 3)
        self.assertEqual(info.ranks, [2, 3])
        self.assertEqual(info.world_size, 2)
        self.assertEqual(info.rank_in_group, 1)
        self.assertIs(info.device_group, device_pg_23)
        self.assertIs(info.cpu_group, cpu_pg_23)

    @patch("vllm.utils.system_utils.suppress_stdout")
    @patch("vllm.distributed.bootstrap.torch.distributed.new_group")
    def test_rank_not_in_any_group_raises(self, mock_new_group, mock_suppress):
        mock_suppress.return_value.__enter__ = MagicMock()
        mock_suppress.return_value.__exit__ = MagicMock(return_value=False)

        mock_new_group.return_value = MagicMock()

        bootstrap = ProcessGroupBootstrap()
        with self.assertRaises(AssertionError):
            bootstrap.create_group(
                group_ranks=[[0, 1]],
                global_rank=5,
                backend="nccl",
            )
