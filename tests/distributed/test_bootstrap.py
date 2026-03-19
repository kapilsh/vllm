# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the BootstrapProvider abstraction.

Run `pytest tests/distributed/test_bootstrap.py`.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm.distributed.bootstrap import (
    BootstrapInfo,
    ProcessGroupBootstrap,
    TorchcommsBootstrap,
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


class BootstrapInfoOptionalFieldsTest(unittest.TestCase):
    """Tests for the new optional fields on BootstrapInfo."""

    def test_optional_groups_default_to_none(self):
        info = BootstrapInfo(
            rank=0,
            ranks=[0, 1],
            world_size=2,
            rank_in_group=0,
        )
        self.assertIsNone(info.cpu_group)
        self.assertIsNone(info.device_group)
        self.assertIsNone(info.device_comm)
        self.assertIsNone(info.cpu_comm)

    def test_comms_fields(self):
        device_comm = MagicMock(name="device_comm")
        cpu_comm = MagicMock(name="cpu_comm")
        info = BootstrapInfo(
            rank=0,
            ranks=[0, 1],
            world_size=2,
            rank_in_group=0,
            device_comm=device_comm,
            cpu_comm=cpu_comm,
        )
        self.assertIs(info.device_comm, device_comm)
        self.assertIs(info.cpu_comm, cpu_comm)
        self.assertIsNone(info.cpu_group)
        self.assertIsNone(info.device_group)


class TorchcommsBootstrapTest(unittest.TestCase):
    """Tests for TorchcommsBootstrap (hybrid: PGs + TorchComm objects)."""

    ENV_VARS = {
        "RANK": "0",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    }

    def _make_mock_torchcomms(self):
        """Create a mock torchcomms module with new_comm."""
        mock_mod = MagicMock()
        mock_world_comm = MagicMock(name="world_comm")
        mock_sub_comm = MagicMock(name="sub_comm")
        mock_world_comm.split.return_value = mock_sub_comm
        mock_mod.new_comm.return_value = mock_world_comm
        return mock_mod, mock_world_comm, mock_sub_comm

    def _make_pg_info(self, global_rank, group_ranks):
        """Create a BootstrapInfo as ProcessGroupBootstrap would return."""
        for ranks in group_ranks:
            if global_rank in ranks:
                return BootstrapInfo(
                    rank=global_rank,
                    ranks=ranks,
                    world_size=len(ranks),
                    rank_in_group=ranks.index(global_rank),
                    cpu_group=MagicMock(name="cpu_pg"),
                    device_group=MagicMock(name="device_pg"),
                )
        raise AssertionError(f"global_rank {global_rank} not in any group")

    @patch.dict("os.environ", ENV_VARS)
    @patch("vllm.distributed.bootstrap.torch.distributed.PrefixStore")
    @patch("vllm.distributed.bootstrap.torch.distributed.TCPStore")
    @patch.object(ProcessGroupBootstrap, "create_group")
    def test_single_group_creates_pgs_and_comms(
        self, mock_pg_create, mock_tcp_store, mock_prefix_store
    ):
        mock_torchcomms, mock_world_comm, mock_sub_comm = (
            self._make_mock_torchcomms()
        )
        mock_tcp_store.return_value = MagicMock()
        pg_info = self._make_pg_info(0, [[0, 1]])
        mock_pg_create.return_value = pg_info

        with patch.dict(sys.modules, {"torchcomms": mock_torchcomms}):
            bootstrap = TorchcommsBootstrap()
            info = bootstrap.create_group(
                group_ranks=[[0, 1]],
                global_rank=0,
                backend="nccl",
            )

        # ProcessGroups present.
        self.assertIs(info.cpu_group, pg_info.cpu_group)
        self.assertIs(info.device_group, pg_info.device_group)
        # TorchComm objects present.
        self.assertIs(info.device_comm, mock_sub_comm)
        self.assertIs(info.cpu_comm, mock_sub_comm)
        # Metadata matches.
        self.assertEqual(info.rank, 0)
        self.assertEqual(info.ranks, [0, 1])
        self.assertEqual(info.world_size, 2)
        self.assertEqual(info.rank_in_group, 0)

    @patch.dict("os.environ", ENV_VARS)
    @patch("vllm.distributed.bootstrap.torch.distributed.PrefixStore")
    @patch("vllm.distributed.bootstrap.torch.distributed.TCPStore")
    @patch.object(ProcessGroupBootstrap, "create_group")
    def test_multiple_groups_selects_correct(
        self, mock_pg_create, mock_tcp_store, mock_prefix_store
    ):
        mock_torchcomms, mock_world_comm, _ = self._make_mock_torchcomms()
        device_sub = MagicMock(name="device_sub")
        cpu_sub = MagicMock(name="cpu_sub")
        mock_world_comm.split.side_effect = [device_sub, cpu_sub]
        mock_tcp_store.return_value = MagicMock()
        pg_info = self._make_pg_info(3, [[0, 1], [2, 3]])
        mock_pg_create.return_value = pg_info

        env = dict(self.ENV_VARS, RANK="3")
        with patch.dict("os.environ", env), \
             patch.dict(sys.modules, {"torchcomms": mock_torchcomms}):
            bootstrap = TorchcommsBootstrap()
            info = bootstrap.create_group(
                group_ranks=[[0, 1], [2, 3]],
                global_rank=3,
                backend="nccl",
            )

        self.assertEqual(info.rank, 3)
        self.assertEqual(info.ranks, [2, 3])
        self.assertEqual(info.world_size, 2)
        self.assertEqual(info.rank_in_group, 1)
        self.assertIs(info.device_comm, device_sub)
        self.assertIs(info.cpu_comm, cpu_sub)
        self.assertIs(info.cpu_group, pg_info.cpu_group)
        self.assertIs(info.device_group, pg_info.device_group)

    @patch.dict("os.environ", ENV_VARS)
    @patch("vllm.distributed.bootstrap.torch.distributed.PrefixStore")
    @patch("vllm.distributed.bootstrap.torch.distributed.TCPStore")
    @patch.object(ProcessGroupBootstrap, "create_group")
    def test_rank_not_found_raises(
        self, mock_pg_create, mock_tcp_store, mock_prefix_store
    ):
        mock_torchcomms, _, _ = self._make_mock_torchcomms()
        mock_tcp_store.return_value = MagicMock()
        # PG bootstrap raises first.
        mock_pg_create.side_effect = AssertionError("not found")

        with patch.dict(sys.modules, {"torchcomms": mock_torchcomms}):
            bootstrap = TorchcommsBootstrap()
            with self.assertRaises(AssertionError):
                bootstrap.create_group(
                    group_ranks=[[0, 1]],
                    global_rank=5,
                    backend="nccl",
                )

    @patch.dict("os.environ", ENV_VARS)
    @patch("vllm.distributed.bootstrap.torch.distributed.PrefixStore")
    @patch("vllm.distributed.bootstrap.torch.distributed.TCPStore")
    @patch.object(ProcessGroupBootstrap, "create_group")
    def test_world_comms_created_only_once(
        self, mock_pg_create, mock_tcp_store, mock_prefix_store
    ):
        mock_torchcomms, _, _ = self._make_mock_torchcomms()
        mock_tcp_store.return_value = MagicMock()

        with patch.dict(sys.modules, {"torchcomms": mock_torchcomms}):
            bootstrap = TorchcommsBootstrap()
            mock_pg_create.return_value = self._make_pg_info(
                0, [[0, 1, 2, 3]]
            )
            bootstrap.create_group(
                group_ranks=[[0, 1, 2, 3]],
                global_rank=0,
                backend="nccl",
            )
            mock_pg_create.return_value = self._make_pg_info(
                0, [[0, 1], [2, 3]]
            )
            bootstrap.create_group(
                group_ranks=[[0, 1], [2, 3]],
                global_rank=0,
                backend="nccl",
            )

        # new_comm called exactly twice (device + cpu), not four times.
        self.assertEqual(mock_torchcomms.new_comm.call_count, 2)

    @patch.dict("os.environ", ENV_VARS)
    @patch("vllm.distributed.bootstrap.torch.distributed.PrefixStore")
    @patch("vllm.distributed.bootstrap.torch.distributed.TCPStore")
    @patch.object(ProcessGroupBootstrap, "create_group")
    def test_store_from_env_vars(
        self, mock_pg_create, mock_tcp_store, mock_prefix_store
    ):
        mock_torchcomms, _, _ = self._make_mock_torchcomms()
        mock_tcp_store.return_value = MagicMock()
        mock_pg_create.return_value = self._make_pg_info(
            0, [[0, 1, 2, 3]]
        )

        with patch.dict(sys.modules, {"torchcomms": mock_torchcomms}):
            bootstrap = TorchcommsBootstrap()
            bootstrap.create_group(
                group_ranks=[[0, 1, 2, 3]],
                global_rank=0,
                backend="nccl",
            )

        mock_tcp_store.assert_called_once_with(
            host_name="127.0.0.1",
            port=29500,
            world_size=4,
            is_master=True,
            timeout=bootstrap._timeout,
        )

    @patch("vllm.distributed.bootstrap.torch.distributed.PrefixStore")
    @patch.object(ProcessGroupBootstrap, "create_group")
    def test_provided_store_used(self, mock_pg_create, mock_prefix_store):
        mock_torchcomms, _, _ = self._make_mock_torchcomms()
        custom_store = MagicMock(name="custom_store")
        mock_pg_create.return_value = self._make_pg_info(
            0, [[0, 1, 2, 3]]
        )

        env = dict(self.ENV_VARS)
        with patch.dict("os.environ", env), \
             patch.dict(sys.modules, {"torchcomms": mock_torchcomms}):
            bootstrap = TorchcommsBootstrap(store=custom_store)
            bootstrap.create_group(
                group_ranks=[[0, 1, 2, 3]],
                global_rank=0,
                backend="nccl",
            )

        # PrefixStore should wrap the custom store, not a TCPStore.
        for call in mock_prefix_store.call_args_list:
            self.assertIs(call[0][1], custom_store)


class TorchCommDeviceCommunicatorTest(unittest.TestCase):
    """Unit tests for TorchCommDeviceCommunicator with mocked TorchComm."""

    # Mock torchcomms module so tests work without a real torchcomms install.
    MOCK_TORCHCOMMS = MagicMock()

    def _make_communicator(self, world_size=2):
        """Create a TorchCommDeviceCommunicator with mocked dependencies."""
        import torch.distributed as dist

        mock_comm = MagicMock(name="device_comm")
        cpu_group = MagicMock(name="cpu_group")
        ranks = list(range(world_size))

        with patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "get_world_size", return_value=world_size), \
             patch.object(dist, "get_process_group_ranks", return_value=ranks), \
             patch.object(dist, "get_group_rank", return_value=0), \
             patch("torch.distributed.distributed_c10d._world") as mock_world, \
             patch("vllm.config.get_current_vllm_config_or_none", return_value=None), \
             patch.dict(sys.modules, {"torchcomms": self.MOCK_TORCHCOMMS}):
            mock_world.pg_map = {cpu_group: ("gloo", None)}

            from vllm.distributed.device_communicators.torchcomm_communicator import (
                TorchCommDeviceCommunicator,
            )

            communicator = TorchCommDeviceCommunicator(
                cpu_group=cpu_group,
                device=torch.device("cpu"),
                device_group=MagicMock(name="device_group"),
                device_comm=mock_comm,
                unique_name="test:0",
            )
        return communicator, mock_comm

    def test_all_reduce(self):
        with patch.dict(sys.modules, {"torchcomms": self.MOCK_TORCHCOMMS}):
            communicator, mock_comm = self._make_communicator()
            tensor = torch.ones(4)
            result = communicator.all_reduce(tensor)
            mock_comm.all_reduce.assert_called_once()
            self.assertIs(result, tensor)

    def test_broadcast(self):
        with patch.dict(sys.modules, {"torchcomms": self.MOCK_TORCHCOMMS}):
            communicator, mock_comm = self._make_communicator()
            tensor = torch.ones(4)
            result = communicator.broadcast(tensor, src=0)
            mock_comm.broadcast.assert_called_once_with(
                tensor, 0, async_op=False
            )
            self.assertIs(result, tensor)

    def test_broadcast_single_rank_noop(self):
        """broadcast is a no-op when world_size == 1."""
        with patch.dict(sys.modules, {"torchcomms": self.MOCK_TORCHCOMMS}):
            communicator, mock_comm = self._make_communicator(world_size=1)
            tensor = torch.ones(4)
            result = communicator.broadcast(tensor, src=0)
            mock_comm.broadcast.assert_not_called()
            self.assertIs(result, tensor)

    def test_send(self):
        with patch.dict(sys.modules, {"torchcomms": self.MOCK_TORCHCOMMS}):
            communicator, mock_comm = self._make_communicator()
            tensor = torch.ones(4)
            communicator.send(tensor, dst=1)
            mock_comm.send.assert_called_once_with(
                tensor, 1, async_op=False
            )

    def test_recv(self):
        with patch.dict(sys.modules, {"torchcomms": self.MOCK_TORCHCOMMS}):
            communicator, mock_comm = self._make_communicator()
            result = communicator.recv(torch.Size([4]), torch.float32, src=1)
            mock_comm.recv.assert_called_once()
            self.assertEqual(result.shape, torch.Size([4]))

    def test_requires_device_comm(self):
        """Constructor raises if device_comm is None."""
        import torch.distributed as dist

        cpu_group = MagicMock()
        with patch.object(dist, "get_rank", return_value=0), \
             patch.object(dist, "get_world_size", return_value=2), \
             patch.object(dist, "get_process_group_ranks", return_value=[0, 1]), \
             patch.object(dist, "get_group_rank", return_value=0), \
             patch("torch.distributed.distributed_c10d._world") as mock_world, \
             patch("vllm.config.get_current_vllm_config_or_none", return_value=None), \
             patch.dict(sys.modules, {"torchcomms": self.MOCK_TORCHCOMMS}):
            mock_world.pg_map = {cpu_group: ("gloo", None)}

            from vllm.distributed.device_communicators.torchcomm_communicator import (
                TorchCommDeviceCommunicator,
            )

            with self.assertRaises(AssertionError):
                TorchCommDeviceCommunicator(
                    cpu_group=cpu_group,
                    device=torch.device("cpu"),
                    device_comm=None,
                    unique_name="test:0",
                )
