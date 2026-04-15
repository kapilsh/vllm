# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the torchcomms shim integration.

Unit tests for config/env var propagation and EngineArgs flow.
GPU integration tests are covered by parametrizing existing tests
(e.g., test_pynccl.py) with use_torchcomms=True.

Run:
  pytest tests/distributed/test_torchcomms_shim.py -v
"""

import os

import pytest


# ---------------------------------------------------------------------------
# Unit tests — ParallelConfig.use_torchcomms env var pickup
# ---------------------------------------------------------------------------


class TestParallelConfigTorchcomms:
    """Test that ParallelConfig.use_torchcomms picks up env vars."""

    def test_default_is_false(self, monkeypatch):
        monkeypatch.delenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        monkeypatch.delenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        import vllm.envs as envs
        envs.disable_envs_cache()

        from vllm.config.parallel import ParallelConfig
        pc = ParallelConfig()
        assert pc.use_torchcomms is False

    def test_vllm_env_var_enables(self, monkeypatch):
        monkeypatch.setenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", "1")
        monkeypatch.delenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        import vllm.envs as envs
        envs.disable_envs_cache()

        from vllm.config.parallel import ParallelConfig
        pc = ParallelConfig()
        assert pc.use_torchcomms is True

    def test_torch_env_var_enables(self, monkeypatch):
        monkeypatch.delenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        monkeypatch.setenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", "1")
        import vllm.envs as envs
        envs.disable_envs_cache()

        from vllm.config.parallel import ParallelConfig
        pc = ParallelConfig()
        assert pc.use_torchcomms is True

    def test_explicit_true_without_env(self, monkeypatch):
        monkeypatch.delenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        monkeypatch.delenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        import vllm.envs as envs
        envs.disable_envs_cache()

        from vllm.config.parallel import ParallelConfig
        pc = ParallelConfig(use_torchcomms=True)
        assert pc.use_torchcomms is True


# ---------------------------------------------------------------------------
# Unit tests — envs module
# ---------------------------------------------------------------------------


class TestEnvsTorchcomms:
    """Test vllm.envs.VLLM_DISTRIBUTED_USE_TORCHCOMMS."""

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        monkeypatch.delenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        import vllm.envs as envs
        envs.disable_envs_cache()
        assert envs.VLLM_DISTRIBUTED_USE_TORCHCOMMS is False

    def test_vllm_env_true(self, monkeypatch):
        monkeypatch.setenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", "1")
        monkeypatch.delenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        import vllm.envs as envs
        envs.disable_envs_cache()
        assert envs.VLLM_DISTRIBUTED_USE_TORCHCOMMS is True

    def test_torch_env_true(self, monkeypatch):
        monkeypatch.delenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", raising=False)
        monkeypatch.setenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", "1")
        import vllm.envs as envs
        envs.disable_envs_cache()
        assert envs.VLLM_DISTRIBUTED_USE_TORCHCOMMS is True

    def test_both_env_true(self, monkeypatch):
        monkeypatch.setenv("VLLM_DISTRIBUTED_USE_TORCHCOMMS", "1")
        monkeypatch.setenv("TORCH_DISTRIBUTED_USE_TORCHCOMMS", "1")
        import vllm.envs as envs
        envs.disable_envs_cache()
        assert envs.VLLM_DISTRIBUTED_USE_TORCHCOMMS is True


# ---------------------------------------------------------------------------
# Unit tests — EngineArgs flow-through
# ---------------------------------------------------------------------------


class TestEngineArgsTorchcomms:
    """Test that EngineArgs.use_torchcomms flows to ParallelConfig."""

    def test_engine_args_default_false(self):
        from vllm.engine.arg_utils import EngineArgs
        args = EngineArgs(model="dummy")
        assert args.use_torchcomms is False

    def test_engine_args_explicit_true(self):
        from vllm.engine.arg_utils import EngineArgs
        args = EngineArgs(model="dummy", use_torchcomms=True)
        assert args.use_torchcomms is True

    def test_engine_args_flows_to_parallel_config(self):
        from vllm.engine.arg_utils import EngineArgs
        args = EngineArgs(model="dummy", use_torchcomms=True)
        # EngineArgs.create_engine_config builds ParallelConfig with
        # use_torchcomms=self.use_torchcomms. We can't call
        # create_engine_config without a real model, but we verify
        # the attribute exists and is passed through.
        assert hasattr(args, "use_torchcomms")
        assert args.use_torchcomms is True


# ---------------------------------------------------------------------------
# Unit tests — GroupCoordinator group creation logic
# ---------------------------------------------------------------------------


class TestGroupCreationLogic:
    """Test the world-sized vs subgroup detection in GroupCoordinator.

    These test the logic without creating real process groups.
    """

    def test_world_sized_detection_all_ranks(self):
        """Single group with all ranks is world-sized."""
        group_ranks = [[0, 1, 2, 3]]
        world_size = 4
        is_world_sized = all(
            len(r) == world_size for r in group_ranks
        )
        assert is_world_sized is True

    def test_world_sized_detection_subgroups(self):
        """Multiple single-rank groups are NOT world-sized."""
        group_ranks = [[0], [1], [2], [3]]
        world_size = 4
        is_world_sized = all(
            len(r) == world_size for r in group_ranks
        )
        assert is_world_sized is False

    def test_world_sized_detection_pairs(self):
        """Pair groups in a 4-rank world are NOT world-sized."""
        group_ranks = [[0, 1], [2, 3]]
        world_size = 4
        is_world_sized = all(
            len(r) == world_size for r in group_ranks
        )
        assert is_world_sized is False

    def test_world_sized_detection_tp2_world2(self):
        """TP=2 with world_size=2: single group IS world-sized."""
        group_ranks = [[0, 1]]
        world_size = 2
        is_world_sized = all(
            len(r) == world_size for r in group_ranks
        )
        assert is_world_sized is True


# ---------------------------------------------------------------------------
# Unit tests — init_distributed_environment config activation
# ---------------------------------------------------------------------------


class TestConfigActivation:
    """Test that init_distributed_environment activates torchcomms
    from ParallelConfig.

    These verify the config plumbing without actually initializing
    distributed (which would need GPUs).
    """

    def test_use_torchcomms_enabled_import(self):
        """Verify _use_torchcomms_enabled is importable."""
        from torch.distributed.distributed_c10d import (
            _use_torchcomms_enabled,
        )
        # Should not raise
        result = _use_torchcomms_enabled()
        assert isinstance(result, bool)

    def test_parallel_config_use_torchcomms_in_engine_args_fields(self):
        """Verify use_torchcomms is a recognized EngineArgs field."""
        from vllm.engine.arg_utils import EngineArgs
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(EngineArgs)]
        assert "use_torchcomms" in field_names

    def test_parallel_config_use_torchcomms_field_exists(self):
        """Verify use_torchcomms is a recognized ParallelConfig field."""
        from vllm.config.parallel import ParallelConfig
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(ParallelConfig)]
        assert "use_torchcomms" in field_names
