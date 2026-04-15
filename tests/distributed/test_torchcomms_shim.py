# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the torchcomms shim integration.

Unit tests for config/env var propagation. GPU integration tests can be run
by using existing distributed tests with use_torchcomms=True in the
ParallelConfig.

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
