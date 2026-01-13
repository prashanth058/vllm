"""Tests for configuration classes."""

import pytest

from vllm_train.config import DPOConfig, LoRATrainConfig, RLHFConfig


class TestLoRATrainConfig:
    """Tests for LoRATrainConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoRATrainConfig()

        assert config.rank == 8
        assert config.alpha == 16
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.num_epochs == 3
        assert config.bf16 is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoRATrainConfig(
            rank=32,
            alpha=64,
            learning_rate=1e-4,
            num_epochs=5,
        )

        assert config.rank == 32
        assert config.alpha == 64
        assert config.learning_rate == 1e-4
        assert config.num_epochs == 5

    def test_target_modules_default(self):
        """Test default target modules."""
        config = LoRATrainConfig()

        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules

    def test_to_peft_config(self):
        """Test conversion to PEFT config."""
        config = LoRATrainConfig(rank=16, alpha=32, dropout=0.1)

        # This will fail if PEFT is not installed, which is expected in CI
        pytest.importorskip("peft")

        peft_config = config.to_peft_config()

        assert peft_config.r == 16
        assert peft_config.lora_alpha == 32
        assert peft_config.lora_dropout == 0.1


class TestRLHFConfig:
    """Tests for RLHFConfig."""

    def test_default_values(self):
        """Test default RLHF configuration."""
        config = RLHFConfig()

        assert config.kl_coef == 0.1
        assert config.clip_range == 0.2
        assert config.ppo_epochs == 4
        assert config.temperature == 0.7

    def test_nested_lora_config(self):
        """Test nested LoRA configuration."""
        config = RLHFConfig(
            lora_config=LoRATrainConfig(rank=32),
            kl_coef=0.05,
        )

        assert config.lora_config.rank == 32
        assert config.kl_coef == 0.05


class TestDPOConfig:
    """Tests for DPOConfig."""

    def test_default_values(self):
        """Test default DPO configuration."""
        config = DPOConfig()

        assert config.beta == 0.1
        assert config.label_smoothing == 0.0
        assert config.loss_type == "sigmoid"
        assert config.reference_free is False

    def test_custom_loss_type(self):
        """Test custom loss type."""
        config = DPOConfig(loss_type="hinge", beta=0.2)

        assert config.loss_type == "hinge"
        assert config.beta == 0.2
