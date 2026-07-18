# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the NVFP4 DeepSeek V3 MLA attention override.

Covers config-time targeting (Attention.Config -> the wq/wkv_b/wo projections
become MLANVFP4Linear.Config while wkv_a and the norms stay bf16, and non-128
dims are skipped) and the TP>1 assertion (MLA NVFP4 is TP=1 only). End-to-end
forward/backward on Blackwell is covered by the run_titan `--nvfp4-target linear`
smoke, which exercises the full model.
"""

import unittest

from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.overrides.nvfp4_linear import (
    MLANVFP4Linear,
    NVFP4Linear,
    nvfp4_mla_attention,
)


def _debugmodel_attention_config():
    """A debugmodel MLA Attention.Config with child sharding_config populated
    (as the Trainer does via update_from_config before applying overrides)."""
    trainer_cfg = deepseek_v3_debugmodel()
    model_cfg = trainer_cfg.model_spec.model
    model_cfg.update_from_config(config=trainer_cfg)
    return model_cfg.layers[0].attention


class _FakeParallelDims:
    def __init__(self, tp_enabled: bool):
        self.tp_enabled = tp_enabled
        self.spmd_backend = "default"


class TestNVFP4MLAAttentionTargeting(unittest.TestCase):
    """Config-time override targeting (hardware-independent)."""

    def test_converts_wq_wkvb_wo_only(self):
        attn = _debugmodel_attention_config()
        out = nvfp4_mla_attention(attn)

        # q_lora_rank == 0 for the debugmodel -> single wq.
        self.assertIsInstance(out.wq, MLANVFP4Linear.Config)
        self.assertEqual(out.wq.tensor_parallel_style, "colwise")
        self.assertIsInstance(out.wkv_b, MLANVFP4Linear.Config)
        self.assertEqual(out.wkv_b.tensor_parallel_style, "colwise")
        self.assertIsInstance(out.wo, MLANVFP4Linear.Config)
        self.assertEqual(out.wo.tensor_parallel_style, "rowwise")

        # wkv_a (576 not 128-aligned + Replicate) and the norms stay bf16, passed
        # through unchanged (same object).
        self.assertNotIsInstance(out.wkv_a, NVFP4Linear.Config)
        self.assertIs(out.wkv_a, attn.wkv_a)
        self.assertIs(out.q_norm, attn.q_norm)
        self.assertIs(out.kv_norm, attn.kv_norm)

    def test_skips_non_128_aligned_projection(self):
        attn = _debugmodel_attention_config()
        attn.wkv_b.out_features = 4095  # no longer a multiple of 128
        out = nvfp4_mla_attention(attn)
        self.assertNotIsInstance(out.wkv_b, NVFP4Linear.Config)
        self.assertIs(out.wkv_b, attn.wkv_b)
        # wq/wo are still aligned and still converted.
        self.assertIsInstance(out.wq, MLANVFP4Linear.Config)
        self.assertIsInstance(out.wo, MLANVFP4Linear.Config)


class TestMLANVFP4LinearValidate(unittest.TestCase):
    """MLA NVFP4 is TP=1 only; _validate must assert loudly under TP>1."""

    def _linear(self):
        return MLANVFP4Linear.Config(
            in_features=512, out_features=4096, tensor_parallel_style="colwise"
        ).build()

    def test_raises_under_tp(self):
        lin = self._linear()
        with self.assertRaisesRegex(ValueError, "tensor_parallel_degree=1"):
            lin._validate(_FakeParallelDims(tp_enabled=True))

    def test_ok_at_tp1(self):
        # TP=1: the MLA guard does not fire; the inherited 128-block check passes
        # for the 128-aligned (512, 4096) dims.
        lin = self._linear()
        lin._validate(_FakeParallelDims(tp_enabled=False))


if __name__ == "__main__":
    unittest.main()
