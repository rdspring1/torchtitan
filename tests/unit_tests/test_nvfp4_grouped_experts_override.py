# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the NVFP4 grouped-experts MoE override.

Covers override targeting (GroupedExperts.Config -> NVFP4GroupedExperts.Config,
the TorchAO token-dispatcher swap, the 128-alignment skip, and the all-to-all
dispatcher requirement), single-GPU expert numerics (Blackwell), and a
TP x EP integration test that applies both the nvfp4_linear and
nvfp4_grouped_experts overrides to a DeepSeek V3 debugmodel and runs
forward/backward across the cartesian product TP=[1,2] x EP=[1,2] (two
Blackwell GPUs).
"""

import tempfile
import unittest

import torch
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config import apply_overrides, clear_overrides, OverrideConfig
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan.overrides.nvfp4_grouped_experts import (
    NVFP4GroupedExperts,
    nvfp4_grouped_experts,
)

_DIM = 256
_HIDDEN = 256  # _DIM, _HIDDEN divisible by 128
_E = 8  # divisible by EP=2
_TOP_K = 1
_LINEAR_MODULE = "torchtitan.overrides.nvfp4_linear"
_EXPERTS_MODULE = "torchtitan.overrides.nvfp4_grouped_experts"


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _blackwell_tp2() -> bool:
    return torch.cuda.device_count() >= 2 and all(
        torch.cuda.get_device_capability(i)[0] >= 10 for i in range(2)
    )


def _grouped_experts_config(
    dispatcher=None, *, dim: int = _DIM, hidden_dim: int = _HIDDEN
) -> GroupedExperts.Config:
    if dispatcher is None:
        dispatcher = AllToAllTokenDispatcher.Config(num_experts=_E, top_k=_TOP_K)
    return GroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=_E,
        token_dispatcher=dispatcher,
    )


class TestNVFP4GroupedExpertsTargeting(unittest.TestCase):
    """Config-time override targeting (hardware-independent)."""

    def test_replaces_grouped_experts_and_swaps_dispatcher(self):
        out = nvfp4_grouped_experts(_grouped_experts_config())
        self.assertIsInstance(out, NVFP4GroupedExperts.Config)
        self.assertIsInstance(out.token_dispatcher, TorchAOTokenDispatcher.Config)
        self.assertEqual(out.token_dispatcher.pad_multiple, 128)
        self.assertEqual(out.token_dispatcher.num_experts, _E)
        self.assertEqual(out.token_dispatcher.top_k, _TOP_K)
        self.assertFalse(out.pad_token_groups_for_grouped_mm)
        # Expert dims are carried through by derive().
        self.assertEqual((out.dim, out.hidden_dim, out.num_experts), (_DIM, _HIDDEN, _E))

    def test_skips_when_dims_not_128_aligned(self):
        cfg = _grouped_experts_config(dim=300)
        self.assertIs(nvfp4_grouped_experts(cfg), cfg)
        cfg = _grouped_experts_config(hidden_dim=300)
        self.assertIs(nvfp4_grouped_experts(cfg), cfg)

    def test_requires_alltoall_dispatcher(self):
        cfg = _grouped_experts_config(
            LocalTokenDispatcher.Config(num_experts=_E, top_k=_TOP_K)
        )
        with self.assertRaisesRegex(ValueError, "all-to-all"):
            nvfp4_grouped_experts(cfg)


@unittest.skipUnless(_blackwell(), "NVFP4 Triton kernels require Blackwell (sm_100+)")
class TestNVFP4GroupedExpertsNumerics(unittest.TestCase):
    """Single-GPU (EP=1) expert numerics against the stock bf16 grouped MM."""

    def test_forward_close_to_reference_and_backward_finite(self):
        torch.manual_seed(0)
        dispatcher = LocalTokenDispatcher.Config(num_experts=_E, top_k=_TOP_K)
        stock = _grouped_experts_config(dispatcher).build().cuda()
        nvfp4 = (
            NVFP4GroupedExperts.Config(
                dim=_DIM,
                hidden_dim=_HIDDEN,
                num_experts=_E,
                token_dispatcher=dispatcher,
            )
            .build()
            .cuda()
        )
        # Materialize sr_seed (and default weights) before copying shared weights.
        nvfp4._init_self_buffers(buffer_device=torch.device("cuda"))

        with torch.no_grad():
            w1 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w3 = 0.1 * torch.randn(_E, _HIDDEN, _DIM, device="cuda")
            w2 = 0.1 * torch.randn(_E, _DIM, _HIDDEN, device="cuda")
            for m in (stock, nvfp4):
                m.w1_EFD.copy_(w1)
                m.w3_EFD.copy_(w3)
                m.w2_EDF.copy_(w2)

        # 128 tokens per expert: each group is 128-aligned and offsets_E[-1] == rows,
        # so there is no dispatcher padding tail to account for.
        num_tokens = torch.full((_E,), 128, device="cuda", dtype=torch.int64)
        rows = int(num_tokens.sum())
        x = torch.randn(rows, _DIM, device="cuda", dtype=torch.bfloat16)
        x_ref = x.detach().clone().requires_grad_()
        x_q = x.detach().clone().requires_grad_()

        out_ref = stock._experts_forward(x_ref, num_tokens)
        out_q = nvfp4._experts_forward(x_q, num_tokens)

        from torchao.quantization.utils import compute_error

        # Three chained NVFP4 GEMMs (gate -> silu*up -> down) accumulate more
        # 4-bit quantization error than a single linear (~15 dB), landing near
        # 12 dB against the bf16 reference; 10 dB leaves margin while still
        # failing hard on broken math (NaN / wrong contraction give ~0 dB).
        sqnr = compute_error(out_ref.float(), out_q.float())
        self.assertGreaterEqual(sqnr.item(), 10.0)

        out_q.sum().backward()
        self.assertTrue(torch.isfinite(x_q.grad).all())


@unittest.skipUnless(_blackwell_tp2(), "NVFP4 TP/EP tests require two Blackwell GPUs")
class TestNVFP4DSV3Integration(DTensorTestBase):
    """Both overrides (nvfp4_linear on the dense FFN linears, nvfp4_grouped_experts
    on the MoE) on a DeepSeek V3 debugmodel, across TP=[1,2] x EP=[1,2].

    All four combos fit world_size=2: dp_replicate*dp_shard*cp*tp*pp == world_size
    is the only hard constraint (ep is orthogonal, borrowing ranks from
    dp_shard*cp*tp), so dp_shard = 2 // tp fills the mesh.
    """

    @property
    def world_size(self):
        return 2

    @with_comms
    def test_tp_ep_cartesian_forward_backward(self):
        for tp, ep in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            with self.subTest(tp=tp, ep=ep):
                self._run_combo(tp, ep)

    def _run_combo(self, tp: int, ep: int) -> None:
        clear_overrides()
        trainer_cfg = deepseek_v3_debugmodel()
        trainer_cfg.parallelism = ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=self.world_size // tp,
            tensor_parallel_degree=tp,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=ep,
            enable_sequence_parallel=(tp > 1),
        )
        # Keep the forward eager so the numeric path is exercised directly.
        trainer_cfg.compile.enable = False

        model_cfg = trainer_cfg.model_spec.model
        model_cfg.update_from_config(config=trainer_cfg)
        apply_overrides(
            OverrideConfig(imports=[_LINEAR_MODULE, _EXPERTS_MODULE]), trainer_cfg
        )

        model = model_cfg.build().to(self.device_type)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=self.world_size // tp,
            cp=1,
            tp=tp,
            pp=1,
            ep=ep,
            world_size=self.world_size,
        )
        with tempfile.TemporaryDirectory() as dump_folder:
            parallelize_deepseekv3(
                model,
                parallel_dims=parallel_dims,
                training=trainer_cfg.training,
                parallelism=trainer_cfg.parallelism,
                compile_config=trainer_cfg.compile,
                ac_config=trainer_cfg.activation_checkpoint,
                dump_folder=dump_folder,
            )
        model.init_states(buffer_device=torch.device(self.device_type))

        B, L = 2, 256  # L divisible by tp and by tp*(cp*2)
        vocab = model_cfg.vocab_size
        # Identical input on every rank: TP ranks must agree (they shard the
        # sequence internally); replicating it is also valid for DP/EP ranks.
        torch.manual_seed(1234)
        tokens = torch.randint(0, vocab, (B, L), device=self.device_type)
        positions = torch.arange(L, device=self.device_type).repeat(B, 1)
        attention_masks = model.get_attention_masks(positions)

        out = model(tokens, attention_masks=attention_masks, positions=positions)
        self.assertEqual(tuple(out.shape), (B, L, vocab))
        out_local = out.to_local() if isinstance(out, DTensor) else out
        self.assertTrue(torch.isfinite(out_local).all())

        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        for g in grads:
            g_local = g.to_local() if isinstance(g, DTensor) else g
            self.assertTrue(torch.isfinite(g_local).all())
