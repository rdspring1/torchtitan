# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: NVFP4 grouped-experts MoE path.

This swaps the stock ``GroupedExperts`` for :class:`NVFP4GroupedExperts`, which
keeps bf16 expert weights and quantizes the three grouped GEMMs (gate, up, down)
to NVFP4 on the fly using TorchAO's
``torchao.prototype.moe_training.nvfp4_training`` grouped-mm kernels. This module
only adapts TorchAO to TorchTitan's ``Module`` protocol and the override
mechanism.

Usage:

    torchtitan_train --module deepseek_v3 --config deepseek_v3_debugmodel \\
        --override.imports torchtitan.overrides.nvfp4_grouped_experts

Expert parallelism -- TorchTitan owns expert-weight sharding (the ``ep`` axis
shards the expert dim 0) and the token dispatch/combine. NVFP4 owns quantization
semantics only: the grouped GEMMs consume the local expert shard, converting the
DTensor weights to local tensors exactly as the stock ``GroupedExperts`` does.
The override also swaps the token dispatcher for ``TorchAOTokenDispatcher`` so
token groups are padded to a multiple of 128 -- the alignment TorchAO's NVFP4
grouped MM requires; the standard EP all-to-all dispatcher is a prerequisite.

The TorchAO kernels target NVIDIA Blackwell (sm_100+), CUDA, Triton, and
PyTorch 2.10+. ``torchao`` is imported lazily so this module (and its
config-time targeting tests) import without it; the factory raises a clear error
if selected when TorchAO is unavailable, and the hardware requirement is checked
in :meth:`NVFP4GroupedExperts.parallelize` (where every real run passes before
training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.utils._triton import has_triton

from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    TorchAOTokenDispatcher,
)

if TYPE_CHECKING:
    _TORCHAO_IMPORT_ERROR: ImportError | None = None
else:
    try:
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
            _to_nvfp4_then_scaled_grouped_mm,
        )
        from torchao.utils import is_sm_at_least_100, torch_version_at_least

        _TORCHAO_IMPORT_ERROR = None
    except ImportError as e:
        _TORCHAO_IMPORT_ERROR = e

__all__ = ["NVFP4GroupedExperts", "nvfp4_grouped_experts"]

# torchao's NVFP4 grouped MM pads token groups to a multiple of 128.
_NVFP4_PAD_MULTIPLE = 128

# NVFP4 passes per-module Python sign-vector tuples into the compiled block.
# Under fullgraph=True those constants plus async collectives can exceed Dynamo's
# default recompile limit before step 1; raise it once at import.
_NVFP4_RECOMPILE_LIMIT = 64
if torch._dynamo.config.recompile_limit < _NVFP4_RECOMPILE_LIMIT:
    torch._dynamo.config.recompile_limit = _NVFP4_RECOMPILE_LIMIT

RHT_SIGN_VECTOR = tuple(1 if i % 2 == 0 else -1 for i in range(16))


def _require_torchao() -> None:
    if _TORCHAO_IMPORT_ERROR is not None:
        raise ImportError(
            "nvfp4 grouped-experts override was requested but torchao's NVFP4 "
            "training prototype is not importable; install a torchao build that "
            "provides torchao.prototype.moe_training.nvfp4_training."
        ) from _TORCHAO_IMPORT_ERROR


def _assert_nvfp4_supported() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 grouped experts require CUDA")
    if not is_sm_at_least_100():
        raise RuntimeError("NVFP4 grouped experts require an SM100+ GPU")
    if not has_triton():
        raise RuntimeError("NVFP4 grouped experts require Triton")
    if not torch_version_at_least("2.10.0"):
        raise RuntimeError("NVFP4 grouped experts require PyTorch 2.10 or newer")


class NVFP4GroupedExperts(GroupedExperts):
    """GroupedExperts implementation backed by torchao NVFP4 grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pad_token_groups_for_grouped_mm: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self._pad_token_groups = config.pad_token_groups_for_grouped_mm
        self.sr_seed = None

    def parallelize(self, parallel_dims) -> None:
        # Hardware requirement is checked here (not in the override factory) so
        # config-time targeting is hardware-independent; parallelize() runs before
        # any real forward under the EP/TP paths this override targets.
        _assert_nvfp4_supported()
        super().parallelize(parallel_dims)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        self.sr_seed = torch.tensor(
            [1234],
            dtype=torch.int64,
            device=buffer_device or self.w1_EFD.device,
        )

    def _experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        # TorchAOTokenDispatcher keeps a sentinel padding tail for unpermute.
        # NVFP4 grouped MM requires the final offset to cover those rows.
        offsets_E[-1] = x_RD.shape[0]
        gate_RF = _to_nvfp4_then_scaled_grouped_mm(
            x_RD.bfloat16(),
            w1_EFD.bfloat16(),
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets_E,
            pad_token_groups_for_grouped_mm=self._pad_token_groups,
        )
        up_RF = _to_nvfp4_then_scaled_grouped_mm(
            x_RD.bfloat16(),
            w3_EFD.bfloat16(),
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets_E,
            pad_token_groups_for_grouped_mm=self._pad_token_groups,
        )
        h_RF = F.silu(gate_RF) * up_RF
        return _to_nvfp4_then_scaled_grouped_mm(
            h_RF,
            w2_EDF.bfloat16(),
            RHT_SIGN_VECTOR,
            self.sr_seed,
            offs=offsets_E,
            pad_token_groups_for_grouped_mm=self._pad_token_groups,
        ).type_as(x_RD)


def _torchao_token_dispatcher(
    cfg: GroupedExperts.Config,
) -> TorchAOTokenDispatcher.Config:
    if not isinstance(cfg.token_dispatcher, AllToAllTokenDispatcher.Config):
        raise ValueError(
            "NVFP4 grouped experts require the standard EP all-to-all token "
            "dispatcher so token groups are padded before grouped MM; got "
            f"{type(cfg.token_dispatcher).__name__}."
        )

    return TorchAOTokenDispatcher.Config(
        num_experts=cfg.token_dispatcher.num_experts,
        top_k=cfg.token_dispatcher.top_k,
        pad_multiple=_NVFP4_PAD_MULTIPLE,
    )


@override(
    "nvfp4_grouped_experts",
    target=GroupedExperts.Config,
    exact=True,
    description="Replace GroupedExperts with torchao NVFP4 grouped GEMMs (Blackwell)",
)
def nvfp4_grouped_experts(
    cfg: GroupedExperts.Config,
) -> GroupedExperts.Config:
    _require_torchao()
    if cfg.dim % 128 or cfg.hidden_dim % 128:
        return cfg
    return derive(
        cfg,
        NVFP4GroupedExperts.Config,
        token_dispatcher=_torchao_token_dispatcher(cfg),
        pad_token_groups_for_grouped_mm=False,
    )
