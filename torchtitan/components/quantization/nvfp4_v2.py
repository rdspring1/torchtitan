# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 *V2* quantization converter.

Swaps dense ``Linear.Config`` nodes for :class:`NVFP4LinearV2Linear`, which keeps
a bf16 ``weight`` and quantizes activations, weights, and gradients to NVFP4 on
the fly via the monorepo's OAI-Triton ``NVFP4LinearV2`` autograd function:
four-over-six adaptive scaling on the forward GEMM and MS-EDEN + RHT-128 on the
two backward GEMMs (NVIDIA Blackwell / sm_100+, CUDA only).

Like :class:`NVFP4LinearConverter`, this is a pure leaf swap: it inherits the
model's stock colwise/rowwise sharding and changes only the GEMM. Unlike the
torchao path it does *not* run the GEMM inside ``spmd.local_map``; ``forward``
unwraps DTensor operands to local shards and runs the Triton op directly.

The low-level V2 kernels live in the monorepo ``kernels`` package
(``cutile/nvfp4_v2_triton``), which is not a torchtitan dependency. It must be on
``PYTHONPATH`` (the ``run_titan.py`` launcher adds it) for the Module to import;
otherwise :data:`NVFP4LinearV2Linear` is ``None`` and the converter raises with a
pointer to that requirement -- mirroring how :mod:`nvfp4` degrades without torchao.

Tensor parallelism: the V2 fp4 collectives (``nvfp4_{col,row}_parallel_linear_v2``)
are only reached when ``process_group`` is set, which stock ``parallelize_llama``
never does. So the converter supports single-GPU and FSDP-only; the TP branch in
``forward`` is retained as latent capability for a future ParallelStyle wiring.
"""

import hashlib
from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import DTensor

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

# NVFP4LinearV2 requires each GEMM dim (M, K, N) divisible by 128.
_NVFP4_BLOCK = 128

_TP_STYLE_COLWISE = "colwise"
_TP_STYLE_ROWWISE = "rowwise"


class NVFP4RHTCadenceManager:
    """Deterministically refresh NVFP4 RHT signs without replacing buffers."""

    def __init__(self, model_parts: list[torch.nn.Module], seed: int):
        self.seed = seed
        self.modules = [
            (
                ".".join(part for part in fqn.split(".") if part != "_orig_mod")
                or "<root>",
                module,
            )
            for model_part in model_parts
            for fqn, module in model_part.named_modules()
            if hasattr(module, "_wgrad_rht") and hasattr(module, "_dgrad_rht")
        ]

    def _refresh(
        self, *, lane: str, step: int, microbatch: int, buffer_name: str
    ) -> None:
        for fqn, module in self.modules:
            key = f"{self.seed}:{step}:{microbatch}:{lane}:{fqn}".encode()
            sample_seed = int.from_bytes(
                hashlib.blake2b(key, digest_size=8).digest(), "little"
            )
            generator = torch.Generator(device="cpu").manual_seed(sample_seed)
            signs = torch.randint(
                0, 2, (_NVFP4_BLOCK,), dtype=torch.int8, generator=generator
            ).mul_(2).sub_(1)
            target = getattr(module, buffer_name)
            if isinstance(target, DTensor):
                target = target.to_local()
            target.copy_(signs.to(target.device))

    def refresh_dgrad(self, step: int) -> None:
        self._refresh(
            lane="dgrad", step=step, microbatch=0, buffer_name="_dgrad_rht"
        )

    def refresh_wgrad(self, step: int, microbatch: int) -> None:
        self._refresh(
            lane="wgrad",
            step=step,
            microbatch=microbatch,
            buffer_name="_wgrad_rht",
        )


try:
    from kernels import (
        NVFP4LinearV2,
        nvfp4_col_parallel_linear_v2,
        nvfp4_row_parallel_linear_v2,
        sign_vector_for,
    )

    class NVFP4LinearV2Linear(Linear):
        """NVFP4 V2 Linear satisfying torchtitan's Module protocol.

        Inherits torchtitan's ``Linear`` (a flat ``nn.Linear`` + ``Module`` leaf)
        so the bf16 ``weight`` is initialized/sharded by the stock path. The NVFP4
        runtime buffers start as ``None`` and are materialized in
        ``_init_self_buffers``. Ported from the retired
        ``nvfp4_linear_v2_override.NVFP4LinearV2Linear``.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            # No new fields: extends Linear.Config so the converter preserves
            # in_features/out_features/bias/param_init verbatim.

            def __post_init__(self) -> None:
                # NVFP4LinearV2's Triton kernels need every GEMM dim to be a
                # multiple of 128. in_features / out_features are known at
                # config-build time (the TP degree is not), so reject the model-dim
                # violations up front here; the kernel itself raises on the per-rank
                # local dims once TP has sharded the weight.
                for name in ("in_features", "out_features"):
                    value = getattr(self, name)
                    if value % _NVFP4_BLOCK:
                        raise ValueError(
                            f"NVFP4 requires {name} divisible by {_NVFP4_BLOCK}; "
                            f"got {name}={value}. NVFP4 cannot quantize this Linear "
                            "(e.g. the LM head); exclude it from the converter fqns."
                        )

        def __init__(self, config: "NVFP4LinearV2Linear.Config"):
            super().__init__(config)
            # Persistent for native V2 checkpoint/resume. Start as None so
            # to_empty() leaves them alone; materialized in _init_self_buffers().
            self.register_buffer("_wgrad_rht", None, persistent=True)
            self.register_buffer("_dgrad_rht", None, persistent=True)
            self.register_buffer("_row_seed", None, persistent=True)
            self.register_buffer("_col_seed", None, persistent=True)
            # Tensor-parallel wiring: set by NVFP4{Colwise,Rowwise}ParallelV2._apply.
            # process_group=None keeps forward on the single-GPU / FSDP path.
            self.process_group = None
            self.world_size = 1
            self.tensor_parallel_style = None

        def _init_self_buffers(
            self, *, buffer_device: torch.device | None = None
        ) -> None:
            dev = buffer_device or self.weight.device
            initial_rht = sign_vector_for(_NVFP4_BLOCK).to(
                device=dev, dtype=torch.int8
            )
            self._wgrad_rht = initial_rht.clone()
            self._dgrad_rht = initial_rht.clone()
            # Fixed per-module SR seeds; offsets are drawn fresh each forward.
            self._row_seed = torch.randint(
                -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
            )
            self._col_seed = torch.randint(
                -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # weight and input are plain local tensors: FSDP all-gathers the weight
            # before forward, and the TP ParallelStyle hooks pass local shards
            # through (self.weight is a DTensor under TP -- take its local shard).
            weight = self.weight
            if isinstance(weight, DTensor):
                weight = weight.to_local()
            if weight.dtype != torch.bfloat16:
                weight = weight.to(torch.bfloat16)
            # NVFP4LinearV2 requires a 2D bf16 input with M, K divisible by 128.
            x = input.to_local() if isinstance(input, DTensor) else input
            x_2d = x.reshape(-1, x.shape[-1])
            if x_2d.dtype != torch.bfloat16:
                x_2d = x_2d.to(torch.bfloat16)
            wgrad_rht = (
                self._wgrad_rht.to_local()
                if isinstance(self._wgrad_rht, DTensor)
                else self._wgrad_rht
            )
            dgrad_rht = (
                self._dgrad_rht.to_local()
                if isinstance(self._dgrad_rht, DTensor)
                else self._dgrad_rht
            )

            if self.process_group is not None:
                # Tensor-parallel: fp4 collectives run inside the autograd function.
                # The wrappers draw fresh SR offsets per forward from the fixed seeds.
                if self.tensor_parallel_style == _TP_STYLE_COLWISE:
                    out_2d = nvfp4_col_parallel_linear_v2(
                        x_2d,
                        weight,
                        bias=None,
                        row_seed=self._row_seed,
                        col_seed=self._col_seed,
                        tp_group=self.process_group,
                        world_size=self.world_size,
                        wgrad_rht=wgrad_rht,
                        dgrad_rht=dgrad_rht,
                    )
                elif self.tensor_parallel_style == _TP_STYLE_ROWWISE:
                    out_2d = nvfp4_row_parallel_linear_v2(
                        x_2d,
                        weight,
                        bias=None,
                        row_seed=self._row_seed,
                        col_seed=self._col_seed,
                        tp_group=self.process_group,
                        world_size=self.world_size,
                        wgrad_rht=wgrad_rht,
                        dgrad_rht=dgrad_rht,
                    )
                else:
                    raise ValueError(
                        f"process_group set but tensor_parallel_style is "
                        f"{self.tensor_parallel_style!r}; expected 'colwise' or "
                        "'rowwise'"
                    )
            else:
                # Single-GPU / FSDP-only path. Fresh offset bases per forward ->
                # per-step SR variation in backward (mirrors drawing offsets each step).
                row_offset = torch.randint(
                    -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=x_2d.device
                )
                col_offset = torch.randint(
                    -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=x_2d.device
                )
                out_2d = NVFP4LinearV2.apply(
                    x_2d,
                    weight,
                    wgrad_rht,
                    dgrad_rht,
                    self._row_seed,
                    row_offset,
                    self._col_seed,
                    col_offset,
                )
            return out_2d.reshape(*x.shape[:-1], out_2d.shape[-1])

except ImportError:
    NVFP4LinearV2Linear = None


class NVFP4LinearV2Converter(QuantizationConverter):
    """Replace matching Linear.Config with NVFP4LinearV2Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply NVFP4 V2 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted. The LM head must be excluded
        (NVFP4 requires each GEMM dim divisible by 128; vocab is not).
        """

    def __init__(self, config: Config):
        self.config = config

        if NVFP4LinearV2Linear is None:
            raise ImportError(
                "The NVFP4 V2 Triton kernels are not importable. They live in the "
                "monorepo `kernels` package (cutile/nvfp4_v2_triton); put that "
                "directory (NVFP4_V2_ROOT) on PYTHONPATH -- the run_titan.py launcher "
                "does this automatically."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

    def convert(self, model_config):
        assert NVFP4LinearV2Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = NVFP4LinearV2Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4LinearV2Linear")
        return model_config
