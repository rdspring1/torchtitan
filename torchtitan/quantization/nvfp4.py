# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 quantized linear building block.

Swaps dense ``Linear.Config`` nodes for :class:`NVFP4Linear`, which keeps a bf16
weight and quantizes activations, weights, and gradients to NVFP4 on the fly via
TorchAO's ``nvfp4_training`` kernels (NVIDIA Blackwell / sm_100+, CUDA only).

Like :class:`MXFP8LinearConverter`, this is a pure leaf swap: it inherits the
model's stock colwise/rowwise sharding and changes only the GEMM. Under tensor
parallelism the block boundary keeps its stock bf16 collectives (all-gather /
reduce-scatter); NVFP4 does not move fp4 codes over the wire.
"""

import logging
import math
from dataclasses import dataclass, replace
from typing import cast

import spmd_types as spmd
import torch
from spmd_types import SpmdType

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module


TP = MeshAxisName.TP
logger = logging.getLogger(__name__)

# TorchAO's NVFP4 Triton kernels require each local GEMM dimension to be a
# multiple of 128.
_NVFP4_BLOCK = 128

# Fixed Random Hadamard Transform basis (the NVFP4 v1 recipe default in torchao
# and Transformer Engine). It must be identical across TP ranks -- rowwise TP
# shards the GEMM contraction dim, and the Hadamard transform only cancels
# between the two operands when both use the same sign vector. Hardcoding it
# makes every rank produce the same vector by construction (no cross-rank
# broadcast). Per-recipe dynamic sign vectors are a future extension.
_HARDCODED_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)

_NVFP4_RECIPES = ("v1", "v1_requant", "v2")
_V2_RHT_SIZE = 128
_V2_LINEAR_WGRAD_SEED = 0x1EA1
_V2_LINEAR_DGRAD_SEED = 0x1EA2

try:
    from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
        cutedsl_nvfp4_kernels_available,
        cutedsl_nvfp4_unavailable_reason,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
        _to_nvfp4_rht_rs_then_scaled_grouped_mm,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
        nvfp4_linear,
        nvfp4_matmul,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear_v2 import (
        _NVFP4LinearV1Requant,
        _NVFP4LinearV2,
        nvfp4_linear_v1_requant,
        nvfp4_linear_v2,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import NVFP4Recipe
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
        _make_rht_sign_vector,
        _rht_sign_vector_to_tuple,
        NVFP4Linear as TorchAONVFP4Linear,
    )
    from torchao.quantization.quantize_.common import KernelPreference

    _SUPPORTED_KERNEL_PREFERENCES = (
        KernelPreference.AUTO,
        KernelPreference.TRITON,
        KernelPreference.CUTEDSL,
    )

    def _to_kernel_preference(name: str) -> KernelPreference:
        try:
            pref = KernelPreference(name)
        except ValueError:
            pref = None
        if pref not in _SUPPORTED_KERNEL_PREFERENCES:
            raise ValueError(
                "NVFP4 kernel_preference must be one of "
                f"{[p.value for p in _SUPPORTED_KERNEL_PREFERENCES]}, got {name!r}"
            )
        return pref

    def _log_kernel_preference(
        what: str,
        pref: KernelPreference,
        use_fast_math: bool,
        ms_eden_fast_path: bool,
        *,
        per_op: bool,
    ) -> None:
        if pref is KernelPreference.CUTEDSL and not cutedsl_nvfp4_kernels_available():
            raise RuntimeError(
                f"{what} kernel_preference=cutedsl, but the CuteDSL runtime is "
                f"unavailable ({cutedsl_nvfp4_unavailable_reason()})."
            )
        logger.info(
            "%s kernel_preference=%s, use_fast_math=%s, ms_eden_fast_path=%s",
            what,
            pref.value,
            use_fast_math,
            ms_eden_fast_path,
        )
        if pref is KernelPreference.AUTO:
            logger.warning(
                "%s kernel_preference=auto: the backend follows the container, not "
                "the recipe (%s), and the SR streams differ between backends.",
                what,
                "resolved per op"
                if per_op
                else ("cutedsl" if cutedsl_nvfp4_kernels_available() else "triton"),
            )

    # The NVFP4 GEMM is a raw autograd Function that runs on local shards inside
    # the local SPMD region. Mark it local-safe so SPMD type checking
    # propagates through it; the region boundary declares the real
    # colwise/rowwise output type.
    spmd.register_local_autograd_function(nvfp4_matmul)
    spmd.register_local_autograd_function(_NVFP4LinearV1Requant)
    spmd.register_local_autograd_function(_NVFP4LinearV2)

    class NVFP4Linear(TorchAONVFP4Linear, Module):
        """NVFP4 Linear satisfying torchtitan's Module protocol.

        Reuses TorchAO's ``NVFP4Linear`` (weight/bias, the ``_sr_seed`` /
        ``_rht_sign_vector`` runtime buffers, RHT logic, functional forward) and
        adds torchtitan's meta-init buffer protocol and local SPMD sharding.
        ``_rht_sign_vector`` is the fixed ``_HARDCODED_SIGN_VECTOR`` (identical on
        every rank by construction) and ``_sr_seed`` is per-rank.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds NVFP4Linear."""

            kernel_preference: str = "cutedsl"
            use_fast_math: bool = True
            recipe: str = "v1"
            ms_eden_fast_path: bool = True

            def __post_init__(self) -> None:
                if self.recipe not in _NVFP4_RECIPES:
                    raise ValueError(
                        f"recipe must be one of {_NVFP4_RECIPES}, got {self.recipe!r}"
                    )
                # NVFP4's Triton kernels need every GEMM dim to be a multiple of
                # 128. in_features / out_features are known at config-build time
                # (the TP degree is not), so reject the model-dim violations up
                # front here; the AO kernel (nvfp4_mm_triton) itself raises on the
                # per-rank local dims once TP has sharded the weight.
                for name in ("in_features", "out_features"):
                    value = getattr(self, name)
                    if value % _NVFP4_BLOCK:
                        raise ValueError(
                            f"NVFP4 requires {name} divisible by {_NVFP4_BLOCK}; "
                            f"got {name}={value}. NVFP4 cannot quantize this Linear; "
                            "exclude it from the converter fqns."
                        )

            def build(self, **kwargs):
                # sharding_config (the stock colwise/rowwise weight placement) is
                # attached by update_from_config after this Config is built, so it
                # is available here but not in __post_init__. Fold it into the
                # local SPMD region for the opaque nvfp4_linear op now, so base
                # Module.parallelize consumes it directly.
                # slots=True breaks zero-arg super(), so call the parent explicitly.
                instance = Linear.Config.build(self, **kwargs)
                if instance._sharding_config is not None:
                    sc = instance._sharding_config
                    weight_tp = sc.state_shardings["weight"].local_type.get(TP)
                    rowwise = (
                        isinstance(weight_tp, spmd.Shard)
                        and weight_tp.dim == instance.weight.ndim - 1
                    )
                    if rowwise:
                        in_layout = dense_activation_placement(
                            tp=spmd.S(-1), cp=spmd.S(0)
                        )
                    else:
                        in_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
                    instance._sharding_config = replace(
                        sc,
                        state_shardings={
                            **sc.state_shardings,
                            "_sr_seed": SpmdType(
                                {
                                    MeshAxisName.DP: spmd.V,
                                    MeshAxisName.CP: spmd.V,
                                    TP: spmd.V,
                                }
                            ),
                        },
                        in_src_shardings={
                            **(sc.in_src_shardings or {}),
                            "input": in_layout,
                        },
                        in_dst_shardings={
                            **(sc.in_dst_shardings or {}),
                            "input": in_layout,
                        },
                        local_spmd=True,
                    )
                return instance

        def __init__(self, config: Config):
            TorchAONVFP4Linear.__init__(
                self,
                config.in_features,
                config.num_linears * config.out_features,
                bias=config.bias,
                kernel_preference=_to_kernel_preference(config.kernel_preference),
                use_fast_math=config.use_fast_math,
                recipe=NVFP4Recipe(config.recipe),
            )
            self.ms_eden_fast_path = config.ms_eden_fast_path
            self.out_features = config.out_features
            self.num_linears = config.num_linears
            if config.num_linears > 1:
                self.weight = torch.nn.Parameter(
                    self.weight.detach().unflatten(
                        0, (config.num_linears, config.out_features)
                    ),
                    requires_grad=self.weight.requires_grad,
                )
                if self.bias is not None:
                    self.bias = torch.nn.Parameter(
                        self.bias.detach().unflatten(
                            0, (config.num_linears, config.out_features)
                        ),
                        requires_grad=self.bias.requires_grad,
                    )
            # TorchAO created the runtime buffers on the (meta) build device.
            # Re-register them as None so ``_distribute_states`` skips them and
            # ``_init_self_buffers`` materializes them on the real device, per
            # torchtitan's buffer protocol.
            # _sr_seed is a stochastic-rounding seed drawn locally per rank with
            # no cross-rank coordination. Ranks that share an RNG stream (all but
            # the pp axis, which set_determinism seeds distinctly) draw the same
            # value, but that is fine: SR stays unbiased and NVFP4 never
            # communicates quantized values, so the seed need not differ across
            # ranks. It is non-persistent (a Philox key needs no checkpointing).
            # Re-register it None so it is not distributed and is re-drawn per
            # rank in _init_self_buffers.
            self.register_buffer("_sr_seed", None, persistent=False)
            # _rht_sign_vector is the fixed _HARDCODED_SIGN_VECTOR (see module
            # top): identical on every rank, so it is non-persistent (a
            # deterministic constant needs no checkpointing) and re-materialized
            # per rank in _init_self_buffers with no cross-rank broadcast.
            self.register_buffer("_rht_sign_vector", None, persistent=False)
            if self.recipe is NVFP4Recipe.V2:
                self.register_buffer("_dgrad_rht_sign_vector", None, persistent=False)
            self._rht_sign_vector_tuple = None

        def _local_rht_sign_vector(self) -> torch.Tensor:
            sign_vector = self._rht_sign_vector
            if sign_vector is not None and sign_vector.device.type != "meta":
                sign_vector = sign_vector.reshape(-1)
            return sign_vector

        def _refresh_rht_sign_vector_tuple(self) -> None:
            if self.recipe is NVFP4Recipe.V2:
                self._rht_sign_vector_tuple = None
                return
            sign_vector = self._local_rht_sign_vector()
            self._rht_sign_vector_tuple = (
                None if sign_vector is None else _rht_sign_vector_to_tuple(sign_vector)
            )

        def _load_from_state_dict(self, *args, **kwargs):
            super()._load_from_state_dict(*args, **kwargs)
            self._refresh_rht_sign_vector_tuple()

        @property
        def rht_sign_vector(self) -> tuple[int, ...]:
            if self._rht_sign_vector_tuple is None:
                self._refresh_rht_sign_vector_tuple()
            if self._rht_sign_vector_tuple is None:
                raise RuntimeError("rht_sign_vector is not materialized")
            return self._rht_sign_vector_tuple

        def _init_self_buffers(
            self, *, buffer_device: torch.device | None = None
        ) -> None:
            dev = (
                buffer_device
                if buffer_device is not None
                else cast(torch.Tensor, self.weight).device
            )
            # Per-rank seed: a plain local tensor (not distributed), so each rank
            # draws its own.
            self._sr_seed = torch.randint(
                -9_223_372_036_854_775_808,
                9_223_372_036_854_775_807,
                (1,),
                dtype=torch.int64,
                device=dev,
            )
            if self.recipe is NVFP4Recipe.V2:
                self._rht_sign_vector = _draw_sign_vector(
                    _V2_RHT_SIZE, _V2_LINEAR_WGRAD_SEED, dev
                )
                self._dgrad_rht_sign_vector = _draw_sign_vector(
                    _V2_RHT_SIZE, _V2_LINEAR_DGRAD_SEED, dev
                )
            else:
                self._rht_sign_vector = _make_rht_sign_vector(
                    _HARDCODED_SIGN_VECTOR, device=dev
                )
            self._refresh_rht_sign_vector_tuple()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            local_out_features = self.weight.shape[-2]
            if local_out_features % _NVFP4_BLOCK:
                raise ValueError(
                    "NVFP4 requires local out_features divisible by "
                    f"{_NVFP4_BLOCK}; got {local_out_features}. Adjust the "
                    "Linear out_features or TP degree so quantization blocks "
                    "do not span projection boundaries."
                )
            weight = self.weight.flatten(0, -2)
            bias = None if self.bias is None else self.bias.flatten()
            if self.recipe is NVFP4Recipe.V2:
                output = nvfp4_linear_v2(
                    input,
                    weight,
                    bias,
                    wgrad_rht=self._rht_sign_vector,
                    dgrad_rht=self._dgrad_rht_sign_vector,
                    sr_seed=self._sr_seed,
                    kernel_preference=self.kernel_preference,
                    use_fast_math=self.use_fast_math,
                    ms_eden_fast_path=self.ms_eden_fast_path,
                )
            elif self.recipe is NVFP4Recipe.V1_REQUANT:
                output = nvfp4_linear_v1_requant(
                    input,
                    weight,
                    bias,
                    sign_vector=self.rht_sign_vector,
                    sr_seed=self._sr_seed,
                    kernel_preference=self.kernel_preference,
                    use_fast_math=self.use_fast_math,
                )
            else:
                output = nvfp4_linear(
                    input,
                    weight,
                    bias,
                    sr_seed=self._sr_seed,
                    sign_vector=self.rht_sign_vector,
                    kernel_preference=self.kernel_preference,
                    use_fast_math=self.use_fast_math,
                )
            if self.num_linears == 1:
                return output
            return output.unflatten(-1, self.weight.shape[:-1])

        def reset_parameters(self) -> None:
            Linear.reset_parameters(self)

except ImportError:
    NVFP4Linear = None


def _draw_sign_vector(length: int, seed: int, device) -> torch.Tensor:
    """Draw a deterministic {-1, +1} vector shared by ranks."""
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def _expects_v2(module) -> bool:
    recipe = getattr(module, "recipe", None)
    if recipe is not None and getattr(recipe, "value", recipe) == "v2":
        return True
    return "v2" in (
        getattr(module, "_fc1_recipe", None),
        getattr(module, "_fc2_recipe", None),
    )


def build_nvfp4_sign_resampler(model_parts, seed: int):
    """Build the V2 RHT sign cadence after model buffers are materialized."""
    try:
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_rht_cadence import (
            iter_dynamic_sign_buffers,
            resample_nvfp4_rht_signs,
        )
    except ImportError:
        return None

    strays = [
        f"{fqn}.{name}"
        for part in model_parts
        for fqn, name, _, _ in iter_dynamic_sign_buffers(part)
        if not _expects_v2(part.get_submodule(fqn))
    ]
    if strays:
        logger.warning(
            "NVFP4: %d dynamic RHT sign buffers sit on modules that do not run "
            "V2 (e.g. %s); they will be resampled but not read.",
            len(strays),
            strays[0],
        )

    counts = [sum(1 for _ in iter_dynamic_sign_buffers(part)) for part in model_parts]
    total = sum(counts)
    if not total:
        if any(_expects_v2(module) for part in model_parts for module in part.modules()):
            logger.warning(
                "NVFP4: a V2 recipe is configured but no dynamic RHT sign buffers "
                "were found; this run does not measure V2."
            )
        return None

    parts = [part for part, count in zip(model_parts, counts, strict=True) if count]
    logger.info("NVFP4 V2: resampling %d RHT sign buffers per microbatch", total)

    def resample(step: int, microbatch: int) -> None:
        for part in parts:
            resample_nvfp4_rht_signs(
                part, seed=seed, step=step, microbatch=microbatch
            )

    return resample


_nvfp4_experts_cache: dict[type, type] = {}


def _get_nvfp4_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an NVFP4-quantized subclass of ``parent_cls``."""
    if parent_cls in _nvfp4_experts_cache:
        return _nvfp4_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class NVFP4GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            kernel_preference: str = "cutedsl"
            use_fast_math: bool = True
            fc1_recipe: str = "v1"
            fc2_recipe: str = "v1"
            ms_eden_fast_path: bool = True

        def __init__(self, config: Config):
            super().__init__(config)
            self._kernel_preference = _to_kernel_preference(config.kernel_preference)
            self._use_fast_math = config.use_fast_math
            self._ms_eden_fast_path = config.ms_eden_fast_path
            for name in ("fc1_recipe", "fc2_recipe"):
                value = getattr(config, name)
                if value not in _NVFP4_RECIPES:
                    raise ValueError(
                        f"{name} must be one of {_NVFP4_RECIPES}, got {value!r}"
                    )
            self._fc1_recipe = config.fc1_recipe
            self._fc2_recipe = config.fc2_recipe
            module = cast(Module, self)
            module.register_buffer("_sr_seed", None, persistent=False)
            module.register_buffer("_rht_sign_vector", None, persistent=False)
            module.register_buffer("_fc2_sr_seed", None, persistent=False)
            module.register_buffer("_fc2_rht_sign_vector", None, persistent=False)
            module.register_buffer(
                "_fc2_dgrad_rht_sign_vector", None, persistent=False
            )
            self._rht_sign_vector_tuple = None

        def _refresh_rht_sign_vector_tuple(self) -> None:
            sign_vector = self._rht_sign_vector
            if sign_vector is not None and sign_vector.device.type != "meta":
                sign_vector = sign_vector.reshape(-1)
            self._rht_sign_vector_tuple = (
                None if sign_vector is None else _rht_sign_vector_to_tuple(sign_vector)
            )

        def _load_from_state_dict(self, *args, **kwargs):
            super()._load_from_state_dict(*args, **kwargs)
            self._refresh_rht_sign_vector_tuple()

        @property
        def rht_sign_vector(self) -> tuple[int, ...]:
            if self._rht_sign_vector_tuple is None:
                self._refresh_rht_sign_vector_tuple()
            if self._rht_sign_vector_tuple is None:
                raise RuntimeError("rht_sign_vector is not materialized")
            return self._rht_sign_vector_tuple

        def _init_self_buffers(
            self, *, buffer_device: torch.device | None = None
        ) -> None:
            super()._init_self_buffers(buffer_device=buffer_device)
            dev = (
                buffer_device
                if buffer_device is not None
                else next(cast(Module, self).parameters()).device
            )
            self._sr_seed = torch.randint(
                -9_223_372_036_854_775_808,
                9_223_372_036_854_775_807,
                (1,),
                dtype=torch.int64,
                device=dev,
            )
            self._rht_sign_vector = _make_rht_sign_vector(
                _HARDCODED_SIGN_VECTOR, device=dev
            )
            self._refresh_rht_sign_vector_tuple()
            self._fc2_sr_seed = torch.randint(
                -9_223_372_036_854_775_808,
                9_223_372_036_854_775_807,
                (1,),
                dtype=torch.int64,
                device=dev,
            )
            if _expects_v2(self):
                self._fc2_rht_sign_vector = _draw_sign_vector(
                    _V2_RHT_SIZE, 0xFC2, dev
                )
                self._fc2_dgrad_rht_sign_vector = _draw_sign_vector(
                    _V2_RHT_SIZE, 0xDEAD, dev
                )

        def _grouped_mm(self, *, A, weight_EOI, offs):
            return _to_nvfp4_rht_rs_then_scaled_grouped_mm(
                A,
                weight_EOI,
                self.rht_sign_vector,
                self._sr_seed,
                offs=offs,
                pad_token_groups_for_grouped_mm=False,
                kernel_preference=self._kernel_preference,
                use_fast_math=self._use_fast_math,
            )

        def _recipe_grouped_mm(self, recipe, A, weight_EOI, offs, *, is_fc2):
            from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
                nvfp4_v1_requant_grouped_mm,
                nvfp4_v2_grouped_mm,
            )

            seed = self._fc2_sr_seed if is_fc2 else self._sr_seed
            if recipe == "v1":
                return _to_nvfp4_rht_rs_then_scaled_grouped_mm(
                    A,
                    weight_EOI,
                    self.rht_sign_vector,
                    seed,
                    offs=offs,
                    pad_token_groups_for_grouped_mm=False,
                    kernel_preference=self._kernel_preference,
                    use_fast_math=self._use_fast_math,
                )
            if recipe == "v1_requant":
                return nvfp4_v1_requant_grouped_mm(
                    A,
                    weight_EOI,
                    sign_vector=self.rht_sign_vector,
                    sr_seed=seed,
                    offs=offs,
                    pad_token_groups_for_grouped_mm=False,
                    kernel_preference=self._kernel_preference,
                    use_fast_math=self._use_fast_math,
                )
            return nvfp4_v2_grouped_mm(
                A,
                weight_EOI,
                wgrad_rht=self._fc2_rht_sign_vector,
                dgrad_rht=self._fc2_dgrad_rht_sign_vector,
                sr_seed=seed,
                offs=offs,
                pad_token_groups_for_grouped_mm=False,
                kernel_preference=self._kernel_preference,
                use_fast_math=self._use_fast_math,
                ms_eden_fast_path=self._ms_eden_fast_path,
            )

        def forward(self, x_RD, num_tokens_per_expert_E):
            if self._fc1_recipe == "v1" and self._fc2_recipe == "v1":
                return super().forward(x_RD, num_tokens_per_expert_E)

            offsets_E = torch.cumsum(
                num_tokens_per_expert_E, dim=0, dtype=torch.int32
            )
            if spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
                for axis in ("dp", "cp"):
                    spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

            gate_RF = self._recipe_grouped_mm(
                self._fc1_recipe,
                x_RD.bfloat16(),
                self.w1_EFD,
                offsets_E,
                is_fc2=False,
            )
            up_RF = self._recipe_grouped_mm(
                self._fc1_recipe,
                x_RD.bfloat16(),
                self.w3_EFD,
                offsets_E,
                is_fc2=False,
            )
            h_RF = self.activation_fn(gate_RF, up_RF)
            return self._recipe_grouped_mm(
                self._fc2_recipe,
                h_RF,
                self.w2_EDF,
                offsets_E,
                is_fc2=True,
            ).type_as(x_RD)

    NVFP4GroupedExperts.__name__ = f"NVFP4{parent_cls.__name__}"
    NVFP4GroupedExperts.__qualname__ = f"NVFP4{parent_cls.__name__}"
    _nvfp4_experts_cache[parent_cls] = NVFP4GroupedExperts
    return NVFP4GroupedExperts


def nvfp4_bf16_tail_fqns(num_layers: int, bf16_tail_fraction: float) -> list[str]:
    """Converter ``fqns`` selecting the leading decoder layers for NVFP4 while
    keeping the last ``ceil(num_layers * bf16_tail_fraction)`` layers in bf16.

    Each fqn has a trailing '.' so 'layers.1.' matches layer 1 only, not
    'layers.10' (NVFP4LinearConverter.convert substring-matches). Raises if the
    fraction would leave no layer to convert: an empty fqns list would instead
    convert *all* Linears (the ``not fqns`` branch in convert), the opposite of
    the intent.
    """
    num_bf16 = math.ceil(num_layers * bf16_tail_fraction)
    convert_upto = num_layers - num_bf16
    if convert_upto <= 0:
        raise ValueError(
            f"bf16_tail_fraction={bf16_tail_fraction} keeps all {num_layers} "
            "layers in bf16; nothing to convert to NVFP4."
        )
    return [f"layers.{i}." for i in range(convert_upto)]
