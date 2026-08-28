# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 quantization converter.

Swaps dense ``Linear.Config`` nodes for :class:`NVFP4Linear`, which keeps a bf16
weight and quantizes activations, weights, and gradients to NVFP4 on the fly via
TorchAO's ``nvfp4_training`` kernels (NVIDIA Blackwell / sm_100+, CUDA only).

Like :class:`MXFP8LinearConverter`, this is a pure leaf swap: it inherits the
model's stock colwise/rowwise sharding and changes only the GEMM. Under tensor
parallelism the block boundary keeps its stock bf16 collectives (all-gather /
reduce-scatter); NVFP4 does not move fp4 codes over the wire.
"""

import math
from dataclasses import dataclass, field, fields, replace
from typing import Literal, cast

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_size
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, SpmdLayout
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

TP = MeshAxisName.TP

# TorchAO's NVFP4 kernels require each local GEMM dimension to be a multiple
# of 128, on both the Triton and the CuteDSL backend.
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
        """Validate a config string into the enum torchao's NVFP4 paths accept.

        Both AO seams default to ``AUTO``, which silently takes CuteDSL when its
        runtime is importable and Triton otherwise -- so with no explicit choice
        the backend is a property of the container image rather than the recipe,
        and two runs of the same config can measure different kernels.
        """
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
        what: str, pref: KernelPreference, use_fast_math: bool, per_op: bool
    ) -> None:
        """Record the quantization backend this recipe pins, and refuse to fall back.

        The backend is a numerics decision, not only a speed one: the stochastic-
        rounding streams differ between Triton and CuteDSL (torchao's own
        ``test_nvfp4_linear_auto_runs_on_triton_fallback`` compares the two under
        RTNE for exactly that reason). A run must therefore not discover its
        backend from the container image, which is why CUTEDSL raises here, at
        converter build, rather than at the first kernel call.
        """
        if pref is KernelPreference.CUTEDSL and not cutedsl_nvfp4_kernels_available():
            raise RuntimeError(
                f"{what} kernel_preference=cutedsl, but the CuteDSL runtime is "
                f"unavailable ({cutedsl_nvfp4_unavailable_reason()})."
            )
        logger.info(
            "%s kernel_preference=%s, use_fast_math=%s", what, pref.value, use_fast_math
        )
        if pref is KernelPreference.AUTO:
            # Only AUTO leaves the backend undetermined by the recipe. No single
            # "resolved" backend is reportable for the grouped path: it resolves
            # per op, so one step can mix Triton and CuteDSL.
            logger.warning(
                "%s kernel_preference=auto: the backend follows the container, not "
                "the recipe (%s), and the SR streams differ between backends.",
                what,
                "resolved per op"
                if per_op
                else ("cutedsl" if cutedsl_nvfp4_kernels_available() else "triton"),
            )

    # The NVFP4 GEMM is a raw autograd Function that runs on local shards inside
    # the spmd.local_map region. Mark it local-safe so SPMD type checking
    # propagates through it; the local_map boundary declares the real
    # colwise/rowwise output and input-gradient types.
    spmd.register_local_autograd_function(nvfp4_matmul)
    # V1_REQUANT's forward is a second raw autograd Function reached from the same
    # local_map region, so it needs the same declaration or SPMD type checking stops
    # at it.
    spmd.register_local_autograd_function(_NVFP4LinearV1Requant)
    # V2's forward is a third such Function, reached from the same region.
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
            """NVFP4 quantization backend: "auto", "triton", or "cutedsl"."""

            use_fast_math: bool = True
            """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

            recipe: str = "v1"
            """NVFP4 recipe: one of _NVFP4_RECIPES.

            "v2" draws a 128-element ``_rht_sign_vector`` and a second
            ``_dgrad_rht_sign_vector``, and expects the training loop to advance
            them through ``build_nvfp4_sign_resampler``. Without that the run is
            still correct, but every step reuses the initial draw and the variance
            reduction V2 exists for is forfeited.
            """

            def __post_init__(self) -> None:
                if self.recipe not in _NVFP4_RECIPES:
                    raise ValueError(
                        f"recipe must be one of {_NVFP4_RECIPES}, "
                        f"got {self.recipe!r}"
                    )
                # NVFP4 needs every GEMM dim to be a multiple of 128 on either
                # backend. in_features / out_features are known at config-build time
                # (the TP degree is not), so reject the model-dim violations up
                # front here; the AO kernel (nvfp4_matmul) itself raises on the
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
                # local_map region for the opaque nvfp4_linear op now, so base
                # Module.parallelize consumes it directly.
                # slots=True breaks zero-arg super(), so call the parent explicitly.
                instance = Linear.Config.build(self, **kwargs)
                if instance._sharding_config is not None:
                    sc = instance._sharding_config
                    weight_tp = (
                        sc.state_shardings["weight"].per_axis_spmd_types().get(TP)
                    )
                    rowwise = isinstance(weight_tp, spmd.Shard) and weight_tp.dim == 1
                    if rowwise:
                        in_layout = dense_activation_placement(tp=spmd.S(-1))
                        in_grad = dense_activation_placement(tp=spmd.S(-1))
                    else:
                        in_layout = dense_activation_placement(tp=spmd.R)
                        in_grad = dense_activation_placement(tp=spmd.P)
                    instance._sharding_config = replace(
                        sc,
                        state_shardings={
                            **sc.state_shardings,
                            "_sr_seed": SpmdLayout(
                                {
                                    MeshAxisName.DP: spmd.V,
                                    MeshAxisName.CP: spmd.V,
                                    TP: spmd.V,
                                }
                            ),
                        },
                        in_src_shardings={
                            **(sc.in_src_shardings or {}),
                            "x": in_layout,
                        },
                        in_dst_shardings={
                            **(sc.in_dst_shardings or {}),
                            "x": in_layout,
                        },
                        local_map=LocalMapConfig(in_grad_placements=(in_grad,)),
                    )
                return instance

        def __init__(self, config: Config):
            # Both flags go to the TorchAO base rather than into private copies:
            # the base owns them (its own forward and from_linear read
            # self.kernel_preference), and its default is AUTO, so a shadowing
            # copy would leave every inherited path resolving by container.
            # Validated here, once per module, so a bad recipe string fails at
            # model build rather than at the first kernel call.
            TorchAONVFP4Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
                kernel_preference=_to_kernel_preference(config.kernel_preference),
                use_fast_math=config.use_fast_math,
                recipe=NVFP4Recipe(config.recipe),
            )
            # V1_REQUANT and V2 are Triton-only; the CuteDSL kernels exist for V1
            # alone. Raise rather than silently downgrade, matching the grouped
            # converter.
            if (
                config.recipe != "v1"
                and self.kernel_preference is KernelPreference.CUTEDSL
            ):
                raise ValueError(
                    "kernel_preference='cutedsl' is only available for recipe 'v1'; "
                    f"got recipe={config.recipe!r}. Use 'triton'."
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
            # per rank in _init_self_buffers with no cross-rank broadcast. Under
            # V2 it holds a 128-element resampled vector instead, but the
            # None-then-materialize protocol is the same.
            self.register_buffer("_rht_sign_vector", None, persistent=False)
            # V2 rotates the dgrad axis too, so TorchAO gave it a second vector.
            # Same treatment: derived per rank, never checkpointed.
            if self.recipe is NVFP4Recipe.V2:
                self.register_buffer("_dgrad_rht_sign_vector", None, persistent=False)
            self._rht_sign_vector_tuple = None

        def _local_rht_sign_vector(self) -> torch.Tensor:
            sign_vector = self._rht_sign_vector
            if sign_vector is not None and sign_vector.device.type != "meta":
                sign_vector = sign_vector.reshape(-1)
            return sign_vector

        def _refresh_rht_sign_vector_tuple(self) -> None:
            # V2 resamples its vector, so there is no stable tuple to cache and
            # nothing on its forward path wants one -- it passes the buffer itself.
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
                # Fixed shape and updated in place by torchao's
                # resample_nvfp4_rht_signs, so the addresses survive CUDA-graph
                # capture. Derived from a constant seed rather than broadcast, for
                # the same reason as the static basis below. These are only the
                # pre-resample draw: once the trainer drives the cadence every
                # vector is keyed by its own FQN.
                self._rht_sign_vector = _draw_sign_vector(
                    _V2_RHT_SIZE, _V2_LINEAR_WGRAD_SEED, dev
                )
                self._dgrad_rht_sign_vector = _draw_sign_vector(
                    _V2_RHT_SIZE, _V2_LINEAR_DGRAD_SEED, dev
                )
            else:
                # Static RHT basis: identical on every rank by construction, so it
                # is a plain local tensor with no cross-rank broadcast.
                self._rht_sign_vector = _make_rht_sign_vector(
                    _HARDCODED_SIGN_VECTOR, device=dev
                )
            self._refresh_rht_sign_vector_tuple()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Dispatched here rather than deferred to the TorchAO base: the base's
            # forward also carries the process_group TP protocol, which this class
            # does not use -- TP here is spmd local_map sharding instead.
            if self.recipe is NVFP4Recipe.V2:
                return nvfp4_linear_v2(
                    x,
                    self.weight,
                    self.bias,
                    wgrad_rht=self._rht_sign_vector,
                    dgrad_rht=self._dgrad_rht_sign_vector,
                    sr_seed=self._sr_seed,
                    use_fast_math=self.use_fast_math,
                )
            if self.recipe is not NVFP4Recipe.V1:
                return nvfp4_linear_v1_requant(
                    x,
                    self.weight,
                    self.bias,
                    sign_vector=self.rht_sign_vector,
                    sr_seed=self._sr_seed,
                    use_fast_math=self.use_fast_math,
                )
            return nvfp4_linear(
                x,
                self.weight,
                self.bias,
                sr_seed=self._sr_seed,
                sign_vector=self.rht_sign_vector,
                kernel_preference=self.kernel_preference,
                use_fast_math=self.use_fast_math,
            )

except ImportError:
    NVFP4Linear = None


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


class NVFP4LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with NVFP4Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply NVFP4 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted -- pass explicit fqns to keep
        the LM head in bf16, which the mixed recipe leaves unquantized for stability.
        """

        kernel_preference: str = "cutedsl"
        """NVFP4 quantization backend: "auto", "triton", or "cutedsl".

        Defaults to "cutedsl" so that throughput runs are pinned by default.
        "auto" resolves to CuteDSL when its runtime is importable and Triton
        otherwise, so it measures different kernels on containers that differ
        only in which packages are installed -- and it does so silently, whereas
        "cutedsl" raises when the runtime is missing.
        """

        use_fast_math: bool = True
        """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

        recipe: Literal["v1", "v1_requant", "v2"] = "v1"
        """NVFP4 recipe for the converted Linears.

        "v1" is the shipped recipe and the default, so an unchanged config produces
        exactly what it produced before. "v1_requant" moves the weight from 16x16 2D
        scaling to 1x16 rowwise plus a lazy backward requantization, which puts the
        forward and dgrad GEMMs on one and the same W_qdq.

        "v2" additionally rotates the dgrad axis by RHT-128 and rounds the gradient
        with MS-EDEN. It carries two 128-element sign vectors that the training loop
        must advance once per microbatch -- ``Trainer`` does this through
        ``build_nvfp4_sign_resampler``. A loop that never calls it still trains
        correctly, but reuses the initial draw forever and measures something other
        than V2, which is why the resampler logs its buffer count at startup.
        """

    def __init__(self, config: Config):
        self.config = config

        if NVFP4Linear is None:
            raise ImportError(
                "torchao is not installed or does not provide the NVFP4 training "
                "prototype. Install a torchao build with "
                "torchao.prototype.moe_training.nvfp4_training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

        _log_kernel_preference(
            f"NVFP4Linear recipe={self.config.recipe}",
            _to_kernel_preference(self.config.kernel_preference),
            self.config.use_fast_math,
            per_op=False,
        )

    def convert(self, model_config):
        assert NVFP4Linear is not None
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = NVFP4Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                    kernel_preference=self.config.kernel_preference,
                    use_fast_math=self.config.use_fast_math,
                    recipe=self.config.recipe,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4Linear")
        return model_config


# The NVFP4 recipes, shared by the linear and MoE paths. Linears carried a
# narrower list until they grew V2's dynamic sign buffers. "v1" is the shipped
# recipe and stays the default everywhere, so an unchanged config produces exactly
# what it produced before the other two existed. Design doc §17 recommends FC1
# (w1 gate, w3 up) on "v1_requant" and FC2 (w2 down) on "v2"; that split is a
# configuration choice, not a hard-coded assumption, and the recipes may be
# swapped.
_NVFP4_RECIPES = ("v1", "v1_requant", "v2")

# V2 rotates by RHT-128 rather than RHT-16.
_V2_RHT_SIZE = 128

# The pre-resample draw for a V2 Linear's two sign vectors. Distinct from the MoE
# path's 0xFC2/0xDEAD so a linear and an expert module in the same layer do not
# open the run on the same basis. Only the first step depends on these: from the
# first resample onward every buffer is keyed by its own FQN.
_V2_LINEAR_WGRAD_SEED = 0x1EA1
_V2_LINEAR_DGRAD_SEED = 0x1EA2


def _draw_sign_vector(length: int, seed: int, device) -> torch.Tensor:
    """Deterministic {-1, +1} int8 vector, identical on every rank for a given seed.

    Derived rather than broadcast, for the same reason ``_HARDCODED_SIGN_VECTOR`` is
    hardcoded: the RHT only cancels between the two operands of a GEMM when both were
    rotated by the same basis, and under EP those operands can live on different
    ranks. Deriving from a fixed seed makes them agree by construction, with no
    collective.
    """
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (length,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def _expects_v2(module) -> bool:
    """Whether *module* was configured to run the V2 recipe."""
    recipe = getattr(module, "recipe", None)
    if recipe is not None and getattr(recipe, "value", recipe) == "v2":
        return True
    return "v2" in (
        getattr(module, "_fc1_recipe", None),
        getattr(module, "_fc2_recipe", None),
    )


def build_nvfp4_sign_resampler(model_parts, seed: int):
    """Return a ``(step, microbatch) -> None`` callable, or ``None`` if nothing needs it.

    V2 resamples its RHT sign vectors -- wgrad every microbatch, dgrad every
    optimizer step -- and TorchAO mutates the buffers in place so their addresses
    survive CUDA-graph capture. Nothing else drives this: the converter protocol
    rewrites a config tree and is never called again, so the cadence can only come
    from the training loop.

    Call this **after** ``init_weights``. The buffers are registered as ``None`` and
    materialized in ``_init_self_buffers``, and ``iter_dynamic_sign_buffers`` skips a
    ``None`` buffer -- so probing earlier finds nothing and silently disables the
    cadence for the whole run.

    Returning ``None`` when there is nothing to resample keeps the per-microbatch
    cost of a non-V2 run at one identity check, and keeps recipe branching out of the
    trainer.
    """
    try:
        from torchao.prototype.moe_training.nvfp4_training.nvfp4_rht_cadence import (
            iter_dynamic_sign_buffers,
            resample_nvfp4_rht_signs,
        )
    except ImportError:
        return None

    # Selection is by buffer length and name suffix, with no reference to the
    # recipe, so a 128-element buffer materialized on a non-V2 module is
    # indistinguishable from a real one here. That is not hypothetical: the grouped
    # experts' _fc2 sign vectors used to be drawn on every recipe, and E14's
    # V1_REQUANT arm resampled 52 buffers nothing reads. Gated at the draw now, so
    # this should be empty -- it is a guard against the next unconditional buffer,
    # because a stray one costs work AND makes a nonzero count stop proving the
    # cadence reached the V2 modules.
    strays = [
        f"{fqn}.{name}"
        for part in model_parts
        for fqn, name, _, _ in iter_dynamic_sign_buffers(part)
        if not _expects_v2(part.get_submodule(fqn))
    ]
    if strays:
        logger.warning(
            "NVFP4: %d dynamic RHT sign buffers sit on modules that do not run V2 "
            "(e.g. %s). They will be resampled every microbatch and read by nothing, "
            "and their presence means a nonzero buffer count no longer proves the "
            "cadence reached the V2 modules.",
            len(strays),
            strays[0],
        )

    counts = [sum(1 for _ in iter_dynamic_sign_buffers(part)) for part in model_parts]
    total = sum(counts)
    if not total:
        # A V2 module with no dynamic buffer means _init_self_buffers did not run,
        # or the recipe never reached the modules. The run would train and log a
        # plausible loss while measuring the initial draw forever, so say so.
        if any(_expects_v2(m) for part in model_parts for m in part.modules()):
            logger.warning(
                "NVFP4: a V2 recipe is configured but no dynamic RHT sign buffers "
                "were found, so the sign vectors will never be resampled. This run "
                "does not measure V2."
            )
        return None

    parts = [part for part, count in zip(model_parts, counts) if count]
    logger.info("NVFP4 V2: resampling %d RHT sign buffers per microbatch", total)

    def resample(step: int, microbatch: int) -> None:
        for part in parts:
            resample_nvfp4_rht_signs(
                part, seed=seed, step=step, microbatch=microbatch
            )

    return resample


_nvfp4_experts_cache: dict[type, type] = {}


def _get_nvfp4_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an NVFP4-quantized subclass of *parent_cls*.

    Works for any experts module exposing the ``_grouped_mm`` seam (the common
    ``GroupedExperts`` and ``GptOssGroupedExperts``). The returned class has a
    proper ``_owner`` set by ``__init_subclass__``.

    The subclass overrides ``_grouped_mm`` to call torchao's
    ``_to_nvfp4_then_scaled_grouped_mm``. It carries the same runtime NVFP4 state
    as :class:`NVFP4Linear` -- a per-rank ``_sr_seed`` and the fixed
    ``_HARDCODED_SIGN_VECTOR`` -- as non-persistent buffers materialized per rank
    in ``_init_self_buffers`` (see NVFP4Linear for why neither is checkpointed).
    """
    if parent_cls in _nvfp4_experts_cache:
        return _nvfp4_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class NVFP4GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            kernel_preference: str = "cutedsl"
            """NVFP4 quantization backend: "auto", "triton", or "cutedsl"."""

            use_fast_math: bool = True
            """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

            fc1_recipe: str = "v1"
            """NVFP4 recipe for the FC1 gate/up GEMMs (``w1``, ``w3``)."""

            fc2_recipe: str = "v1"
            """NVFP4 recipe for the FC2 down GEMM (``w2``)."""

        def __init__(self, config: Config):
            super().__init__(config)
            self._kernel_preference = _to_kernel_preference(config.kernel_preference)
            self._use_fast_math = config.use_fast_math
            for name in ("fc1_recipe", "fc2_recipe"):
                value = getattr(config, name)
                if value not in _NVFP4_RECIPES:
                    raise ValueError(
                        f"{name} must be one of {_NVFP4_RECIPES}, got {value!r}"
                    )
            self._fc1_recipe = config.fc1_recipe
            self._fc2_recipe = config.fc2_recipe
            # V1_REQUANT and V2 are Triton-only for now; the CuteDSL grouped kernels
            # exist for V1 alone. Raise rather than silently downgrade the backend.
            if (
                self._fc1_recipe != "v1" or self._fc2_recipe != "v1"
            ) and self._kernel_preference is KernelPreference.CUTEDSL:
                raise ValueError(
                    "kernel_preference='cutedsl' is only available for recipe 'v1'; "
                    f"got fc1_recipe={self._fc1_recipe!r}, "
                    f"fc2_recipe={self._fc2_recipe!r}. Use 'triton'."
                )
            # Same buffer protocol as NVFP4Linear.__init__: register the runtime
            # buffers as None so _distribute_states skips them and
            # _init_self_buffers materializes them per rank on the real device.
            self.register_buffer("_sr_seed", None, persistent=False)
            self.register_buffer("_rht_sign_vector", None, persistent=False)
            self._rht_sign_vector_tuple = None
            # Once FC1 and FC2 run different recipes they no longer share a
            # quantization path, so they must not share a seed or a sign vector:
            # correlated noise between them defeats the point of drawing either.
            # Registered unconditionally so the buffer set does not depend on the
            # recipe, which keeps _distribute_states and meta-init uniform. The
            # two sign vectors are only DRAWN under V2 -- see _init_self_buffers.
            self.register_buffer("_fc2_sr_seed", None, persistent=False)
            self.register_buffer("_fc2_rht_sign_vector", None, persistent=False)
            self.register_buffer("_fc2_dgrad_rht_sign_vector", None, persistent=False)

        def _local_rht_sign_vector(self) -> torch.Tensor:
            sign_vector = self._rht_sign_vector
            if sign_vector is not None and sign_vector.device.type != "meta":
                sign_vector = sign_vector.reshape(-1)
            return sign_vector

        def _refresh_rht_sign_vector_tuple(self) -> None:
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
            super()._init_self_buffers(buffer_device=buffer_device)
            dev = (
                buffer_device
                if buffer_device is not None
                else cast(torch.Tensor, self.w1_EFD).device
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
            # Per-rank, like _sr_seed, and distinct from it.
            self._fc2_sr_seed = torch.randint(
                -9_223_372_036_854_775_808,
                9_223_372_036_854_775_807,
                (1,),
                dtype=torch.int64,
                device=dev,
            )
            # Drawn only when a half actually runs V2: the V2 branch of
            # _recipe_grouped_mm is the only reader, and the v1 and v1_requant
            # branches pass the 16-element _HARDCODED_SIGN_VECTOR instead.
            #
            # Leaving them None otherwise is load-bearing, not tidiness.
            # iter_dynamic_sign_buffers selects on numel() == 128 and the name
            # suffix alone, so a materialized pair is resampled every microbatch on
            # a run with no V2 module anywhere -- work for buffers nothing reads,
            # and a "NVFP4 V2: resampling N ..." line on a pure V1 run. It also
            # destroyed the count's diagnostic value: with an unconditional pair, a
            # nonzero count no longer proved the cadence had reached the V2
            # modules. Measured at 52 on the 16B V1_REQUANT arm of E14 (26 MoE
            # layers x 2); see dsv3/v2/README.md in core-models.
            #
            # Numerics were never affected, and gating does not change V2: the
            # buffers V2 reads are still drawn here, under the same names, so
            # torchao's FQN-derived resample schedule is untouched.
            if _expects_v2(self):
                # Fixed shape and updated in place by torchao's
                # resample_nvfp4_rht_signs, so the addresses survive CUDA-graph
                # capture.
                self._fc2_rht_sign_vector = _draw_sign_vector(_V2_RHT_SIZE, 0xFC2, dev)
                self._fc2_dgrad_rht_sign_vector = _draw_sign_vector(
                    _V2_RHT_SIZE, 0xDEAD, dev
                )

        def _grouped_mm(self, *, A, B_t, offs):
            # torchao's NVFP4 grouped MM takes the un-transposed weight B (E, N, K)
            # and uses the final dispatcher offset as the logical token bound. A may
            # have additional allocation capacity, which torchao leaves untouched.
            #
            # This is the V1 seam and stays the path for the default configuration.
            # The seam takes (A, B_t, offs) and so cannot tell the three call sites
            # apart; the per-layer recipe split therefore lives in forward() below,
            # which is the last place FC1 and FC2 are still distinguishable.
            return _to_nvfp4_rht_rs_then_scaled_grouped_mm(
                A,
                B_t.transpose(-2, -1),
                self.rht_sign_vector,
                self._sr_seed,
                offs=offs,
                pad_token_groups_for_grouped_mm=False,
                kernel_preference=self._kernel_preference,
                use_fast_math=self._use_fast_math,
            )

        def _recipe_grouped_mm(self, recipe, A, B, offs, *, is_fc2):
            """One grouped GEMM under ``recipe``. ``B`` is un-transposed (E, N, K)."""
            from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
                nvfp4_v1_requant_grouped_mm,
                nvfp4_v2_grouped_mm,
            )

            seed = self._fc2_sr_seed if is_fc2 else self._sr_seed
            if recipe == "v1":
                return _to_nvfp4_rht_rs_then_scaled_grouped_mm(
                    A,
                    B,
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
                    B,
                    sign_vector=self.rht_sign_vector,
                    sr_seed=seed,
                    offs=offs,
                    pad_token_groups_for_grouped_mm=False,
                    use_fast_math=self._use_fast_math,
                )
            return nvfp4_v2_grouped_mm(
                A,
                B,
                wgrad_rht=self._fc2_rht_sign_vector,
                dgrad_rht=self._fc2_dgrad_rht_sign_vector,
                sr_seed=seed,
                offs=offs,
                pad_token_groups_for_grouped_mm=False,
                use_fast_math=self._use_fast_math,
            )

        def forward(self, x_RD, num_tokens_per_expert_E):
            if self._fc1_recipe == "v1" and self._fc2_recipe == "v1":
                # Default configuration: fall through to the parent's forward, which
                # reaches NVFP4 through the _grouped_mm seam above. Keeping the common
                # case on the parent's code path means the DTensor/spmd preamble is
                # not duplicated and stays in sync.
                return super().forward(x_RD, num_tokens_per_expert_E)

            if isinstance(self.w1_EFD, DTensor):
                # Same reason as the parent: EP's dynamic shapes are not expressible
                # as DTensors, so the grouped GEMM runs on plain local tensors.
                w1_EFD = self.w1_EFD.to_local()
                w2_EDF = self.w2_EDF.to_local()
                w3_EFD = self.w3_EFD.to_local()
            else:
                w1_EFD, w2_EDF, w3_EFD = self.w1_EFD, self.w2_EDF, self.w3_EFD

            offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
            if (
                get_spmd_backend() == "spmd_types"
                and spmd.is_type_checking()
                and spmd_mesh_size("ep") == 1
            ):
                for axis in ("dp", "cp"):
                    spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

            # Weights go in as stored: torchao takes B un-transposed as (E, N, K),
            # the opposite of the seam's B_t.
            gate = self._recipe_grouped_mm(
                self._fc1_recipe,
                x_RD.bfloat16(),
                w1_EFD.bfloat16(),
                offsets_E,
                is_fc2=False,
            )
            up = self._recipe_grouped_mm(
                self._fc1_recipe,
                x_RD.bfloat16(),
                w3_EFD.bfloat16(),
                offsets_E,
                is_fc2=False,
            )
            h_RF = F.silu(gate) * up
            out = self._recipe_grouped_mm(
                self._fc2_recipe, h_RF, w2_EDF.bfloat16(), offsets_E, is_fc2=True
            )
            return out.type_as(x_RD)

    NVFP4GroupedExperts.__name__ = f"NVFP4{parent_cls.__name__}"
    NVFP4GroupedExperts.__qualname__ = f"NVFP4{parent_cls.__name__}"
    _nvfp4_experts_cache[parent_cls] = NVFP4GroupedExperts
    return NVFP4GroupedExperts


class NVFP4GroupedExpertsConverter(QuantizationConverter):
    """Apply NVFP4 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of experts modules to quantize. Only
        GroupedExperts.Config entries whose FQN contains a match are converted.
        If empty, all experts are converted. Pass explicit fqns (e.g. the leading
        decoder layers) to keep the tail layers' experts in bf16.
        """
        pad_multiple: int = 128
        """
        Pad per-expert token groups to this multiple for NVFP4 grouped GEMM
        alignment. TorchAO's NVFP4 kernels require multiples of 128.
        """

        kernel_preference: str = "cutedsl"
        """NVFP4 quantization backend: "auto", "triton", or "cutedsl".

        Defaults to "cutedsl" so that throughput runs are pinned by default.
        The grouped path resolves per op, so "auto" can mix backends within one
        step; "cutedsl" cannot, and raises if the runtime is missing or the
        local expert count exceeds torchao's MAX_GROUPS (64).
        """

        use_fast_math: bool = True
        """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

        fc1_recipe: Literal["v1", "v1_requant", "v2"] = "v1"
        """NVFP4 recipe for the FC1 gate/up GEMMs (``w1``, ``w3``).

        Defaults to "v1" so an unchanged config is unaffected by this option's
        existence. Design doc §17 recommends "v1_requant" here and "v2" for FC2.
        Both non-v1 recipes require kernel_preference="triton".
        """

        fc2_recipe: Literal["v1", "v1_requant", "v2"] = "v1"
        """NVFP4 recipe for the FC2 down GEMM (``w2``). See ``fc1_recipe``."""

    def __init__(self, config: Config):
        self.config = config

        # NVFP4Linear is None iff the whole torchao NVFP4 prototype import block
        # (which also provides _to_nvfp4_then_scaled_grouped_mm used by the seam)
        # failed, so this is the correct guard rather than a bare find_spec.
        if NVFP4Linear is None:
            raise ImportError(
                "torchao is not installed or does not provide the NVFP4 training "
                "prototype. Install a torchao build with "
                "torchao.prototype.moe_training.nvfp4_training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("NVFP4 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of NVFP4 dynamic quantization."
            )

        self._kernel_preference = _to_kernel_preference(self.config.kernel_preference)
        _log_kernel_preference(
            f"NVFP4GroupedExperts fc1={self.config.fc1_recipe} "
            f"fc2={self.config.fc2_recipe}",
            self._kernel_preference,
            self.config.use_fast_math,
            per_op=True,
        )

    def convert(self, model_config):
        fqns = self.config.fqns
        num_experts = 0
        for fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            if fqns and not any(target_fqn in fqn for target_fqn in fqns):
                continue
            num_experts = max(num_experts, config.num_experts)
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_nvfp4_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)}
            )
            # The comprehension copies the parent GroupedExperts.Config fields
            # only, so the two NVFP4-specific ones are set explicitly.
            new_config.kernel_preference = self.config.kernel_preference
            new_config.use_fast_math = self.config.use_fast_math
            new_config.fc1_recipe = self.config.fc1_recipe
            new_config.fc2_recipe = self.config.fc2_recipe
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        if num_experts and self._kernel_preference is KernelPreference.CUTEDSL:
            # The CuteDSL grouped RHT kernel caps the group count, and the count it
            # sees is per rank: num_experts // expert_parallel_degree. That degree
            # lives in ParallelismConfig, which a converter never receives, so the
            # requirement can only be stated here -- torchao raises on the exact
            # local count at the first grouped GEMM.
            from torchao.prototype.moe_training.nvfp4_training._cutedsl_group_kernels_impl import (
                MAX_GROUPS,
            )

            if num_experts > MAX_GROUPS:
                logger.warning(
                    "NVFP4GroupedExperts kernel_preference=cutedsl with %d experts "
                    "requires expert_parallel_degree >= %d: the CuteDSL grouped "
                    "kernel takes at most %d experts per rank and raises otherwise.",
                    num_experts,
                    math.ceil(num_experts / MAX_GROUPS),
                    MAX_GROUPS,
                )

        logger.info(
            "Converted GroupedExperts to use dynamic NVFP4 quantization for "
            "grouped_mm ops"
        )
        return model_config
