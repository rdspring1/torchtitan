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
from typing import cast

import spmd_types as spmd
import torch

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, SpmdLayout
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

TP = MeshAxisName.TP

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

try:
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
        _to_nvfp4_rht_rs_then_scaled_grouped_mm,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
        nvfp4_linear,
        nvfp4_mm_triton,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
        cutedsl_nvfp4_kernels_available,
        cutedsl_nvfp4_unavailable_reason,
    )
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

    def _log_kernel_preference(what: str, pref: KernelPreference) -> None:
        """Record the preference and the backend it actually resolves to.

        AUTO is legitimate, but it must not be silent: the resolved backend is
        what a later comparison against these numbers has to match.
        """
        if pref is KernelPreference.CUTEDSL and not cutedsl_nvfp4_kernels_available():
            raise RuntimeError(
                f"{what} kernel_preference=cutedsl, but the CuteDSL runtime is "
                f"unavailable ({cutedsl_nvfp4_unavailable_reason()})."
            )
        resolved = (
            "triton"
            if pref is KernelPreference.TRITON
            else ("cutedsl" if cutedsl_nvfp4_kernels_available() else "triton")
        )
        logger.info(
            "%s kernel_preference=%s, resolved backend=%s", what, pref.value, resolved
        )

    # The NVFP4 GEMM is a raw autograd Function that runs on local shards inside
    # the spmd.local_map region. Mark it local-safe so SPMD type checking
    # propagates through it; the local_map boundary declares the real
    # colwise/rowwise output and input-gradient types.
    spmd.register_local_autograd_function(nvfp4_mm_triton)

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

            kernel_preference: str = "auto"
            """NVFP4 quantization backend: "auto", "triton", or "cutedsl"."""

            use_fast_math: bool = True
            """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

            def __post_init__(self) -> None:
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

        def __init__(self, config: Linear.Config):
            TorchAONVFP4Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
            )
            # Resolved once here rather than per forward: _resolve_use_cutedsl
            # walks importlib.util.find_spec over four packages, and forward and
            # backward must agree on the backend.
            self._kernel_preference = _to_kernel_preference(config.kernel_preference)
            self._use_fast_math = config.use_fast_math
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
            self._rht_sign_vector_tuple = None

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
            # Static RHT basis: identical on every rank by construction, so it is
            # a plain local tensor with no cross-rank broadcast.
            self._rht_sign_vector = _make_rht_sign_vector(
                _HARDCODED_SIGN_VECTOR, device=dev
            )
            self._refresh_rht_sign_vector_tuple()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nvfp4_linear(
                x,
                self.weight,
                self.bias,
                sr_seed=self._sr_seed,
                sign_vector=self.rht_sign_vector,
                kernel_preference=self._kernel_preference,
                use_fast_math=self._use_fast_math,
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

        kernel_preference: str = "auto"
        """NVFP4 quantization backend: "auto", "triton", or "cutedsl".

        Pin this rather than leaving it at "auto" for any run whose throughput is
        compared against another: "auto" resolves to CuteDSL when its runtime is
        importable and Triton otherwise, so the same recipe measures different
        kernels on containers that differ only in which packages are installed.
        """

        use_fast_math: bool = True
        """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

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

        self._kernel_preference = _to_kernel_preference(self.config.kernel_preference)
        _log_kernel_preference("NVFP4Linear", self._kernel_preference)

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
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to NVFP4Linear")
        return model_config


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
            kernel_preference: str = "auto"
            """NVFP4 quantization backend: "auto", "triton", or "cutedsl"."""

            use_fast_math: bool = True
            """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

        def __init__(self, config: Config):
            super().__init__(config)
            self._kernel_preference = _to_kernel_preference(config.kernel_preference)
            self._use_fast_math = config.use_fast_math
            # Same buffer protocol as NVFP4Linear.__init__: register the runtime
            # buffers as None so _distribute_states skips them and
            # _init_self_buffers materializes them per rank on the real device.
            self.register_buffer("_sr_seed", None, persistent=False)
            self.register_buffer("_rht_sign_vector", None, persistent=False)
            self._rht_sign_vector_tuple = None

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

        def _grouped_mm(self, *, A, B_t, offs):
            # torchao's NVFP4 grouped MM takes the un-transposed weight B (E, N, K)
            # and uses the final dispatcher offset as the logical token bound. A may
            # have additional allocation capacity, which torchao leaves untouched.
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
        alignment. TorchAO's NVFP4 Triton kernels require multiples of 128.
        """

        kernel_preference: str = "auto"
        """NVFP4 quantization backend: "auto", "triton", or "cutedsl".

        The grouped path resolves per op, so "auto" can mix backends within one
        step. Pin it for any run whose throughput is compared against another.
        """

        use_fast_math: bool = True
        """Approximate-reciprocal RHT quantize, matching TE's NVTE_USE_FAST_MATH=1."""

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
        _log_kernel_preference("NVFP4GroupedExperts", self._kernel_preference)

    def convert(self, model_config):
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            if fqns and not any(target_fqn in fqn for target_fqn in fqns):
                continue
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
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            "Converted GroupedExperts to use dynamic NVFP4 quantization for "
            "grouped_mm ops"
        )
        return model_config
