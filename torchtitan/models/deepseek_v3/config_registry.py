# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.quantization import (
    Float8GroupedExpertsConverter,
    Float8LinearConverter,
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
    NVFP4GroupedExpertsConverter,
    NVFP4LinearConverter,
)
from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.trainer import Trainer

from . import model_registry

# NVFP4 needs every GEMM dim to be a multiple of 128. In DSV3 that admits the
# dense FeedForward and the MoE shared experts, but not the MLA projections
# (wkv_a is dim -> kv_lora_rank + qk_rope_head_dim = 576) or the router gate
# (dim -> num_experts), so the fqns name FFN submodules explicitly rather than
# whole layers.
_NVFP4_FFN_SUBMODULES = ("feed_forward.", "moe.shared_experts.")
# 16B's dense FeedForward is dim=2048 -> dense_hidden_dim=10944, and
# 10944 % 128 == 64, so NVFP4Linear.Config rejects it. Its one dense layer
# (layer 0) stays bf16; the MoE layers' shared experts still convert.
_NVFP4_FFN_SUBMODULES_NO_DENSE = ("moe.shared_experts.",)


def _nvfp4_ffn_linear_fqns(
    layer_fqns: list[str], submodules: tuple[str, ...]
) -> list[str]:
    """Cross-product leading-layer prefixes with FFN submodule paths.

    ``layer_fqns`` comes from ``nvfp4_bf16_tail_fqns``, so the bf16 tail is
    single-sourced with the grouped-experts converter's fqns.
    """
    return [f"{layer}{submodule}" for layer in layer_fqns for submodule in submodules]


def enable_fused_swiglu(config: Trainer.Config) -> None:
    # fused_swiglu.py registers two overrides (dense FeedForward + MoE grouped
    # experts); activate both by naming each factory.
    for override in (
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
    ):
        assert override not in config.override.imports
        config.override.imports.append(override)


def deepseek_v3_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def deepseek_v3_debugmodel_mxfp8() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    # Quantize the MoE expert grouped GEMMs to MXFP8, plus the dense Linear
    # layers in attention, the shared experts, and the dense-layer feed-forward.
    # fqns is an include-list (substring match), so the MoE router gate
    # (moe.router.gate) and lm_head are left in bf16.
    # pad_multiple=128 is required by the CuTeDSL quantization kernel
    # on sm_100 (e.g. B200)
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            MXFP8LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=["attention", "shared_experts", "feed_forward"],
            ),
            MXFP8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_debugmodel_nvfp4() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    assert config.model_spec is not None
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # Quantize every FFN GEMM to NVFP4: the MoE expert grouped GEMMs, the MoE
    # shared experts, and the dense layers' FeedForward. Convert the leading
    # decoder layers and keep the last _NVFP4_BF16_TAIL_FRACTION of layers in
    # bf16. Attention stays bf16 everywhere -- DSV3's MLA has projections (e.g.
    # wkv_a, out=576) whose dims are not divisible by 128, which NVFP4's Triton
    # kernels require.
    # pad_multiple=128 is required by the NVFP4 grouped-mm kernel on sm_100.
    n_layers = len(config.model_spec.model.layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.15
    fqns = nvfp4_bf16_tail_fqns(n_layers, _NVFP4_BF16_TAIL_FRACTION)
    config.model_spec = model_registry(
        "debugmodel",
        converters=[
            NVFP4LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=_nvfp4_ffn_linear_fqns(fqns, _NVFP4_FFN_SUBMODULES),
            ),
            NVFP4GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=fqns,
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_debugmodel_hybridep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def deepseek_v3_debugmodel_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_spec = model_registry(
        "debugmodel",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    return config


def deepseek_v3_16b() -> Trainer.Config:
    model_spec = model_registry("16B", attn_backend="flex")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/deepseek-moe-16b-base",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=1000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=8,
        ),
        checkpoint=CheckpointManager.Config(interval=10),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_16b_hybridep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=1.0,
    )
    return config


def deepseek_v3_16b_nvfp4() -> Trainer.Config:
    config = deepseek_v3_16b()
    assert config.model_spec is not None
    # Assign compile BEFORE deriving the flag, matching
    # deepseek_v3_671b_nvfp4_mixed(). deepseek_v3_16b() sets compile=["loss"], so
    # deriving from the inherited value gave False and the NVFP4 quantize around
    # each grouped GEMM never fused. Deriving before assigning would be worse
    # still: the model would compile while the converters believe they run eager.
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    # Match the bf16 and mxfp8 arms of the 16B convergence campaign. The default
    # is 1/5 of upstream's steps=1000; the campaign runs 1500 and deliberately
    # keeps 200, so a 100-step stable phase survives. All arms must share it.
    config.lr_scheduler.warmup_steps = 200
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # Same recipe as deepseek_v3_debugmodel_nvfp4 at 16B scale, with one
    # exception: 16B's dense FeedForward is 2048 -> 10944, and 10944 % 128 == 64,
    # so it cannot be NVFP4 and layer 0 stays entirely bf16. The MoE layers'
    # grouped experts and shared experts convert for the leading decoder layers;
    # attention stays bf16. pad_multiple=128 is required by the NVFP4 grouped-mm
    # kernel on sm_100.
    #
    # F0L0: no bf16 tail. Was 0.15, i.e. ceil(27 * 0.15) = 5 trailing layers held
    # in bf16 (F0L5). MLPerf's DSV3-671B FP4 config sets neither
    # num_layers_at_start_in_bf16 nor num_layers_at_end_in_bf16, so F0L0 is the
    # reference recipe, and the 671B config here moves with it -- this arm exists
    # to validate what 671B will actually run, so a more conservative holdout at
    # 16B would test something we do not ship.
    #
    # ceil(0) = 0 leaves convert_upto = n_layers, which does NOT trip the
    # convert_upto <= 0 guard in nvfp4_bf16_tail_fqns; layer 0 is still excluded
    # separately because 16B's dense FeedForward is 2048 -> 10944 and
    # 10944 % 128 == 64, so it cannot be NVFP4.
    n_layers = len(config.model_spec.model.layers)
    _NVFP4_BF16_TAIL_FRACTION = 0.0
    fqns = nvfp4_bf16_tail_fqns(n_layers, _NVFP4_BF16_TAIL_FRACTION)
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        # HybridEP with a bounded buffer, matching the bf16 and mxfp8 arms.
        # Omitting these ran the blocking dropless path, which is unbounded under
        # real routing -- that is what killed bf16 round 1 (148 -> 214 GiB over
        # 40 steps). cf 0.1875 is 2x balanced at ep-32: 6 / (32 x 2) = 0.09375.
        # The 671B configs' 0.03125 does not port; 16B has 2 local experts
        # against 671B's 8.
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=0.1875,
        converters=[
            # TEST-ONLY BRANCH: kernel_preference pinned to cutedsl. Should
            # reproduce the AUTO cell exactly, since AUTO resolved to cutedsl on
            # this image, and unlike AUTO it raises rather than silently falling
            # back if the runtime is missing. Never merged.
            NVFP4LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=_nvfp4_ffn_linear_fqns(fqns, _NVFP4_FFN_SUBMODULES_NO_DENSE),
                kernel_preference="cutedsl",
            ),
            NVFP4GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=fqns,
                pad_multiple=128,
                kernel_preference="cutedsl",
            ),
            # Attention in MXFP8, matching deepseek_v3_671b_nvfp4_mixed. This arm
            # exists to de-risk the 671B recipe, so the set of quantized modules
            # has to be the same or its convergence result does not transfer.
            # Bare fqns, no leading-layer prefix, so this covers every layer
            # including any the NVFP4 tail would have skipped -- at F0L0 there is
            # no tail, so both converters now span all 27 layers.
            #
            # 16B has q_lora_rank=0, so attention.wq is a single 2048->3072
            # Linear rather than 671B's wq_a/wq_b pair; the same bare fqns match
            # it by substring. wkv_a/wkv_b stay bf16, as at 671B.
            MXFP8LinearConverter.Config(
                fqns=["attention.wq", "attention.wo"],
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def deepseek_v3_16b_minimal_async_ep() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_spec = model_registry(
        "16B",
        attn_backend="flex",
        moe_comm_backend="minimal_async_ep",
    )
    enable_fused_swiglu(config)
    config.parallelism = ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
        expert_parallel_degree=1,
        enable_sequence_parallel=False,
    )
    return config


def deepseek_v3_671b() -> Trainer.Config:
    model_spec = model_registry(
        "671B",
        attn_backend="flex",
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/DeepSeek-V3.1-Base",
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        optimizer=default_adamw(lr=2.2e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2000,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10000,
        ),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
            expert_parallel_degree=2,
        ),
        checkpoint=CheckpointManager.Config(interval=500),
        activation_checkpoint=SelectiveAC.Config(),
        compile=CompileConfig(enable=True, components=["loss"]),
    )


def deepseek_v3_671b_12_layers_nvfp4_mixed() -> Trainer.Config:
    config = deepseek_v3_671b()
    assert config.model_spec is not None
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    n_layers = 12
    fqns = nvfp4_bf16_tail_fqns(n_layers, 0.15)
    config.model_spec = model_registry(
        "671B_12_layers",
        attn_backend="flex",
        moe_comm_backend="hybridep",
        non_blocking_capacity_factor=0.0625,
        converters=[
            NVFP4LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=_nvfp4_ffn_linear_fqns(fqns, _NVFP4_FFN_SUBMODULES),
            ),
            NVFP4GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=fqns,
                pad_multiple=128,
            ),
        ],
    )
    return config


def deepseek_v3_671b_nvfp4_mixed() -> Trainer.Config:
    config = deepseek_v3_671b()
    assert config.model_spec is not None
    config.compile = CompileConfig(enable=True, components=["model", "loss"])
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    # F0L0: no bf16 tail. Was 0.15, i.e. ceil(61 * 0.15) = 10 trailing layers in
    # bf16 -- exactly the deck's F0L10 row. MLPerf's DSV3-671B FP4 config sets
    # neither num_layers_at_start_in_bf16 nor num_layers_at_end_in_bf16, so F0L0
    # is the reference recipe.
    #
    # CAVEAT, not yet tested here: MLPerf pairs F0L0 with BF16_PROJ=True (all MLA
    # projections in BF16) and an MXFP8 block-scaled DPA. This branch instead puts
    # wq_a/wq_b/wo in MXFP8, leaves wkv_a/wkv_b bf16, and does not quantize the
    # DPA. The Kitchen deck records "F0L0, but BF16 linear for projections in all
    # MLA improves convergence significantly", so MLPerf's convergence result
    # covers F0L0 + bf16 MLA projections, not this combination. Empirically the
    # mxfp8-attn cell is the loss-cleanest nvfp4 cell measured (mean +0.0099 vs
    # bf16, slope -0.0033, 0.19% relative), so the projections are not hurting at
    # a 340-step horizon -- but F0L0 + mxfp8 projections is a new cell.
    n_layers = len(config.model_spec.model.layers)
    fqns = nvfp4_bf16_tail_fqns(n_layers, 0.0)
    config.model_spec = model_registry(
        "671B",
        attn_backend="flex",
        moe_comm_backend="hybridep",
        # Sized to exactly the round-robin demand (32768 tokens/rank x 64 ep
        # x min(4, 8) x 0.03125 = 262144 rows = 65536 per local expert, a
        # multiple of 128). Per-rank demand is tokens_per_rank x top_k
        # independent of ep, so top_k/256 = 0.03125 is the exact fit for any
        # ep >= 32; it only changes at ep-16, to 0.0625.
        #
        # This is a no-waste default. It is no longer a constraint: the earlier
        # 12-layer NaN at a larger capacity factor was the offs[-1] = A.shape[0]
        # rewrite extending the last expert group over the whole capacity
        # buffer, so the spare rows fed that group's global amax. The NVFP4
        # kernels do mask loads against logical_packed_length, but that bound is
        # padded_group_end_offsets[-1], so the rewrite made the mask vacuous --
        # which is why the masking looked like it ruled the mechanism out.
        # Fixed by torchtitan 791f2415 + torchao 14555766 and verified at the
        # unchanged failing cf 0.1875, so exact fit is kept here only because it
        # wastes nothing, not because raising it is unsafe.
        non_blocking_capacity_factor=0.03125,
        converters=[
            NVFP4LinearConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=_nvfp4_ffn_linear_fqns(fqns, _NVFP4_FFN_SUBMODULES),
            ),
            NVFP4GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled,
                fqns=fqns,
                pad_multiple=128,
            ),
            # Attention in MXFP8, not NVFP4. The shapes would permit NVFP4 --
            # wq_a 7168->1536, wq_b 1536->24576 and wo 16384->7168 are all
            # 128-divisible; only wkv_a (7168->576) is not -- but the NVFP4
            # recipe is not the right instrument for attention, so the format is
            # chosen per module rather than per shape. This is what makes the
            # flavor name "mixed" literal.
            #
            # Bare fqns, no leading-layer prefix, so this matches the fqns of
            # deepseek_v3_671b_mxfp8_exp exactly and covers all layers including
            # the bf16 tail. Deliberate: it gives module-level coverage parity
            # with the MXFP8 baseline, so a throughput comparison against it
            # isolates NVFP4-vs-MXFP8 on the FFN and expert GEMMs instead of
            # measuring how much of the model each flavor happens to convert.
            # Consequence: the tail layers are bf16 only for FFN and experts,
            # not for attention.
            MXFP8LinearConverter.Config(
                fqns=["attention.wq", "attention.wo"],
                model_compile_enabled=model_compile_enabled,
            ),
        ],
    )
    return config


def deepseek_v3_671b_float8() -> Trainer.Config:
    config = deepseek_v3_671b()
    # Quantize the dense Linear layers and the MoE expert grouped GEMMs to
    # float8 (fp8). This requires torchao and is only supported on NVIDIA SM89+
    # or AMD MI300+; on other backends (e.g. Intel XPU) the converter raises at
    # build time, so use the plain deepseek_v3_671b config there.
    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    config.model_spec = model_registry(
        "671B",
        attn_backend="flex",
        converters=[
            Float8LinearConverter.Config(
                filter_fqns=["lm_head", "router.gate"],
                model_compile_enabled=model_compile_enabled,
            ),
            Float8GroupedExpertsConverter.Config(
                model_compile_enabled=model_compile_enabled
            ),
        ],
    )
    return config
