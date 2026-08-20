# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import spmd_types as spmd
import torch

from torchtitan.components.quantization import Float8Linear
from torchtitan.components.quantization.float8 import _get_float8_grouped_experts_cls
from torchtitan.components.quantization.mx import _get_mxfp8_grouped_experts_cls
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import ConfigManager
from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    model_config = config.model_spec.model
    assert not has_quantization(model_config)
    # All Linear.Config instances should remain Linear.Config
    if Float8Linear is not None:
        for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
            assert not isinstance(lc, Float8Linear.Config)


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_float8_emulate_lora"]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)
    # Some Linear.Config instances should be swapped to Float8Linear
    converted = [
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, Float8Linear.Config)
    ]
    assert len(converted) > 0


@pytest.mark.parametrize(
    "module, recipe, expected_num_layers",
    [
        ("llama3", "llama3_debugmodel_nvfp4", 6),
        ("qwen3", "qwen3_debugmodel_nvfp4", 8),
    ],
)
def test_nvfp4_converter_targets_layers_not_lm_head(
    monkeypatch, module, recipe, expected_num_layers
):
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    # Exercise convert() targeting independent of GPU: bypass the sm100 gate
    # that NVFP4LinearConverter.__init__ enforces (hardware is irrelevant to the
    # config-tree transform under test).
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)

    config_manager = ConfigManager()
    config = config_manager.parse_args(["--module", module, "--config", recipe])
    model_config = config.model_spec.model
    assert has_quantization(model_config)

    converted, stock = [], []
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        (converted if isinstance(lc, NVFP4Linear.Config) else stock).append(fqn)

    # Every in-layer linear is swapped; the lm_head stays stock (NVFP4 requires
    # each GEMM dim divisible by 128, which the vocab projection violates).
    assert converted and all("layers" in fqn for fqn in converted)
    assert {int(fqn.split(".")[1]) for fqn in converted} == set(
        range(expected_num_layers)
    )
    assert stock == ["lm_head"]


def test_nvfp4_bf16_tail_fqns():
    from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns

    # 32 layers, 15% tail -> ceil(4.8)=5 bf16, convert layers 0..26.
    fqns = nvfp4_bf16_tail_fqns(32, 0.15)
    assert fqns == [f"layers.{i}." for i in range(27)]
    # Every fqn is trailing-dot anchored so "layers.2." matches layer 2 only,
    # not "layers.20".."layers.29" (the converter substring-matches).
    assert all(f.startswith("layers.") and f.endswith(".") for f in fqns)
    # Fraction 0 keeps nothing in bf16 -> every layer converted.
    assert nvfp4_bf16_tail_fqns(4, 0.0) == [
        "layers.0.",
        "layers.1.",
        "layers.2.",
        "layers.3.",
    ]
    # A fraction that rounds up to all layers leaves nothing to convert -> raise
    # (an empty fqns list would instead convert *all* Linears).
    with pytest.raises(ValueError, match="nothing to convert"):
        nvfp4_bf16_tail_fqns(4, 1.0)


@pytest.mark.parametrize(
    "module, recipe, expected_cutoff",
    [
        ("llama3", "llama3_debugmodel_first_85_pct_layers_nvfp4", 5),
        ("llama3", "llama3_8b_first_85_pct_layers_nvfp4", 27),
        ("qwen3", "qwen3_debugmodel_first_85_pct_layers_nvfp4", 6),
        ("qwen3", "qwen3_8b_first_85_pct_layers_nvfp4", 30),
    ],
)
def test_nvfp4_first_85_pct_layers_converts_only_leading_layers(
    monkeypatch, module, recipe, expected_cutoff
):
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    import math

    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)

    config = ConfigManager().parse_args(["--module", module, "--config", recipe])
    model_config = config.model_spec.model
    n_layers = len(model_config.layers)
    cutoff = n_layers - math.ceil(n_layers * 0.15)
    assert cutoff == expected_cutoff
    assert 0 < cutoff < n_layers  # a real split: some NVFP4, some bf16

    converted_layers, stock = set(), []
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        if isinstance(lc, NVFP4Linear.Config):
            converted_layers.add(int(fqn.split(".")[1]))
        else:
            stock.append(fqn)

    # Only the leading layers are NVFP4; the bf16 tail + lm_head stay stock.
    assert converted_layers == set(range(cutoff))
    assert "lm_head" in stock
    assert all(
        not fqn.startswith("layers.") or int(fqn.split(".")[1]) >= cutoff
        for fqn in stock
    )


def _nvfp4_linear_cls():
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    return NVFP4Linear


@pytest.mark.parametrize("in_features, out_features", [(512, 300), (300, 512)])
def test_nvfp4_config_rejects_non_128_dims(in_features, out_features):
    # The model dims are known at config-build time, so a non-128 in/out_features
    # (e.g. the LM head) is rejected in Config.__post_init__ before any TP.
    NVFP4Linear = _nvfp4_linear_cls()
    with pytest.raises(ValueError, match="divisible by 128"):
        NVFP4Linear.Config(in_features=in_features, out_features=out_features)


@pytest.mark.parametrize(
    "sharding_config_factory, input_tp, input_grad_tp",
    [
        pytest.param(lambda: colwise_config(), spmd.R, spmd.P, id="colwise"),
        pytest.param(
            lambda: rowwise_config(output_sp=True),
            spmd.S(-1),
            spmd.S(-1),
            id="rowwise",
        ),
    ],
)
def test_nvfp4_build_configures_local_spmd_sharding(
    sharding_config_factory, input_tp, input_grad_tp
):
    # Config.build() folds the stock colwise/rowwise sharding into the local
    # SPMD region for the opaque NVFP4 GEMM.
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
    from torchtitan.models.common.decoder_sharding import dense_activation_placement

    module = NVFP4Linear.Config(
        in_features=512,
        out_features=1024,
        sharding_config=sharding_config_factory(),
    ).build()
    sc = module._sharding_config
    assert sc.local_map is not None
    input_layout = dense_activation_placement(tp=input_tp)
    assert sc.in_src_shardings == {"x": input_layout}
    assert sc.in_dst_shardings == {"x": input_layout}
    assert sc.local_map.in_grad_placements == (
        dense_activation_placement(tp=input_grad_tp),
    )
    assert "weight" in sc.state_shardings
    assert sc.state_shardings["_sr_seed"] == SpmdLayout(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.V,
        }
    )


@pytest.mark.parametrize(
    "module, recipe",
    [
        ("llama3", "llama3_debugmodel_nvfp4"),
        ("llama3", "llama3_debugmodel_first_85_pct_layers_nvfp4"),
        ("llama3", "llama3_8b_first_85_pct_layers_nvfp4"),
        ("qwen3", "qwen3_debugmodel_nvfp4"),
        ("qwen3", "qwen3_debugmodel_first_85_pct_layers_nvfp4"),
        ("qwen3", "qwen3_8b_first_85_pct_layers_nvfp4"),
    ],
)
def test_nvfp4_recipes_default_to_spmd_types_and_allow_cli_override(
    monkeypatch, module, recipe
):
    _nvfp4_linear_cls()
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    base_args = ["--module", module, "--config", recipe]

    config = ConfigManager().parse_args(base_args)
    assert config.parallelism.spmd_backend == "spmd_types"

    overridden = ConfigManager().parse_args(
        [*base_args, "--parallelism.spmd_backend", "default"]
    )
    assert overridden.parallelism.spmd_backend == "default"


@pytest.mark.parametrize(
    "recipe",
    [
        "qwen3_debugmodel_nvfp4",
        "qwen3_debugmodel_first_85_pct_layers_nvfp4",
        "qwen3_8b_first_85_pct_layers_nvfp4",
    ],
)
def test_qwen3_recipes_resolve(monkeypatch, recipe):
    _nvfp4_linear_cls()
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    config = ConfigManager().parse_args(["--module", "qwen3", "--config", recipe])
    assert config.model_spec.name == "qwen3"
    if recipe == "qwen3_8b_first_85_pct_layers_nvfp4":
        assert config.dataloader.dataset_path == "openai/gsm8k"
        assert config.checkpoint.initial_load_in_hf
        assert config.compile.enable
        assert "model" in config.compile.components


def test_nvfp4_module_buffers_and_native_checkpoint():
    """Built module has the stock weight param plus the two NVFP4 runtime
    buffers, and both buffers are non-persistent -- the RHT vector is a fixed
    constant and the SR seed is per-rank -- so a native checkpoint carries only
    the stock weight."""
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.components.quantization.nvfp4 import _HARDCODED_SIGN_VECTOR

    module = NVFP4Linear.Config(in_features=512, out_features=1024).build()
    assert {name for name, _ in module.named_parameters()} == {"weight"}
    module.init_states()
    buffers = dict(module.named_buffers())
    assert set(buffers) == {"_sr_seed", "_rht_sign_vector"}
    assert buffers["_sr_seed"].dtype == torch.int64
    assert tuple(buffers["_rht_sign_vector"].shape) == (16,)
    # The RHT vector is the fixed v1-recipe constant, identical on every rank.
    assert tuple(int(v) for v in buffers["_rht_sign_vector"]) == _HARDCODED_SIGN_VECTOR
    # Both runtime buffers are non-persistent, so a native checkpoint carries
    # only the stock weight.
    assert set(module.state_dict()) == {"weight"}


def test_nvfp4_stock_checkpoint_loads_before_init_states():
    """A stock bf16 checkpoint (no NVFP4 buffers) loads; buffers stay unmaterialized
    until init_states creates them."""
    NVFP4Linear = _nvfp4_linear_cls()
    stock = Linear.Config(in_features=512, out_features=1024).build()
    nvfp4 = NVFP4Linear.Config(in_features=512, out_features=1024).build()

    nvfp4.load_state_dict(stock.state_dict(), strict=False)
    assert nvfp4._rht_sign_vector is None
    assert nvfp4._rht_sign_vector_tuple is None

    nvfp4.init_states()
    assert nvfp4._rht_sign_vector is not None
    assert nvfp4._rht_sign_vector_tuple is not None


def test_nvfp4_hf_export_strips_buffers(monkeypatch):
    """The HF export boundary contains only stock keys -- no NVFP4 runtime buffers."""
    NVFP4Linear = _nvfp4_linear_cls()
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

    config = ConfigManager().parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_nvfp4"]
    )
    model_config = config.model_spec.model
    model = model_config.build()
    model.init_states()
    assert isinstance(model.get_submodule("layers.0.feed_forward.w1"), NVFP4Linear)

    sd = model.state_dict()
    # Both NVFP4 runtime buffers are non-persistent, so neither the RHT vector
    # nor the per-rank SR seed appears in the native state dict.
    assert not any("_rht_sign_vector" in k for k in sd)
    assert not any("_sr_seed" in k for k in sd)

    hf_sd = Llama3StateDictAdapter(model_config, hf_assets_path=None).to_hf(sd)
    assert "model.layers.0.mlp.gate_proj.weight" in hf_sd
    assert not any("_rht_sign_vector" in k for k in hf_sd)


def test_quantized_grouped_experts():
    """Quantized GroupedExperts: _owner, subclass handling, extra config fields."""
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear
    from torchtitan.components.quantization.nvfp4 import _get_nvfp4_grouped_experts_cls

    # Base case
    MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)
    Float8GroupedExperts = _get_float8_grouped_experts_cls(GroupedExperts)

    assert MXFP8GroupedExperts.Config._owner is MXFP8GroupedExperts
    assert Float8GroupedExperts.Config._owner is Float8GroupedExperts

    # Subclass case (GptOssGroupedExperts has extra swiglu_limit field)
    mxfp8_cls = _get_mxfp8_grouped_experts_cls(GptOssGroupedExperts)
    float8_cls = _get_float8_grouped_experts_cls(GptOssGroupedExperts)

    assert mxfp8_cls.Config._owner is mxfp8_cls
    assert float8_cls.Config._owner is float8_cls
    assert issubclass(mxfp8_cls, GptOssGroupedExperts)
    assert issubclass(float8_cls, GptOssGroupedExperts)
    assert hasattr(mxfp8_cls.Config, "swiglu_limit")
    assert hasattr(float8_cls.Config, "swiglu_limit")

    # NVFP4 mirrors the MXFP8/Float8 pattern: _owner is wired, the subclass
    # overrides _grouped_mm, and it works for GptOssGroupedExperts too. Skip if
    # the torchao NVFP4 training prototype is unavailable.
    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    nvfp4_cls = _get_nvfp4_grouped_experts_cls(GroupedExperts)
    assert nvfp4_cls.Config._owner is nvfp4_cls
    assert issubclass(nvfp4_cls, GroupedExperts)
    assert nvfp4_cls._grouped_mm is not GroupedExperts._grouped_mm
    # Cached: repeated calls return the same class.
    assert _get_nvfp4_grouped_experts_cls(GroupedExperts) is nvfp4_cls

    nvfp4_gptoss_cls = _get_nvfp4_grouped_experts_cls(GptOssGroupedExperts)
    assert nvfp4_gptoss_cls.Config._owner is nvfp4_gptoss_cls
    assert issubclass(nvfp4_gptoss_cls, GptOssGroupedExperts)
    assert hasattr(nvfp4_gptoss_cls.Config, "swiglu_limit")


def test_nvfp4_grouped_experts_preserves_logical_tail_offset(monkeypatch):
    """Dispatcher allocation capacity must not extend the final expert."""
    pytest.importorskip("torchao")
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    if nvfp4_mod.NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")

    forwarded = {}

    def grouped_mm_stub(*args, **kwargs):
        forwarded["offs"] = kwargs["offs"]
        return args[0]

    monkeypatch.setattr(
        nvfp4_mod, "_to_nvfp4_rht_rs_then_scaled_grouped_mm", grouped_mm_stub
    )
    NVFP4Experts = nvfp4_mod._get_nvfp4_grouped_experts_cls(GroupedExperts)

    class RuntimeState:
        rht_sign_vector = (1,) * 16
        _sr_seed = torch.zeros(1, dtype=torch.int64)

    A = torch.empty(384, 128)
    B_t = torch.empty(2, 128, 128)
    offs = torch.tensor([128, 256], dtype=torch.int32)

    NVFP4Experts._grouped_mm(RuntimeState(), A=A, B_t=B_t, offs=offs)

    assert forwarded["offs"] is offs
    assert offs[-1] < A.shape[0]


def test_nvfp4_grouped_experts_converter_targets_leading_moe_layers(monkeypatch):
    """The DSV3 NVFP4 recipe swaps only the leading-85% MoE layers' experts to the
    NVFP4 subclass and swaps their dispatcher to TorchAOTokenDispatcher(pad=128);
    the bf16-tail MoE layer keeps stock GroupedExperts and its original dispatcher.
    """
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear
    from torchtitan.components.quantization.nvfp4 import _get_nvfp4_grouped_experts_cls
    from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    # Config-tree transform is GPU-independent: bypass the sm100 gate the
    # converter __init__ enforces.
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)

    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "deepseek_v3", "--config", "deepseek_v3_debugmodel_nvfp4"]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)

    NVFP4Experts = _get_nvfp4_grouped_experts_cls(GroupedExperts)
    converted, stock = [], []
    for fqn, cfg, parent, _attr in model_config.traverse(GroupedExperts.Config):
        if isinstance(cfg, NVFP4Experts.Config):
            converted.append(fqn)
            # The sibling dispatcher was swapped to the padded TorchAO variant.
            assert isinstance(parent.token_dispatcher, TorchAOTokenDispatcher.Config)
            assert parent.token_dispatcher.pad_multiple == 128
        else:
            stock.append(fqn)
            assert not isinstance(
                parent.token_dispatcher, TorchAOTokenDispatcher.Config
            )

    # debugmodel: 6 layers, 1 dense -> MoE in layers 1..5; 15% tail keeps layer 5
    # in bf16, so layers 1..4 convert.
    assert {int(fqn.split(".")[1]) for fqn in converted} == {1, 2, 3, 4}
    assert {int(fqn.split(".")[1]) for fqn in stock} == {5}

    # The same recipe converts every FFN Linear in those layers -- the dense
    # layer's FeedForward and the MoE shared experts -- and nothing else. An
    # exact set catches both over-matching (attention or router.gate, whose dims
    # are not 128-divisible) and under-matching (the whole-layer fqns leaking in).
    converted_linear = {
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, NVFP4Linear.Config)
    }
    assert converted_linear == {
        f"layers.0.feed_forward.{w}" for w in ("w1", "w2", "w3")
    } | {
        f"layers.{i}.moe.shared_experts.{w}"
        for i in (1, 2, 3, 4)
        for w in ("w1", "w2", "w3")
    }


@pytest.mark.parametrize(
    "recipe",
    [
        "deepseek_v3_debugmodel_nvfp4",
        "deepseek_v3_16b_nvfp4",
        "deepseek_v3_671b_12_layers_nvfp4_mixed",
        "deepseek_v3_671b_nvfp4_mixed",
    ],
)
def test_deepseek_v3_nvfp4_recipes_resolve(monkeypatch, recipe):
    """Every DSV3 NVFP4 flavor builds its config tree.

    The failure mode is NVFP4Linear.Config's ``% 128`` rejection: the FFN fqns
    are per-flavor because 16B's dense FeedForward (2048 -> 10944) cannot be
    NVFP4. Without this, a mismatch only surfaces at launch time.
    """
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    config = ConfigManager().parse_args(["--module", "deepseek_v3", "--config", recipe])
    assert config.model_spec.name == "deepseek_v3"
    assert has_quantization(config.model_spec.model)
