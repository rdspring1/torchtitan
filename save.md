# NVFP4 V2 recipe integration — DSV3

## Goal

Two DSV3 arms: **V2 on every NVFP4-eligible linear and both MoE FFN layers**, and
design doc §17's **split** (FC1 on V1_REQUANT, FC2 on V2). V2's torchao kernels
only became functional today (torchao `6aa99b83`), which is why torchtitan had
staged the MoE half but deliberately refused `"v2"` for linears.

## Current state

Implemented and validated on 4x GB200. **Not committed.**

| File | Change |
| --- | --- |
| `torchtitan/components/quantization/nvfp4.py` | V2 on `NVFP4Linear`; `build_nvfp4_sign_resampler` |
| `torchtitan/components/quantization/__init__.py` | export the resampler |
| `torchtitan/trainer.py` | build the resampler after `init_weights`; drive it per microbatch |
| `torchtitan/models/deepseek_v3/config_registry.py` | leaf-aware fqns, split helper, 6 new flavors |
| `tests/unit_tests/test_quantization.py` | +13 tests |
| `torchtitan/components/quantization/nvfp4.md` | V2 section |

### What Changed

**Gap 1 — linears accept `"v2"`.** `_NVFP4_LINEAR_RECIPES` collapsed into
`_NVFP4_RECIPES` (they became identical). `_init_self_buffers` draws a 128-element
`_rht_sign_vector` plus `_dgrad_rht_sign_vector` under V2; `forward` branches to
`nvfp4_linear_v2` **before** the V1_REQUANT branch;
`_refresh_rht_sign_vector_tuple` returns early for V2; `_NVFP4LinearV2` registered
as a local autograd function for SPMD; cutedsl + non-v1 raises.

**Gap 2 — the resample cadence, which nothing drove (live defect).**
`resample_nvfp4_rht_signs` had **zero call sites**, so the existing MoE V2 buffers
kept their initial draw for entire runs — correct gradients, not measuring V2.
`build_nvfp4_sign_resampler(model_parts, seed)` returns a
`(step, microbatch) -> None` closure or `None`. `Trainer.__init__` builds it
**after `init_weights`** (buffers are `None` before that and
`iter_dynamic_sign_buffers` skips `None`, so probing early would silently disable
the cadence). `train_step` calls it per accumulation group.

**Gap 3 — flavors.** `deepseek_v3_{debugmodel,16b,671b}_nvfp4_{v2,split}`.
The three base functions gained `fc2_recipe=None` (defaults to `recipe`, so
unsplit configs are unchanged). The dense-FFN split uses two
`NVFP4LinearConverter` instances over **disjoint** fqns.

The debugmodel arms were not in the original plan. They exist because the 16B and
671B arms need 8-way EP and the DeepEP backend, so neither is runnable as a smoke
test on one node — see Blocked below.

## Validation

### Unit tests

`PYTHONPATH=~/nvfp4/third_party/torchao python -m pytest tests/unit_tests/test_quantization.py -q`

| | result |
| --- | --- |
| baseline (`git stash`) | **41 passed**, 0 failed |
| with this change | **54 passed**, 0 failed |

### 4x GB200, `deepseek_v3_debugmodel_*`, 5 steps

| arm | step 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `_nvfp4` (v1) | 8.19229 | 6.28795 | 4.96934 | 4.77453 | 4.51121 |
| `_nvfp4_v2` | 8.02221 | 6.07373 | 4.85160 | 4.72505 | 4.46864 |
| `_nvfp4_split` | 8.00478 | 6.03033 | 4.92661 | 4.80783 | 4.58106 |

All finite, all descending, and **all three curves distinct** — an identical curve
would have meant the recipe never reached the kernels.

Under `torch.compile --compile.components model`, both new arms exit 0 with finite
descending loss.

### Resampler, live

`NVFP4 V2: resampling N RHT sign buffers per microbatch`, on every rank:

- `deepseek_v3_16b_nvfp4_v2` → **208** = 2 buffers x 4 V2 modules x 26 MoE layers.
- `deepseek_v3_debugmodel_nvfp4_v2` → **38**; `_split` → **18**. The 20-buffer
  difference is exactly the 10 `w1`/`w3` linears that drop to v1_requant under the
  split, so the count independently confirms the split reached the right modules.

**The DTensor risk is retired.** In the 16B run the resampler both built and fired
against the fully parallelized model (FSDP + EP), before the forward, with no
error — so `iter_dynamic_sign_buffers` finds the buffers and `buffer.copy_()`
works on them after `Module.parallelize`. That was the top unknown.

## Blocked

`deepseek_v3_16b_nvfp4_v2` reached the training loop and resampled 208 buffers,
then failed in the MoE forward on `ModuleNotFoundError: No module named 'deep_ep'`.
The 16B and 671B configs set `moe_comm_backend="hybridep"`, which needs DeepEP —
a compiled CUDA library absent from this container. Unrelated to this change; the
v1 16B arm would fail identically here.

The 16B config also wants `expert_parallel_degree=8` (2-node) and
`./assets/hf/deepseek-moe-16b-base`; the run above used
`--parallelism.expert-parallel-degree 4 --hf-assets-path ./tests/assets/tokenizer
--dataloader.dataset c4_test`.

## Next action

Run `deepseek_v3_16b_nvfp4_v2` and `_split` on a DeepEP-equipped 8-GPU host for
~20 steps and confirm the loss curves differ from `deepseek_v3_16b_nvfp4_v1_requant`.
Everything below 16B scale is already green.

## Environment mutation (checkpoint-class)

Installed — all pure-Python wheels; `torch` stayed at `2.15.0a0+git0f3e7e2`:

`docstring_parser`, `spmd_types==0.2.1`, `typeguard`, `tyro`, `torchdata`,
`datasets`, `tokenizers`

Two consequences to know about:
- **`fsspec` 2026.7.0 → 2026.6.0**, forced by `datasets`. torch requires only
  `fsspec>=0.8.5`, so this is well inside its constraint.
- `click` went to 8.5.0, which the dev tool `spin 0.17` pins below 8.4. `spin` is
  unrelated to training; noted rather than fixed.

Without these, no torchtitan test runs at all — 23 of 41 failed on import before.

## Confidence

HIGH for everything at debugmodel scale and for the config routing, buffer
lifecycle, resample cadence, DTensor interaction and `torch.compile` — all
directly executed. MEDIUM for 16B/671B, which have never completed a step here
for reasons predating this change.

## Surgical Simplicity

- `_expects_v2`: turns a silent mis-wire into a warning; the failure mode is a
  full run measuring the wrong recipe.
- `_nvfp4_linear_converters`: shared by all three model sizes so the split cannot
  drift between them.
- `leaves` param on `_nvfp4_ffn_linear_fqns`: defaults to `()`, preserving the
  existing call shape exactly.
- `_V2_LINEAR_{WGRAD,DGRAD}_SEED`: named so the "distinct from the MoE seeds"
  reason sits with the values.
- debugmodel arms: the only single-node-runnable V2 configs, and the fast smoke
  this recipe otherwise lacks.
- 13 tests: one per new behaviour. The resampler-replay assertion doubles as the
  seed-determinism guard; the flavor test asserts per-leaf recipes on the
  transformed config tree, so it catches a recipe that stops short of the modules.
