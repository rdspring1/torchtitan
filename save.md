# NVFP4 Local SPMD Cleanup

## Goal

Remove NVFP4 DTensor support and unrelated branch changes so NVFP4 uses the
`spmd_types` local-tensor path exclusively.

## Current State

- NVFP4 forwards local weight, bias, and seed tensors directly to TorchAO.
- `Config.build` inlines colwise/rowwise local-map sharding and declares
  `_sr_seed` varying across DP, CP, and TP.
- All three supported Llama NVFP4 recipes default to `spmd_types` while allowing CLI
  overrides.
- The unsupported full-NVFP4 8B recipe is removed; the full debug-model recipe
  remains available for debugging.
- Both Blackwell-only NVFP4 integration variants and their skip gate are removed.
- `torchtitan/overrides/README.md` matches `origin/main`.

## What Changed

- Removed NVFP4 DTensor imports, unwrapping, type detection, helper APIs, and
  forward-time seed assertion.
- Inlined colwise/rowwise sharding construction into `NVFP4Linear.Config.build`.
- Added `_sr_seed` to module state sharding as varying across DP, CP, and TP.
- Set the three supported Llama NVFP4 recipes to default to `spmd_types`.
- Removed `llama3_8b_nvfp4`; production-sized usage remains represented by
  `llama3_8b_nvfp4_mixed`.
- Removed both NVFP4 integration variants and `skip_if_no_blackwell`.
- Restored `torchtitan/overrides/README.md` from `origin/main`.
- Replaced helper-oriented unit coverage with direct colwise/rowwise build,
  seed-layout, recipe-default, and CLI-override assertions.

## Validation

- PASS: Direct ufmt check over all six changed Python files; all already formatted.
- PASS: Direct flake8 with `.flake8` over all six changed Python files.
- PASS: Direct ufmt and flake8 checks for the recipe removal.
- PASS: `pytest -q tests/unit_tests/test_quantization.py` -- 17 passed.
- PASS: `git diff --check`.
- PASS: `git diff origin/main -- torchtitan/overrides/README.md` is empty.
- PASS: Searches confirm the removed NVFP4 DTensor/helper symbols, integration
  variants, and Blackwell gate are absent.
- SKIPPED: Pre-commit and Pyrefly, by explicit user direction.
- NOT RUN: Paired 10-step Blackwell numerical comparison. SM100 hardware is
  available, but the resumed validation scope was explicitly limited.

## Preflight

- Contract: Make only the planned NVFP4, recipe, integration-test, unit-test,
  upstream README restoration, and required state-file changes.
- Next action: No implementation work remains.
- Expected outcome: The supported recipe set contains full and mixed debug-model
  recipes plus the mixed 8B recipe.
- Risk: Full-NVFP4 8B training no longer has a registered convenience recipe.
- Confidence: HIGH.

## Surgical Simplicity

The unit-test edits replace tests for removed helpers with direct colwise/rowwise
build assertions and recipe-default coverage. Removing the unsupported 8B recipe
also removes its existing parameterized test case; no replacement abstraction or
test is needed. `debug-session.md` is required by the failed-validation recovery
loop; this file is required because the change touches more than three files and
validation required an environment checkpoint. No new production file,
abstraction, parameter, or standalone test file was introduced.
