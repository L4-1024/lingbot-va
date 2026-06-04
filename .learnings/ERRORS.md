# Errors

## [ERR-20260602-001] fake_checkpoint_meta_regression

**Logged**: 2026-06-02T07:32:00Z
**Priority**: medium
**Status**: pending
**Area**: tests

### Summary
A temporary safetensors fake-checkpoint regression script for Lingbot VA meta tensor loading exited with Bus error in the current sandbox.

### Details
The script created a tiny WanTransformer3DModel, saved it with safe_serialization, pruned action-branch weights, and attempted to reload through load_transformer. In this sandbox it dumped core before Python-level assertions, likely due environment/runtime instability rather than a verified code regression. The targeted helper test for _init_missing_transformer_weights passed separately.

### Suggested Action
When validating meta-tensor checkpoint loading in this Ascend sandbox, prefer focused helper tests or run checkpoint reload tests in a stable external environment with visible Ascend/runtime support.

### Metadata
- Source: error
- Related Files: wan_va/modules/utils.py
- Tags: safetensors, meta-tensor, ascend-env

---
