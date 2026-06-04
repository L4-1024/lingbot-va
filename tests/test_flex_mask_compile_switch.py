import importlib.util
from pathlib import Path
import sys

import torch


def _load_model_module(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, *args, **kwargs: fn)

    module_name = "wan_va_model_under_test"
    sys.modules.pop(module_name, None)
    model_path = Path(__file__).resolve().parents[1] / "wan_va" / "modules" / "model.py"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_init_mask(model_mod, monkeypatch, raw_create, compiled_create):
    monkeypatch.setattr(model_mod, "create_block_mask", raw_create)
    monkeypatch.setattr(
        model_mod.FlexAttnFunc,
        "compiled_create_block_mask",
        compiled_create,
    )

    model_mod.FlexAttnFunc.init_mask(
        latent_shape=(1, 1, 1, 2, 2),
        action_shape=(1, 1, 1, 1, 1),
        padded_length=0,
        chunk_size=1,
        window_size=1,
        patch_size=(1, 1, 1),
        device="cpu",
    )


def test_disable_flex_compile_uses_uncompiled_block_mask(monkeypatch):
    model_mod = _load_model_module(monkeypatch)
    raw_calls = []
    compiled_calls = []

    def raw_create(*args, **kwargs):
        raw_calls.append(kwargs)
        return object()

    def compiled_create(*args, **kwargs):
        compiled_calls.append(kwargs)
        return object()

    monkeypatch.setenv("WAN_VA_DISABLE_FLEX_COMPILE", "1")

    _run_init_mask(model_mod, monkeypatch, raw_create, compiled_create)

    assert len(raw_calls) == 2
    assert compiled_calls == []
    assert [call["_compile"] for call in raw_calls] == [False, False]


def test_flex_compile_enabled_uses_compiled_block_mask(monkeypatch):
    model_mod = _load_model_module(monkeypatch)
    raw_calls = []
    compiled_calls = []

    def raw_create(*args, **kwargs):
        raw_calls.append(kwargs)
        return object()

    def compiled_create(*args, **kwargs):
        compiled_calls.append(kwargs)
        return object()

    monkeypatch.delenv("WAN_VA_DISABLE_FLEX_COMPILE", raising=False)

    _run_init_mask(model_mod, monkeypatch, raw_create, compiled_create)

    assert raw_calls == []
    assert len(compiled_calls) == 2
    assert [call["_compile"] for call in compiled_calls] == [True, True]
