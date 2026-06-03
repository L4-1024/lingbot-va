# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from torch import nn
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def _set_submodule_tensor(module, tensor_name, tensor):
    parts = tensor_name.split(".")
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    leaf_name = parts[-1]

    if leaf_name in target._parameters:
        old_param = target._parameters[leaf_name]
        target._parameters[leaf_name] = nn.Parameter(
            tensor,
            requires_grad=old_param.requires_grad,
        )
    elif leaf_name in target._buffers:
        target._buffers[leaf_name] = tensor
    else:
        raise RuntimeError(f"Cannot materialize unknown tensor: {tensor_name}")


def _empty_like_on_cpu(tensor):
    return torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        layout=tensor.layout,
        device="cpu",
    )


def _init_missing_transformer_weights(model):
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    materialized_modules = set()

    for name, param in list(named_params.items()):
        if not param.is_meta:
            continue

        source_name = None
        if name.startswith("condition_embedder_action."):
            source_name = name.replace(
                "condition_embedder_action.",
                "condition_embedder.",
                1,
            )

        if source_name and source_name in named_params and not named_params[source_name].is_meta:
            value = named_params[source_name].detach().to(device="cpu", dtype=param.dtype).clone()
        else:
            value = _empty_like_on_cpu(param)
            materialized_modules.add(name.rsplit(".", 1)[0])
        _set_submodule_tensor(model, name, value)

    named_buffers = dict(model.named_buffers())
    for name, buffer in list(named_buffers.items()):
        if not buffer.is_meta:
            continue

        source_name = None
        if name.startswith("condition_embedder_action."):
            source_name = name.replace(
                "condition_embedder_action.",
                "condition_embedder.",
                1,
            )

        if source_name and source_name in named_buffers and not named_buffers[source_name].is_meta:
            value = named_buffers[source_name].detach().to(device="cpu", dtype=buffer.dtype).clone()
        else:
            value = torch.zeros(
                buffer.shape,
                dtype=buffer.dtype,
                layout=buffer.layout,
                device="cpu",
            )
            materialized_modules.add(name.rsplit(".", 1)[0])
        _set_submodule_tensor(model, name, value)

    for module_name in sorted(materialized_modules):
        module = model.get_submodule(module_name)
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    remaining_meta = [
        name for name, param in model.named_parameters() if param.is_meta
    ] + [
        name for name, buffer in model.named_buffers() if buffer.is_meta
    ]
    if remaining_meta:
        raise RuntimeError(
            "Transformer checkpoint left meta tensors uninitialized: "
            + ", ".join(remaining_meta)
        )


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    loaded = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        output_loading_info=True,
        **kwargs
    )
    if isinstance(loaded, tuple):
        model, _loading_info = loaded
    else:
        model = loaded
    _init_missing_transformer_weights(model)
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
