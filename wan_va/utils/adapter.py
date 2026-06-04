
import os
import torch
import torch.distributed as dist
import math
from einops import rearrange
from yunchang import LongContextAttention
from yunchang.kernels import AttnType

try:
    import torch_npu
    import torchair
    from torch_npu.contrib import transfer_to_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = False
    torch_npu.npu.set_per_process_memory_fraction(0.99)
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

def has_npu() -> bool:
    return HAS_NPU

def adapt_to_accelerator_device():
    if has_npu():
        print("load adapter for accelerator device, now support to run on Ascend NPU")
    else:
        print("load adapter for accelerator device, now support to run on Nvidia GPU")

def init_distributed_adapter():
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None or local_rank == "":
        return
    else:
        local_rank = int(local_rank)
    if HAS_NPU:
        dist.init_process_group(backend="cpu:gloo,npu:hccl",init_method="env://")
    else:
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))  

def deterministic_on():
    torch.use_deterministic_algorithms(True)
    from msprobe.pytorch import seed_all
    seed_all(seed=42, mode=True, rm_dropout=True) 

def get_longcontext_attention():
    if HAS_NPU:
        return LongContextAttention(ring_impl_type="basic_npu", attn_type=AttnType.NPU)
    else:
        return LongContextAttention(ring_impl_type="basic", attn_type=AttnType.FA3)

def get_compiler_backend():
    if HAS_NPU:
        return torchair.get_npu_backend()
    else:
        return "inductor"

def contiguous_for_channels_last_3d_memory_format(tensor: torch.Tensor):
    if HAS_NPU:
        ## NPU contiguous operator only supportted contiguous memory format
        return tensor.contiguous()
    else:
        return tensor.contiguous(memory_format=torch.channels_last_3d)

def adapter_gelu(x):
    if HAS_NPU:
        return torch_npu.fast_gelu(x)
    else:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def adapter_norm(self, x):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


def adpater_rmsnorm(x, weight, eps):
    if HAS_NPU:
        return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
    else:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            eps)
        if weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(weight)
        return weight * x


def apply_rotary_pos_emb_adapter(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if HAS_NPU:
        return _npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
    else:
        return _apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)

def _npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed

def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

def get_adapter_optimizer(model, accelerator):
    if HAS_NPU:
        optimizer = torch.optim.NpuFusedAdamW(
            accelerator.unwrap_model(model).trainable_modules(), 
            lr=1e-5,
            weight_decay=0.001,
        )
    else:
        optimizer = torch.optim.AdamW(
            accelerator.unwrap_model(model).trainable_modules(), 
            lr=1e-5,
            weight_decay=0.001,
        )