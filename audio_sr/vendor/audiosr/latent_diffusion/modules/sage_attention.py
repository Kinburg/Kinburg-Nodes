"""SageAttention GPU-optimized attention implementation."""

import torch
import logging

logger = logging.getLogger("AudioSR")

SAGE_ATTENTION_AVAILABLE = False
SAGE_ATTENTION_FUNC = None

try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
    SAGE_ATTENTION_FUNC = sageattn
    logger.info("[SageAttention] Available - will use auto kernel selection")
except ImportError:
    logger.info("[SageAttention] Not installed (pip install sageattention)")


def get_gpu_info():
    """Get GPU compute capability info."""
    if not torch.cuda.is_available():
        return None, None
    
    major, minor = torch.cuda.get_device_capability()
    arch_code = major * 10 + minor
    return major, minor, arch_code


def check_sage_compatibility():
    """Check if SageAttention is compatible with current GPU."""
    if not SAGE_ATTENTION_AVAILABLE:
        return False, "SageAttention not installed"
    
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    major, minor, arch_code = get_gpu_info()
    
    if arch_code >= 80:
        return True, f"SM{arch_code} supported"
    else:
        return False, f"SM{arch_code} not supported (requires SM80+)"


def sageattn_forward(q, k, v, is_causal=False):
    """
    SageAttention forward with automatic kernel selection.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        is_causal: Whether to use causal attention (default: False)
    
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    if not SAGE_ATTENTION_AVAILABLE:
        raise RuntimeError("SageAttention not available")
    
    major, minor, arch_code = get_gpu_info()
    
    dtype = q.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        needs_cast_back = True
        orig_dtype = dtype
    else:
        needs_cast_back = False
    
    out = SAGE_ATTENTION_FUNC(q, k, v, is_causal=is_causal)
    
    if needs_cast_back:
        out = out.to(orig_dtype)
    
    return out
