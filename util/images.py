"""Image helpers shared by the pack's long-running samplers.

Both of these were Morpheus-private until Phantas needed them too, and both are the kind of thing
that must be identical wherever it is used: two slightly different fp16 roundings would split a
cache, and two different thumbnail encoders would make one node's live log heavier than another's.

torch is imported inside the functions, not at module level, so a headless test can import this
module without pulling the whole stack in.
"""
import logging


def log_uris(images, max_side=320):
    """Small JPEG data URIs for the live log — the same encoder the LLM vision path uses.

    Deliberately downscaled: these go over the websocket on every shot or frame, and the log only
    needs a thumbnail. Never raises: a picture that fails to encode must not take a render down."""
    try:
        import torch
        from ..local_llm.llm_node import _encode_images
        frames = [im for im in (images or []) if im is not None]
        if not frames:
            return []
        batch = frames[0] if len(frames) == 1 else torch.cat(frames, dim=0)
        return _encode_images(batch, max_side)
    except Exception as e:  # pragma: no cover
        logging.debug(f"[Kinburg] live-log image encode skipped: {e}")
        return []


def fp16_round(img):
    """Round a frame through fp16 so the fresh path and the cached path are bit-identical.

    Every frame that both leaves a node AND lands in a cache has to go through this. The cache
    stores fp16, so a fresh float32 frame and its own cached copy would otherwise differ — and any
    downstream cache keyed on the frame's contents (`diskcache.tensor_key`) would miss forever.

    `.contiguous()` is not cosmetic: a real video VAE hands frames back as a permuted view, `.to()`
    preserves those strides, and safetensors refuses to write a non-contiguous tensor — which is
    exactly how the shot cache silently wrote nothing on its first real run.
    """
    import torch
    return img.detach().to("cpu", torch.float16).clamp(0.0, 1.0).to(torch.float32).contiguous()
