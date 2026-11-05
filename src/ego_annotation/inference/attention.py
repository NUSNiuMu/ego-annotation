"""Versioned new-worker SDPA representation fix for Turing FP16 GQA.

No weights, image tensors, precision or attention equation are changed. Explicit
KV repetition is PyTorch GQA's documented equation; fused rounding is not bit exact.
"""
from contextlib import contextmanager
ContractError = RuntimeError

RUNTIME_ID='turing-fp16-sdpa-explicit-kv-v1'


def _target(query,key,value,enabled,torch):
    if not enabled or query.dtype!=torch.float16 or key.dtype!=query.dtype or value.dtype!=query.dtype:
        return False
    if any(x.device.type!='cuda' or x.device!=query.device or x.ndim!=4 for x in (query,key,value)):
        return False
    if torch.cuda.get_device_capability(query.device)!=(7,5):
        return False
    qh,kh,vh=query.shape[1],key.shape[1],value.shape[1]
    if kh<=0 or kh!=vh or qh<=kh or qh%kh:
        return False
    return True


@contextmanager
def turing_sdpa(runtime_id,*,_torch=None):
    if runtime_id!=RUNTIME_ID:
        raise ContractError('New-worker attention runtime identity differs')
    if _torch is None:
        import torch
    else:torch=_torch
    # Imported only inside a new-workflow inference scope. The process still
    # owns the same exclusive GPU lock; no legacy inference runs concurrently.
    functional=torch.nn.functional
    original=functional.scaled_dot_product_attention
    stats={'runtime_id':RUNTIME_ID,'explicit_kv_calls':0,'original_gqa_heads':[],
           'target_backend':'EFFICIENT_ATTENTION','precision':'torch.float16','bit_exact_claim':False}
    def dispatch(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
        if not _target(query,key,value,enable_gqa,torch):
            return original(query,key,value,attn_mask=attn_mask,dropout_p=dropout_p,
                            is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
        repeats=query.shape[1]//key.shape[1]
        key=key.repeat_interleave(repeats,dim=1)
        value=value.repeat_interleave(repeats,dim=1)
        stats['explicit_kv_calls']+=1
        heads=[query.shape[1],key.shape[1]//repeats,value.shape[1]//repeats]
        if heads not in stats['original_gqa_heads']:stats['original_gqa_heads'].append(heads)
        # Unsupported backend raises. Never silently compute the quadratic math
        # scores matrix at a large sequence length after this representation fix.
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
            return original(query,key,value,attn_mask=attn_mask,dropout_p=dropout_p,
                            is_causal=is_causal,scale=scale,enable_gqa=False)
    functional.scaled_dot_product_attention=dispatch
    try:yield stats
    finally:
        changed=functional.scaled_dot_product_attention is not dispatch
        functional.scaled_dot_product_attention=original
        if changed:raise ContractError('Attention dispatch changed during exclusive new-worker scope')
