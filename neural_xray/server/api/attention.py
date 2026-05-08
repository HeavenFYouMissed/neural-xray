"""
Model Surgery MRI — /api/attention endpoints.
"""
import asyncio
import concurrent.futures
import logging
import sys
import traceback
from typing import List, Optional

import torch
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import state

logger = logging.getLogger("model-surgery")
router = APIRouter()


class AttentionRequest(BaseModel):
    model_id: str
    sentence: str
    layer: Optional[int] = None  # None = all layers


@router.post("/heads")
async def attention_heads(req: AttentionRequest):
    """Get attention head patterns for a sentence."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    model = sess.loader.model
    tokenizer = sess.loader.tokenizer
    device = next(model.parameters()).device

    enc = tokenizer(req.sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"].to(device)
    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    try:
        with torch.no_grad():
            # SDPA doesn't support output_attentions — temporarily switch to eager
            # Must bypass config __setattr__ validation using object.__setattr__
            old_impl = getattr(model.config, '_attn_implementation', None)
            need_eager = old_impl == 'sdpa'
            if need_eager:
                object.__setattr__(model.config, '_attn_implementation', 'eager')
                for module in model.modules():
                    if hasattr(module, '_attn_implementation'):
                        object.__setattr__(module, '_attn_implementation', 'eager')
            try:
                outputs = model(input_ids=input_ids, output_attentions=True)
            finally:
                if need_eager:
                    object.__setattr__(model.config, '_attn_implementation', 'sdpa')
                    for module in model.modules():
                        if hasattr(module, '_attn_implementation'):
                            object.__setattr__(module, '_attn_implementation', 'sdpa')
        
        attentions = outputs.attentions  # tuple of [1, num_heads, seq_len, seq_len]
        if not attentions:
            raise HTTPException(
                status_code=400,
                detail="Model did not return attention weights. This architecture may not support output_attentions."
            )
        
        layers_data = []
        for layer_idx, attn in enumerate(attentions):
            if req.layer is not None and layer_idx != req.layer:
                continue
            
            attn_np = attn[0].float().cpu().numpy()  # [num_heads, seq, seq]
            num_heads = attn_np.shape[0]
            
            heads = []
            for h in range(num_heads):
                head_attn = attn_np[h]  # [seq, seq]
                # Entropy per position, averaged
                entropy_per_pos = -np.sum(
                    head_attn * np.log(head_attn + 1e-10), axis=-1
                )
                avg_entropy = float(np.mean(entropy_per_pos))
                # Top attended positions (averaged across query positions)
                avg_attn = head_attn.mean(axis=0)  # [seq]
                
                heads.append({
                    "head": h,
                    "entropy": round(avg_entropy, 4),
                    "avg_pattern": [round(float(x), 4) for x in avg_attn],
                    # Full matrix only for single-layer requests (large data)
                    "matrix": [[round(float(x), 4) for x in row] for row in head_attn] if req.layer is not None else None,
                })
            
            layers_data.append({
                "layer": layer_idx,
                "num_heads": num_heads,
                "heads": heads,
            })
        
        return {
            "model_id": req.model_id,
            "sentence": req.sentence,
            "tokens": tokens,
            "num_layers": len(attentions),
            "layers": layers_data,
        }
    
    except Exception as e:
        logger.error(f"Attention analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
