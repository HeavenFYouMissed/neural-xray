"""
Model Surgery MRI — /api/patching endpoints.
Activation patching / causal tracing for mechanistic interpretability.
"""
import asyncio
import concurrent.futures
import logging
import sys
import traceback
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import state

logger = logging.getLogger("model-surgery")
router = APIRouter()


class PatchRequest(BaseModel):
    model_id: str
    sentence: str
    corrupted_sentence: str  # Same sentence with key word replaced
    patch_layer: Optional[int] = None  # None = sweep all layers
    token_position: Optional[int] = None  # None = all positions


def _get_hidden_states(model, tokenizer, sentence: str, device) -> List[torch.Tensor]:
    """Run forward pass and collect all hidden states."""
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    return outputs.hidden_states, outputs.logits, input_ids


@router.post("/trace")
async def causal_trace(req: PatchRequest):
    """Run activation patching / causal tracing.
    
    Process:
    1. Run clean sentence → get clean logits (baseline)
    2. Run corrupted sentence → get corrupted logits (damaged)
    3. For each layer: replace corrupted hidden state with clean state at that layer,
       continue forward → measure how much the output recovers toward clean
    
    Returns per-layer "recovery score" showing which layers causally matter.
    """
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    model = sess.loader.model
    tokenizer = sess.loader.tokenizer
    device = next(model.parameters()).device

    # Get clean and corrupted hidden states
    try:
        clean_hidden, clean_logits, clean_ids = _get_hidden_states(
            model, tokenizer, req.sentence, device
        )
        corrupt_hidden, corrupt_logits, corrupt_ids = _get_hidden_states(
            model, tokenizer, req.corrupted_sentence, device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forward pass failed: {e}")

    if clean_ids.shape != corrupt_ids.shape:
        raise HTTPException(
            status_code=400,
            detail="Clean and corrupted sentences must tokenize to same length. "
                   "Try keeping the same sentence structure."
        )

    # Baseline: clean output probability for last token
    clean_last_logits = clean_logits[0, -1].float()
    clean_probs = F.softmax(clean_last_logits, dim=-1)
    clean_pred_token = int(clean_last_logits.argmax())
    clean_pred_prob = float(clean_probs[clean_pred_token])
    
    corrupt_last_logits = corrupt_logits[0, -1].float()
    corrupt_probs = F.softmax(corrupt_last_logits, dim=-1)
    corrupt_pred_prob = float(corrupt_probs[clean_pred_token])
    
    # Effect = how much the corruption changed the prediction
    total_effect = clean_pred_prob - corrupt_pred_prob

    clean_tokens = [tokenizer.decode(t) for t in clean_ids[0]]
    num_layers = len(clean_hidden) - 1  # -1 for embedding layer

    # Patch each layer: replace corrupted activation with clean at that layer
    results = []
    layers_to_test = [req.patch_layer] if req.patch_layer is not None else list(range(num_layers))
    
    for layer_idx in layers_to_test:
        # We need to hook into the model and replace activations at this layer
        hook_handle = None
        patched_output = None
        
        def make_hook(clean_h, pos):
            def hook_fn(module, input, output):
                # output is usually a tuple; hidden states are first element
                if isinstance(output, tuple):
                    h = output[0].clone()
                    if pos is not None:
                        h[0, pos] = clean_h[0, pos]
                    else:
                        h[0] = clean_h[0]
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    if pos is not None:
                        h[0, pos] = clean_h[0, pos]
                    else:
                        h[0] = clean_h[0]
                    return h
            return hook_fn

        # Find the target layer module
        target_module = None
        for name, module in model.named_modules():
            # Match transformer block by index
            if any(f".{layer_idx}." in name or f"[{layer_idx}]" in name 
                   for _ in [None]):
                if hasattr(module, 'self_attn') or hasattr(module, 'attn'):
                    target_module = module
                    break

        if target_module is None:
            # Fallback: use hidden states index
            # Just compute KL divergence between clean and corrupt at this layer
            if layer_idx + 1 < len(clean_hidden):
                clean_h = clean_hidden[layer_idx + 1][0].float()
                corrupt_h = corrupt_hidden[layer_idx + 1][0].float()
                cos_sim = float(F.cosine_similarity(
                    clean_h.mean(dim=0, keepdim=True),
                    corrupt_h.mean(dim=0, keepdim=True)
                ).item())
                # Indirect effect estimation from hidden state divergence
                divergence = float(torch.norm(clean_h - corrupt_h).item())
                results.append({
                    "layer": layer_idx,
                    "indirect_effect": round(1.0 - cos_sim, 4),
                    "divergence": round(divergence, 4),
                    "recovery": None,
                    "method": "divergence",
                })
            continue

        # Actual activation patching
        try:
            clean_h_at_layer = clean_hidden[layer_idx + 1]
            hook = make_hook(clean_h_at_layer, req.token_position)
            hook_handle = target_module.register_forward_hook(hook)
            
            with torch.no_grad():
                patched_outputs = model(input_ids=corrupt_ids)
            
            hook_handle.remove()
            
            patched_logits = patched_outputs.logits[0, -1].float()
            patched_probs = F.softmax(patched_logits, dim=-1)
            patched_pred_prob = float(patched_probs[clean_pred_token])
            
            recovery = (patched_pred_prob - corrupt_pred_prob) / (total_effect + 1e-10)
            
            results.append({
                "layer": layer_idx,
                "indirect_effect": round(float(patched_pred_prob - corrupt_pred_prob), 4),
                "recovery": round(float(min(max(recovery, 0), 2)), 4),
                "patched_prob": round(patched_pred_prob, 4),
                "method": "activation_patch",
            })
        except Exception as e:
            if hook_handle:
                hook_handle.remove()
            results.append({
                "layer": layer_idx,
                "error": str(e),
                "method": "failed",
            })

    return {
        "model_id": req.model_id,
        "sentence": req.sentence,
        "corrupted_sentence": req.corrupted_sentence,
        "tokens": clean_tokens,
        "clean_prediction": tokenizer.decode(clean_pred_token),
        "clean_prob": round(clean_pred_prob, 4),
        "corrupt_prob": round(corrupt_pred_prob, 4),
        "total_effect": round(total_effect, 4),
        "num_layers": num_layers,
        "layers": results,
    }
