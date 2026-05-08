"""
Model Surgery MRI — /api/abliteration endpoints.
Concept removal via direction subtraction (abliteration).
"""
import logging
import sys
import traceback
from typing import List, Optional

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import state

logger = logging.getLogger("model-surgery")
router = APIRouter()


class AbliterateRequest(BaseModel):
    model_id: str
    concept: str
    strength: float = 1.0  # How much to subtract (1.0 = full removal)
    layers: Optional[List[int]] = None  # None = all MLP layers in fact zone


class ProbeRequest(BaseModel):
    model_id: str
    concept: str
    sentences: List[str]


@router.post("/probe")
async def probe_concept(req: ProbeRequest):
    """Probe how strongly a concept is represented before/after abliteration."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    if req.concept not in sess.concept_vectors:
        raise HTTPException(status_code=404, detail=f"Concept '{req.concept}' not extracted")

    model = sess.loader.model
    tokenizer = sess.loader.tokenizer
    device = next(model.parameters()).device
    concept_vec = sess.concept_vectors[req.concept].to(device).float()

    scores = []
    for sentence in req.sentences:
        enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        # Average hidden state across all layers and positions
        all_hidden = torch.stack([h[0].float().mean(dim=0) for h in outputs.hidden_states[1:]])
        avg_hidden = all_hidden.mean(dim=0)
        sim = float(F.cosine_similarity(avg_hidden.unsqueeze(0), concept_vec.unsqueeze(0)).item())
        scores.append({"sentence": sentence, "similarity": round(sim, 4)})

    return {
        "concept": req.concept,
        "model_id": req.model_id,
        "avg_similarity": round(sum(s["similarity"] for s in scores) / len(scores), 4),
        "scores": scores,
    }


@router.post("/remove")
async def abliterate(req: AbliterateRequest):
    """Remove a concept from model weights by subtracting its direction.
    
    This modifies the model IN MEMORY. Use probe before/after to verify.
    The concept direction is subtracted from MLP weight matrices in the fact zone.
    """
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    # Need either concept vector or cartography map
    cmap = sess.cartography_maps.get(req.concept)
    if cmap is None and req.concept not in sess.concept_vectors:
        raise HTTPException(
            status_code=404,
            detail=f"Concept '{req.concept}' not mapped or extracted. "
                   "Run cartography or extract concepts first."
        )

    model = sess.loader.model
    arch_map = sess.arch_map
    
    layers_edited = 0
    total_delta = 0.0

    if cmap and hasattr(cmap, 'final_directions') and cmap.final_directions:
        # Use cartography directions (more precise, per-layer)
        for layer_name, direction in cmap.final_directions.items():
            # Find the module
            target = None
            for name, module in model.named_modules():
                if name == layer_name:
                    target = module
                    break
            if target is None or not hasattr(target, 'weight'):
                continue
            
            direction = direction.to(target.weight.device).to(target.weight.dtype)
            # Subtract the outer product of the direction from the weight
            # W = W - strength * (W @ d) ⊗ d  (project out the concept direction)
            with torch.no_grad():
                proj = target.weight.data.float() @ direction.float()
                delta = req.strength * torch.outer(proj, direction.float())
                target.weight.data -= delta.to(target.weight.dtype)
                total_delta += float(delta.norm().item())
            layers_edited += 1
    else:
        # Fallback: use concept vector on MLP layers in fact zone
        concept_vec = sess.concept_vectors[req.concept].float()
        concept_dir = F.normalize(concept_vec, dim=0)
        
        # GPT-2 uses transformers Conv1D (weight is transposed vs nn.Linear)
        try:
            from transformers.pytorch_utils import Conv1D
            linear_types = (torch.nn.Linear, Conv1D)
        except ImportError:
            linear_types = (torch.nn.Linear,)
        
        for name, module in model.named_modules():
            if not isinstance(module, linear_types):
                continue
            if not hasattr(module, 'weight'):
                continue
            # Check if in fact zone
            layer_num = None
            for part in name.split('.'):
                if part.isdigit():
                    layer_num = int(part)
                    break
            if layer_num is None:
                continue
            if req.layers is not None and layer_num not in req.layers:
                continue
            if layer_num < arch_map.fact_layer_start or layer_num > arch_map.fact_layer_end:
                continue
            if not any(frag in name for frag in ('mlp', 'ffn', 'feed_forward', 'c_fc', 'fc')):
                continue
            
            device = module.weight.device
            d = concept_dir.to(device)
            w = module.weight.data
            # Conv1D weight is (in, out); nn.Linear is (out, in)
            is_conv1d = type(module).__name__ == 'Conv1D'
            in_dim = w.shape[0] if is_conv1d else w.shape[1]
            if in_dim != d.shape[0]:
                continue
            with torch.no_grad():
                if is_conv1d:
                    # Conv1D: weight is (in, out), so d @ weight gives (out,)
                    proj = w.float().T @ d  # (out,)
                    delta = req.strength * torch.outer(d, proj)  # (in, out)
                else:
                    proj = w.float() @ d  # (out,)
                    delta = req.strength * torch.outer(proj, d)  # (out, in)
                w.data -= delta.to(w.dtype)
                total_delta += float(delta.norm().item())
            layers_edited += 1

    logger.info(
        f"Abliterated '{req.concept}' from {req.model_id}: "
        f"{layers_edited} layers, strength={req.strength}, total_delta={total_delta:.2f}"
    )

    return {
        "concept": req.concept,
        "model_id": req.model_id,
        "layers_edited": layers_edited,
        "strength": req.strength,
        "total_delta_norm": round(total_delta, 4),
        "warning": "Model weights modified in memory. Reload model to restore original.",
    }
