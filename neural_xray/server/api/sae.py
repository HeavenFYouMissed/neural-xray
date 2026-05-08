"""
Model Surgery MRI — /api/sae endpoints.

Train and query Sparse Autoencoders for mechanistic interpretability.
"""

import logging
import sys
import traceback
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from neural_xray import SparseAutoencoder  # noqa: E402
from ..licensing import require_feature  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


class SAETrainRequest(BaseModel):
    model_id: str
    sentences: List[str]
    n_features: int = 8192
    epochs: int = 3
    batch_size: int = 256
    lr: float = 2e-4


class SAEDecomposeRequest(BaseModel):
    model_id: str
    concept: str
    threshold: float = 0.0


@router.post("/train")
async def train_sae(req: SAETrainRequest):
    """Train a Sparse Autoencoder on a model's hidden states."""
    try:
        require_feature("sae")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    try:
        sae = SparseAutoencoder(
            hidden_dim=sess.loader.hidden_size,
            n_features=req.n_features,
        )
        stats = sae.train_on_model(
            loader=sess.loader,
            arch_map=sess.arch_map,
            sentences=req.sentences,
            epochs=req.epochs,
            batch_size=req.batch_size,
            lr=req.lr,
        )
        with state.lock:
            sess.sae = sae

        logger.info(f"Trained SAE for {req.model_id} — layer {sae.target_layer}, "
                     f"{req.n_features} features")
        return {
            "model_id": req.model_id,
            "target_layer": sae.target_layer,
            "target_layer_role": sae.target_layer_role,
            "n_features": req.n_features,
            "training_stats": stats,
        }

    except Exception as e:
        logger.error(f"SAE training failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decompose")
async def decompose_concept(req: SAEDecomposeRequest):
    """Decompose a concept vector into SAE features."""
    try:
        require_feature("sae")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    if sess.sae is None:
        raise HTTPException(
            status_code=400,
            detail="No SAE trained — call /api/sae/train first",
        )

    vec = sess.concept_vectors.get(req.concept)
    if vec is None:
        raise HTTPException(
            status_code=404,
            detail=f"Concept '{req.concept}' not extracted",
        )

    try:
        indices, values = sess.sae.get_active_features(vec, threshold=req.threshold)
        # Align features to concepts if we have vectors
        feature_labels = {}
        if sess.concept_vectors:
            feature_labels = sess.sae.align_features_to_concepts(sess.concept_vectors)

        features = []
        for idx, val in zip(indices.tolist(), values.tolist()):
            label = feature_labels.get(idx, [])
            features.append({
                "feature_idx": idx,
                "activation": val,
                "aligned_concepts": [{"concept": c, "score": s} for c, s in label],
            })

        return {
            "concept": req.concept,
            "num_active_features": len(features),
            "features": features,
        }

    except Exception as e:
        logger.error(f"SAE decompose failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
