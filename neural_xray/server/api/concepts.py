"""
Model Surgery MRI — /api/concepts endpoints.

Extract concept vectors, search, and list concepts for a loaded model.
"""

import logging
import sys
import traceback
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from neural_xray import ConceptExtractor  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


class ExtractRequest(BaseModel):
    model_id: str
    concepts: List[str]
    type: str = "entity"  # "entity" or "relation"
    batch_size: int = 8
    contrastive: bool = True  # Use CAA contrastive extraction


class ConceptSummary(BaseModel):
    concept: str
    norm: float
    dim: int


@router.get("/debug")
async def debug_concepts():
    """Debug: check if contrastive methods are available."""
    import inspect
    has_contrastive = hasattr(ConceptExtractor, 'extract_entities_contrastive')
    source_file = inspect.getfile(ConceptExtractor)
    methods = [m for m in dir(ConceptExtractor) if 'contrastive' in m or 'baseline' in m]
    return {
        "source_file": source_file,
        "has_contrastive": has_contrastive,
        "contrastive_methods": methods,
        "ExtractRequest_fields": list(ExtractRequest.model_fields.keys()),
    }


@router.post("/extract")
async def extract_concepts(req: ExtractRequest):
    """Extract concept vectors from a loaded model."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    contrastive = getattr(req, 'contrastive', True)

    try:
        extractor = ConceptExtractor(sess.loader, sess.arch_map)

        if req.type == "relation":
            if contrastive:
                vectors = extractor.extract_relations_contrastive(req.concepts)
            else:
                vectors = extractor.extract_relations(req.concepts)
            with state.lock:
                sess.relation_vectors.update(vectors)
        else:
            if contrastive:
                vectors = extractor.extract_entities_contrastive(req.concepts, batch_size=req.batch_size)
            else:
                vectors = extractor.extract_entities(req.concepts, batch_size=req.batch_size)
            with state.lock:
                sess.concept_vectors.update(vectors)

        summaries = [
            ConceptSummary(
                concept=name,
                norm=float(vec.norm()),
                dim=vec.shape[-1],
            )
            for name, vec in vectors.items()
        ]
        logger.info(f"Extracted {len(vectors)} {req.type} vectors from {req.model_id}")
        return {"extracted": [s.model_dump() for s in summaries]}

    except Exception as e:
        logger.error(f"Extraction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_concepts(model_id: str = Query(...)):
    """List all extracted concept names for a model."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    return {
        "entities": sorted(sess.concept_vectors.keys()),
        "relations": sorted(sess.relation_vectors.keys()),
    }


@router.get("/search")
async def search_concepts(
    model_id: str = Query(...),
    query: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
):
    """Fuzzy search over extracted concept names."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    all_names = list(sess.concept_vectors.keys()) + list(sess.relation_vectors.keys())
    q = query.lower()

    # Simple fuzzy: prefix match first, then substring, then contains
    exact = [n for n in all_names if n.lower() == q]
    prefix = [n for n in all_names if n.lower().startswith(q) and n not in exact]
    substr = [n for n in all_names if q in n.lower() and n not in exact and n not in prefix]

    results = (exact + prefix + substr)[:limit]
    return {"matches": results, "total": len(all_names)}


@router.get("/detail")
async def concept_detail(
    model_id: str = Query(...),
    concept: str = Query(...),
):
    """Get detailed info about a single extracted concept vector."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    vec = sess.concept_vectors.get(concept)
    if vec is None:
        vec = sess.relation_vectors.get(concept)
    if vec is None:
        raise HTTPException(status_code=404, detail=f"Concept '{concept}' not extracted yet")

    import torch
    return {
        "concept": concept,
        "dim": vec.shape[-1],
        "norm": float(vec.norm()),
        "mean": float(vec.mean()),
        "std": float(vec.std()),
        "min": float(vec.min()),
        "max": float(vec.max()),
        "type": "entity" if concept in sess.concept_vectors else "relation",
    }


# ---------------------------------------------------------------------------
# Concept packs — curated domain-specific concept sets
# ---------------------------------------------------------------------------
CONCEPT_PACKS = {
    "Common Objects": [
        "dog", "cat", "car", "tree", "house", "book", "chair", "water",
        "fire", "sun", "moon", "phone", "door", "window", "food",
    ],
    "Emotions": [
        "happy", "sad", "angry", "fear", "love", "hate", "joy",
        "anxiety", "trust", "surprise", "disgust", "hope", "grief",
    ],
    "Science": [
        "gravity", "electron", "photon", "energy", "mass", "force",
        "temperature", "pressure", "velocity", "atom", "molecule", "wave",
    ],
    "People & Society": [
        "doctor", "teacher", "king", "mother", "child", "friend",
        "soldier", "artist", "scientist", "criminal", "hero", "victim",
    ],
    "Actions": [
        "run", "eat", "think", "build", "destroy", "create", "learn",
        "fight", "sleep", "speak", "write", "dance", "fly", "kill",
    ],
    "Abstract": [
        "truth", "justice", "freedom", "power", "time", "death", "life",
        "knowledge", "beauty", "evil", "chaos", "order", "democracy",
    ],
    "Code & Tech": [
        "function", "variable", "array", "loop", "class", "server",
        "database", "algorithm", "memory", "network", "bug", "compile",
    ],
    "Relations": [
        "causes", "prevents", "requires", "contains", "produces",
        "similar_to", "opposite_of", "part_of", "leads_to", "depends_on",
    ],
}


@router.get("/packs")
async def list_concept_packs():
    """List available concept packs with their contents."""
    return {
        "packs": {name: concepts for name, concepts in CONCEPT_PACKS.items()},
    }


@router.post("/extract_pack")
async def extract_concept_pack(req: dict):
    """Extract all concepts from a named pack."""
    model_id = req.get("model_id")
    pack_name = req.get("pack")

    if not model_id or not pack_name:
        raise HTTPException(status_code=400, detail="model_id and pack required")

    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    if pack_name not in CONCEPT_PACKS:
        raise HTTPException(status_code=404, detail=f"Pack '{pack_name}' not found")

    concept_list = CONCEPT_PACKS[pack_name]
    is_relation = pack_name == "Relations"
    contrastive = req.get("contrastive", True)

    try:
        extractor = ConceptExtractor(sess.loader, sess.arch_map)

        if is_relation:
            if contrastive:
                vectors = extractor.extract_relations_contrastive(concept_list)
            else:
                vectors = extractor.extract_relations(concept_list)
            with state.lock:
                sess.relation_vectors.update(vectors)
        else:
            if contrastive:
                vectors = extractor.extract_entities_contrastive(concept_list, batch_size=8)
            else:
                vectors = extractor.extract_entities(concept_list, batch_size=8)
            with state.lock:
                sess.concept_vectors.update(vectors)

        logger.info(f"Extracted pack '{pack_name}' ({len(vectors)} concepts) from {model_id}")
        return {
            "pack": pack_name,
            "extracted": len(vectors),
            "concepts": sorted(vectors.keys()),
        }
    except Exception as e:
        logger.error(f"Pack extraction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
