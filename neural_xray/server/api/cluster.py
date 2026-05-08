"""
Model Surgery MRI — /api/cluster endpoints.

Build concept similarity clusters and query nearest neighbors.
"""

import logging
import sys
import traceback

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from neural_xray import ConceptCluster  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


class BuildClusterRequest(BaseModel):
    model_id: str
    threshold: float = 0.85


@router.post("/build")
async def build_clusters(req: BuildClusterRequest):
    """Build concept similarity clusters from extracted vectors."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    if not sess.concept_vectors:
        raise HTTPException(
            status_code=400,
            detail="No concept vectors — call /api/concepts/extract first",
        )

    try:
        cluster = ConceptCluster(sess.concept_vectors)
        cluster.build()
        with state.lock:
            sess.cluster = cluster

        result = cluster.to_dict()
        clusters = cluster.clusters(threshold=req.threshold)
        result["clusters"] = clusters
        result["num_clusters"] = len(clusters)

        logger.info(f"Built clusters for {req.model_id} "
                     f"— {len(clusters)} groups at threshold {req.threshold}")
        return result

    except Exception as e:
        logger.error(f"Cluster build failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nearest")
async def nearest_concepts(
    model_id: str = Query(...),
    concept: str = Query(...),
    top_k: int = Query(10, ge=1, le=100),
):
    """Find nearest neighbor concepts by cosine similarity."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    if sess.cluster is None:
        raise HTTPException(
            status_code=400,
            detail="Clusters not built — call /api/cluster/build first",
        )

    try:
        neighbors = sess.cluster.nearest(concept, top_k=top_k)
        return {
            "concept": concept,
            "neighbors": [{"concept": n, "similarity": s} for n, s in neighbors],
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Concept '{concept}' not in cluster")


@router.get("/similarity")
async def concept_similarity(
    model_id: str = Query(...),
    a: str = Query(...),
    b: str = Query(...),
):
    """Get cosine similarity between two concepts."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    if sess.cluster is None:
        raise HTTPException(status_code=400, detail="Clusters not built")

    try:
        sim = sess.cluster.similarity(a, b)
        return {"a": a, "b": b, "similarity": sim}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
