"""
Model Surgery MRI — /api/trace endpoints.

Run concept flow traces, dead zone analysis, and contamination maps.
"""

import logging
import sys
import traceback
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from neural_xray import ConceptFlowTracer  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


class TraceRequest(BaseModel):
    model_id: str
    concept: str


class TraceAllRequest(BaseModel):
    model_id: str
    concepts: Optional[List[str]] = None  # None = trace all extracted


def _all_vectors(sess) -> dict:
    """Merge entity + relation vectors so both can be traced."""
    merged = dict(sess.concept_vectors)
    merged.update(sess.relation_vectors)
    return merged


def _get_tracer(model_id: str) -> ConceptFlowTracer:
    sess = state.get_session(model_id)
    vectors = _all_vectors(sess)
    if not vectors:
        raise HTTPException(
            status_code=400,
            detail="No concept vectors extracted yet — call /api/concepts/extract first",
        )
    return ConceptFlowTracer(
        loader=sess.loader,
        arch_map=sess.arch_map,
        concept_vectors=vectors,
    )


def _get_tracer_no_concepts(model_id: str) -> ConceptFlowTracer:
    """Get a tracer that doesn't require concept vectors (for logit lens etc.)."""
    sess = state.get_session(model_id)
    return ConceptFlowTracer(
        loader=sess.loader,
        arch_map=sess.arch_map,
        concept_vectors=sess.concept_vectors or {},
    )


@router.post("/run")
async def run_trace(req: TraceRequest):
    """Run a concept flow trace through all layers."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    if req.concept not in sess.concept_vectors and req.concept not in sess.relation_vectors:
        raise HTTPException(status_code=404, detail=f"Concept '{req.concept}' not extracted")

    try:
        tracer = _get_tracer(req.model_id)
        trace = tracer.trace(req.concept)
        # Cache it
        with state.lock:
            sess.traces[req.concept] = trace
        logger.info(f"Traced '{req.concept}' through {req.model_id}")
        return trace.to_dict()
    except Exception as e:
        logger.error(f"Trace failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run_all")
async def run_trace_all(req: TraceAllRequest):
    """Trace multiple concepts (or all extracted concepts)."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    concepts = req.concepts or list(_all_vectors(sess).keys())
    if not concepts:
        raise HTTPException(status_code=400, detail="No concepts to trace")

    try:
        tracer = _get_tracer(req.model_id)
        traces = tracer.trace_all(concepts)
        with state.lock:
            sess.traces.update(traces)

        # Build contamination map
        cmap = tracer.build_contamination_map(traces)
        with state.lock:
            sess.contamination_map = cmap

        logger.info(f"Traced {len(traces)} concepts through {req.model_id}")
        return {
            "traces": {name: t.to_dict() for name, t in traces.items()},
            "contamination_map": cmap.to_dict(),
        }
    except Exception as e:
        logger.error(f"Trace-all failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dead_zones")
async def get_dead_zones(
    model_id: str = Query(...),
    concept: str = Query(...),
):
    """Analyze dead zones for a previously traced concept."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    trace = sess.traces.get(concept)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trace for '{concept}' — run /api/trace/run first",
        )

    try:
        tracer = _get_tracer(model_id)
        analysis = tracer.analyze_dead_zones(trace)
        return {
            "concept": analysis.concept,
            "total_layers": analysis.total_layers,
            "dead_threshold": analysis.dead_threshold,
            "zones": [
                {
                    "start_layer": z.start_layer,
                    "end_layer": z.end_layer,
                    "entry_concept": z.entry_concept,
                    "exit_concept": z.exit_concept,
                    "trajectory_shift": z.trajectory_shift,
                    "is_active": z.is_active,
                    "causal_layer": z.causal_layer,
                }
                for z in analysis.zones
            ],
            "active_zones": len(analysis.active_zones),
            "minimum_cut_layers": [
                {"layer": l, "from": f, "to": t, "shift": s}
                for l, f, t, s in analysis.minimum_cut_layers
            ],
        }
    except Exception as e:
        logger.error(f"Dead zone analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cached")
async def get_cached_trace(
    model_id: str = Query(...),
    concept: str = Query(...),
):
    """Get a previously computed trace from cache."""
    try:
        sess = state.get_session(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")

    trace = sess.traces.get(concept)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No cached trace for '{concept}'")
    return trace.to_dict()


# ── Logit Lens ────────────────────────────────────────────────────────────────


class LogitLensRequest(BaseModel):
    model_id: str
    sentence: str
    top_k: int = 10
    token_position: int = -1  # -1 = last token


@router.post("/logit_lens")
async def run_logit_lens(req: LogitLensRequest):
    """
    Run the logit lens on a sentence — shows what each layer would predict
    if the model stopped processing there. The "disassembly view".
    """
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    try:
        tracer = _get_tracer_no_concepts(req.model_id)
        result = tracer.logit_lens(
            sentence=req.sentence,
            top_k=req.top_k,
            token_position=req.token_position,
        )
        logger.info(
            f"Logit lens: '{req.sentence}' through {req.model_id}, "
            f"{len(result.layers)} layers"
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Logit lens failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Sentence Trace (Per-Token Graph) ─────────────────────────────────────────


class TraceGraphRequest(BaseModel):
    model_id: str
    sentence: str
    top_k_tokens: int = 5
    attention_threshold: float = 0.05
    num_paths: int = 5


@router.post("/graph")
async def run_trace_graph(req: TraceGraphRequest):
    """
    Trace every token through every layer — the "step through" view.
    Returns a full graph with per-token nodes, attention edges, and paths.
    """
    try:
        sess = state.get_session(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not loaded")

    try:
        tracer = _get_tracer_no_concepts(req.model_id)
        result = tracer.trace_sentence(
            sentence=req.sentence,
            top_k_tokens=req.top_k_tokens,
            attention_threshold=req.attention_threshold,
            num_paths=req.num_paths,
        )
        logger.info(
            f"Trace graph: '{req.sentence}' through {req.model_id}, "
            f"{len(result.nodes)} nodes, {len(result.edges)} edges, "
            f"{len(result.paths)} paths"
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Trace graph failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
