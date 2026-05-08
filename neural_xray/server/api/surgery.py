"""
Model Surgery MRI — /api/surgery endpoints.

Cross-model knowledge transplantation: gap analysis, alignment, transplant, verify.
Includes: perplexity measurement, QA benchmarking, integrity validation, concept scanning.
"""

import asyncio
import gc
import logging
import math
import sys
import time
import traceback
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from neural_xray import LoRACartographer  # noqa: E402
from ..licensing import require_feature  # noqa: E402
from ..state import state  # noqa: E402

logger = logging.getLogger("model-surgery")
router = APIRouter()


class GapAnalysisRequest(BaseModel):
    donor_id: str
    patient_id: str
    concepts: Dict[str, List[str]]  # concept -> sentences


class AlignRequest(BaseModel):
    donor_id: str
    patient_id: str
    n_bins: int = 4


class TransplantRequest(BaseModel):
    donor_id: str
    patient_id: str
    concepts: List[str]
    scale: float = 0.05
    rank_k: int = 4
    interference_threshold: float = 0.7


class VerifyRequest(BaseModel):
    model_id: str
    concepts: List[str]
    sentences_per_concept: Dict[str, List[str]]


@router.post("/gap")
async def gap_analysis(req: GapAnalysisRequest):
    """Map concepts in both models and compute alignment gaps."""
    try:
        require_feature("surgery")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        donor_sess = state.get_session(req.donor_id)
        patient_sess = state.get_session(req.patient_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        # Map concepts in both models (run in thread to avoid blocking event loop)
        donor_carto = LoRACartographer(donor_sess.loader, donor_sess.arch_map, rank=4)
        patient_carto = LoRACartographer(patient_sess.loader, patient_sess.arch_map, rank=4)

        # Limit concepts to avoid OOM with two models in VRAM
        concept_dict = dict(list(req.concepts.items())[:3])
        concept_names = list(concept_dict.keys())

        logger.info(f"Mapping {len(concept_names)} concepts in donor ({req.donor_id})...")
        donor_maps = await asyncio.to_thread(donor_carto.fast_map_batch, concept_dict)

        # Free GPU memory before mapping patient
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Mapping {len(concept_names)} concepts in patient ({req.patient_id})...")
        patient_maps = await asyncio.to_thread(patient_carto.fast_map_batch, concept_dict)

        # Cache maps
        with state.lock:
            donor_sess.cartography_maps.update(donor_maps)
            donor_sess.cartographer = donor_carto
            patient_sess.cartography_maps.update(patient_maps)
            patient_sess.cartographer = patient_carto

        # Compare each concept
        gaps = {}
        for concept in concept_names:
            if concept in donor_maps and concept in patient_maps:
                alignment = LoRACartographer.compare_maps(
                    donor_maps[concept], patient_maps[concept]
                )
                gaps[concept] = {
                    "global_alignment": alignment.global_alignment,
                    "cosine_per_layer": {
                        k: v for k, v in sorted(
                            alignment.cosine_per_layer.items(),
                            key=lambda x: x[1],
                        )
                    },
                }

        logger.info(f"Gap analysis complete — {len(gaps)} concepts compared")
        return {
            "donor": req.donor_id,
            "patient": req.patient_id,
            "gaps": gaps,
        }

    except Exception as e:
        logger.error(f"Gap analysis failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/align")
async def build_alignment(req: AlignRequest):
    """Build layerwise alignment matrix between donor and patient."""
    try:
        require_feature("surgery")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        donor_sess = state.get_session(req.donor_id)
        patient_sess = state.get_session(req.patient_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not donor_sess.cartography_maps or not patient_sess.cartography_maps:
        raise HTTPException(
            status_code=400,
            detail="Run gap analysis first to map concepts in both models",
        )

    try:
        alignment = await asyncio.to_thread(
            LoRACartographer.build_layerwise_alignment,
            donor_sess.cartography_maps,
            patient_sess.cartography_maps,
            req.n_bins,
        )

        logger.info(f"Built layerwise alignment ({req.n_bins} bins) — "
                     f"residuals: {alignment.residuals_per_depth}")
        return {
            "donor": req.donor_id,
            "patient": req.patient_id,
            "n_bins": alignment.n_bins,
            "shared_concepts": alignment.shared_concepts,
            "residuals_per_depth": {k: float(v) for k, v in alignment.residuals_per_depth.items()},
        }

    except Exception as e:
        logger.error(f"Alignment failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transplant")
async def transplant_concepts(req: TransplantRequest):
    """Perform concept transplantation from donor to patient."""
    try:
        require_feature("surgery")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        donor_sess = state.get_session(req.donor_id)
        patient_sess = state.get_session(req.patient_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not patient_sess.cartographer:
        raise HTTPException(
            status_code=400,
            detail="No cartographer for patient — run gap analysis first",
        )

    try:
        # Build alignment if not already done
        alignment = await asyncio.to_thread(
            LoRACartographer.build_layerwise_alignment,
            donor_sess.cartography_maps,
            patient_sess.cartography_maps,
        )

        def _do_transplant():
            reports = {}
            for concept in req.concepts:
                donor_map = donor_sess.cartography_maps.get(concept)
                if donor_map is None:
                    logger.warning(f"Skipping '{concept}' — not mapped in donor")
                    continue

                # Check interference before transplant
                interference = patient_sess.cartographer.check_interference(
                    donor_map, alignment, patient_sess.cartography_maps,
                    threshold_abort=req.interference_threshold,
                )

                if interference.rating == "abort":
                    reports[concept] = {
                        "status": "aborted",
                        "reason": f"Interference too high (max cosine: {interference.max_cosine:.3f})",
                        "risky_layers": interference.risky_layers,
                    }
                    continue

                report = patient_sess.cartographer.transplant_concept(
                    concept_map=donor_map,
                    alignment=alignment,
                    rank_k=req.rank_k,
                    scale=req.scale,
                    existing_maps=patient_sess.cartography_maps,
                )

                reports[concept] = {
                    "status": "transplanted",
                    "layers_edited": report.layers_edited,
                    "total_delta_norm": float(report.total_delta_norm),
                    "edit_norms": {k: float(v) for k, v in report.edit_norms.items()},
                    "interference_rating": interference.rating,
                    "post_probe_alignment": (
                        float(report.post_probe_alignment)
                        if report.post_probe_alignment is not None else None
                    ),
                }
            return reports

        reports = await asyncio.to_thread(_do_transplant)

        logger.info(f"Transplant complete — {len(reports)} concepts processed")
        return {
            "donor": req.donor_id,
            "patient": req.patient_id,
            "scale": req.scale,
            "reports": reports,
        }

    except Exception as e:
        logger.error(f"Transplant failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Perplexity Measurement ───────────────────────────────

class PerplexityRequest(BaseModel):
    model_id: str
    sentences: List[str]


def _compute_perplexity(model, tokenizer, sentences: List[str]) -> dict:
    """Compute perplexity of sentences under a model."""
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    per_sentence = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss.float().item()
            n_tok = input_ids.shape[1] - 1

        ppl = math.exp(min(loss, 100))  # clamp to avoid overflow
        per_sentence.append({"sentence": sent[:100], "perplexity": round(ppl, 2), "loss": round(loss, 4)})
        total_loss += loss * n_tok
        total_tokens += n_tok

    avg_ppl = math.exp(min(total_loss / max(total_tokens, 1), 100))
    return {
        "avg_perplexity": round(avg_ppl, 2),
        "total_tokens": total_tokens,
        "per_sentence": per_sentence,
    }


@router.post("/perplexity")
async def measure_perplexity(req: PerplexityRequest):
    """Measure model perplexity on domain-specific text."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = await asyncio.to_thread(
            _compute_perplexity, sess.loader.model, sess.loader.tokenizer, req.sentences
        )
        result["model_id"] = req.model_id
        logger.info(f"Perplexity for {req.model_id}: {result['avg_perplexity']}")
        return result
    except Exception as e:
        logger.error(f"Perplexity failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── QA Benchmark ─────────────────────────────────────────

class QABenchmarkRequest(BaseModel):
    model_id: str
    questions: List[Dict]  # [{"prompt": ..., "expected_keywords": [...]}]
    max_tokens: int = 40
    temperature: float = 0.7


def _run_qa_benchmark(model, tokenizer, questions: list, max_tokens: int, temperature: float) -> dict:
    """Run QA benchmark: generate answers and check for expected keywords."""
    device = next(model.parameters()).device
    results = []
    hits = 0

    for q in questions:
        prompt = q["prompt"]
        expected = [kw.lower() for kw in q.get("expected_keywords", [])]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_lower = generated.lower()
        found = [kw for kw in expected if kw in gen_lower]
        hit = len(found) > 0

        if hit:
            hits += 1
        results.append({
            "prompt": prompt,
            "generated": generated[:200],
            "expected_keywords": expected,
            "found_keywords": found,
            "hit": hit,
        })

    accuracy = hits / max(len(questions), 1)
    return {
        "accuracy": round(accuracy, 4),
        "hits": hits,
        "total": len(questions),
        "results": results,
    }


@router.post("/qa_benchmark")
async def qa_benchmark(req: QABenchmarkRequest):
    """Run QA benchmark on a model."""
    try:
        sess = state.get_session(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = await asyncio.to_thread(
            _run_qa_benchmark, sess.loader.model, sess.loader.tokenizer,
            req.questions, req.max_tokens, req.temperature
        )
        result["model_id"] = req.model_id
        logger.info(f"QA benchmark {req.model_id}: {result['accuracy']*100:.1f}% ({result['hits']}/{result['total']})")
        return result
    except Exception as e:
        logger.error(f"QA benchmark failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Full Transplant Validation ───────────────────────────

class ValidateTransplantRequest(BaseModel):
    model_id: str
    transplanted_concepts: List[str]
    domain_sentences: List[str]  # domain text for perplexity
    qa_questions: List[Dict]     # QA pairs
    general_sentences: List[str]  # general text for integrity check
    probe_sentences: Optional[Dict[str, List[str]]] = None  # concept -> sentences


@router.post("/validate")
async def validate_transplant(req: ValidateTransplantRequest):
    """
    Full post-transplant validation suite:
    1. Domain perplexity (did the model get better at domain text?)
    2. QA benchmark (can the model answer domain questions?)
    3. General integrity (did general capability degrade?)
    4. Concept probes (are concepts measurably stronger?)
    """
    try:
        sess = state.get_session(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    t0 = time.time()

    try:
        model = sess.loader.model
        tokenizer = sess.loader.tokenizer

        # 1. Domain perplexity
        domain_ppl = await asyncio.to_thread(
            _compute_perplexity, model, tokenizer, req.domain_sentences
        )

        # 2. QA benchmark
        qa = await asyncio.to_thread(
            _run_qa_benchmark, model, tokenizer, req.qa_questions, 40, 0.7
        )

        # 3. General integrity (perplexity on general text — should stay same)
        general_ppl = await asyncio.to_thread(
            _compute_perplexity, model, tokenizer, req.general_sentences
        )

        # 4. Concept probes
        probes = {}
        if req.probe_sentences:
            device = next(model.parameters()).device
            for concept_name, sents in req.probe_sentences.items():
                concept_vec = sess.concept_vectors.get(concept_name)
                if concept_vec is None:
                    continue
                concept_vec = concept_vec.to(device)
                scores = []
                for sent in sents:
                    inputs = tokenizer(sent, return_tensors="pt", truncation=True).to(device)
                    with torch.no_grad():
                        out = model(**inputs, output_hidden_states=True)
                        # Average all hidden states
                        hidden = torch.stack(out.hidden_states, dim=0).mean(dim=[0, 2])  # [hidden]
                        sim = F.cosine_similarity(hidden.float(), concept_vec.float().unsqueeze(0)).item()
                    scores.append({"sentence": sent[:100], "similarity": round(sim, 4)})
                avg_sim = sum(s["similarity"] for s in scores) / max(len(scores), 1)
                probes[concept_name] = {"avg_similarity": round(avg_sim, 4), "scores": scores}

        elapsed = time.time() - t0

        return {
            "model_id": req.model_id,
            "validation_time_seconds": round(elapsed, 2),
            "domain_perplexity": domain_ppl,
            "qa_benchmark": qa,
            "general_integrity": general_ppl,
            "concept_probes": probes,
            "transplanted_concepts": req.transplanted_concepts,
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Model Diagnostic Scan ────────────────────────────────

class ScanRequest(BaseModel):
    donor_id: str
    patient_id: str
    concept_packs: List[str] = ["Science", "Emotions", "Abstract"]


# Built-in concept packs for scanning
SCAN_CONCEPTS = {
    "Science": [
        "atom", "molecule", "electron", "gravity", "energy", "photon",
        "cell", "DNA", "protein", "evolution", "species", "ecosystem",
        "velocity", "mass", "force", "temperature", "pressure", "wave",
        "chemical", "reaction", "element", "compound", "nucleus", "ion",
    ],
    "Emotions": [
        "anger", "joy", "sadness", "fear", "surprise", "disgust",
        "anxiety", "love", "hate", "trust", "hope", "shame",
        "pride", "jealousy", "empathy", "grief", "contempt", "awe",
    ],
    "Abstract": [
        "justice", "freedom", "democracy", "truth", "beauty", "time",
        "consciousness", "morality", "knowledge", "power", "identity",
        "causation", "infinity", "entropy", "probability", "logic",
    ],
    "Technology": [
        "algorithm", "software", "network", "database", "encryption",
        "processor", "memory", "bandwidth", "protocol", "latency",
        "compiler", "server", "interface", "API", "thread", "cache",
    ],
    "Medicine": [
        "diagnosis", "treatment", "symptom", "disease", "vaccine",
        "antibody", "inflammation", "surgery", "therapy", "pathogen",
        "immune", "organ", "tissue", "chronic", "acute", "prognosis",
    ],
    "Economics": [
        "inflation", "market", "supply", "demand", "capital",
        "investment", "currency", "trade", "recession", "growth",
        "profit", "labor", "monopoly", "subsidy", "tariff", "debt",
    ],
}


@router.post("/scan")
async def scan_model_gaps(req: ScanRequest):
    """
    Scan a patient model against a donor to find all knowledge gaps.
    Returns a ranked list of concepts the patient is weakest in.
    """
    try:
        require_feature("surgery")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    try:
        donor_sess = state.get_session(req.donor_id)
        patient_sess = state.get_session(req.patient_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    t0 = time.time()

    try:
        # Collect all concepts to scan
        all_concepts = {}
        for pack_name in req.concept_packs:
            concepts = SCAN_CONCEPTS.get(pack_name, [])
            for c in concepts:
                all_concepts[c] = [f"The {c} is a fundamental concept.", f"Understanding {c} is important."]

        if not all_concepts:
            raise HTTPException(status_code=400, detail="No valid concept packs selected")

        # Process in batches to avoid OOM
        batch_size = 5
        concept_items = list(all_concepts.items())
        all_gaps = {}

        donor_carto = LoRACartographer(donor_sess.loader, donor_sess.arch_map, rank=4)
        patient_carto = LoRACartographer(patient_sess.loader, patient_sess.arch_map, rank=4)

        for i in range(0, len(concept_items), batch_size):
            batch = dict(concept_items[i:i + batch_size])
            batch_names = list(batch.keys())

            logger.info(f"Scanning batch {i // batch_size + 1}: {batch_names}")

            try:
                donor_maps = await asyncio.to_thread(donor_carto.fast_map_batch, batch)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                patient_maps = await asyncio.to_thread(patient_carto.fast_map_batch, batch)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for concept in batch_names:
                    if concept in donor_maps and concept in patient_maps:
                        alignment = LoRACartographer.compare_maps(
                            donor_maps[concept], patient_maps[concept]
                        )
                        all_gaps[concept] = {
                            "global_alignment": round(alignment.global_alignment, 4),
                            "knowledge_gap_pct": round((1 - alignment.global_alignment) * 100, 1),
                        }
            except Exception as batch_err:
                logger.warning(f"Scan batch failed: {batch_err}")
                for concept in batch_names:
                    all_gaps[concept] = {"global_alignment": 0, "knowledge_gap_pct": 100, "error": str(batch_err)}

        # Cache for future transplants
        with state.lock:
            donor_sess.cartographer = donor_carto
            patient_sess.cartographer = patient_carto

        # Sort by gap (biggest gaps first)
        ranked = sorted(all_gaps.items(), key=lambda x: x[1].get("knowledge_gap_pct", 0), reverse=True)

        # Categorize
        critical = [{"concept": c, **v} for c, v in ranked if v.get("knowledge_gap_pct", 0) > 15]
        moderate = [{"concept": c, **v} for c, v in ranked if 5 < v.get("knowledge_gap_pct", 0) <= 15]
        healthy = [{"concept": c, **v} for c, v in ranked if v.get("knowledge_gap_pct", 0) <= 5]

        elapsed = time.time() - t0

        # Pack breakdown
        pack_stats = {}
        for pack_name in req.concept_packs:
            concepts_in_pack = SCAN_CONCEPTS.get(pack_name, [])
            pack_gaps = [all_gaps[c]["knowledge_gap_pct"] for c in concepts_in_pack if c in all_gaps]
            if pack_gaps:
                pack_stats[pack_name] = {
                    "avg_gap_pct": round(sum(pack_gaps) / len(pack_gaps), 1),
                    "max_gap_pct": round(max(pack_gaps), 1),
                    "concepts_scanned": len(pack_gaps),
                    "critical_count": sum(1 for g in pack_gaps if g > 15),
                }

        return {
            "donor": req.donor_id,
            "patient": req.patient_id,
            "scan_time_seconds": round(elapsed, 2),
            "total_concepts_scanned": len(all_gaps),
            "summary": {
                "critical": len(critical),
                "moderate": len(moderate),
                "healthy": len(healthy),
                "avg_gap_pct": round(sum(v["knowledge_gap_pct"] for v in all_gaps.values()) / max(len(all_gaps), 1), 1),
            },
            "pack_breakdown": pack_stats,
            "critical_gaps": critical,
            "moderate_gaps": moderate,
            "healthy_concepts": healthy,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Available Scan Packs ─────────────────────────────────

@router.get("/scan_packs")
async def list_scan_packs():
    """Return available concept packs for scanning."""
    return {
        pack: {"count": len(concepts), "sample": concepts[:5]}
        for pack, concepts in SCAN_CONCEPTS.items()
    }
