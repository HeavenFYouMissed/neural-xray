"""
neural-xray — X-ray vision into any LLM
========================================

X-ray vision into any transformer model. Works on CPU, no GPU required.
Inspect concept flow, run diagnostics, extract knowledge vectors,
and perform surgical modifications.

Pipeline stages:
    1. LOAD      — Load any HuggingFace model with auto quantization
    2. MAP       — Discover layer structure: hidden dims, MLP patterns, layer count
    3. LOCALIZE  — Find WHERE specific facts live (causal tracing)
    4. EXTRACT   — Pull concept vectors via activation hooks
    5. CLUSTER   — Build semantic map: which concepts group together
    6. BLUEPRINT — Save structured JSON description of knowledge geometry
    7. PROJECT   — Train a tiny projection layer: source dim → target dim
    8. TRANSPLANT— Write extracted knowledge into target model weights directly
    9. VERIFY    — Confirm transplant preserved semantic relationships

Quick start:
    from neural_xray import ModelLoader, ConceptFlowTracer, ModelDiagnostics

    loader = ModelLoader("gpt2", force_quantization="float32")
    loader.load()

    tracer = ConceptFlowTracer(loader)
    trace = tracer.trace("fire")

    diag = ModelDiagnostics(loader)
    report = diag.run_all()

CLI:
    neural-xray trace --model gpt2 --concept fire
    neural-xray diagnose --model gpt2
    neural-xray extract --model gpt2 --concepts fire water gravity
    neural-xray visualize --model gpt2 --output viz.html
"""

__version__ = "0.1.0"

from .loader import ModelLoader
from .extractor import ConceptExtractor
from .cluster import ConceptCluster
from .blueprint import NeuralBlueprint
from .projector import ConceptProjector
from .transplanter import KnowledgeTransplanter
from .tracer import ConceptFlowTracer, FlowTrace, ContaminationMap, DeadZone, DeadZoneAnalysis, LogitLensResult, LogitLensLayer, TraceGraph, TraceNode, TraceEdge, TracePath
from .diagnostics import ModelDiagnostics, DiagnosticReport, CheckResult
from .evidence import EvidenceCollector, EvidencePackage
from .sae import SparseAutoencoder, get_target_layer
from .visualizer import NeuralVisualizer
from .cartography import (
    LoRACartographer,
    ConceptMap,
    ConceptFrame,
    CartographyAlignment,
    ConceptAlignmentMatrix,
    LayerwiseAlignment,
    InterferenceReport,
    TransplantReport,
)

__all__ = [
    "ModelLoader",
    "ArchitectureMapper",
    "ConceptExtractor",
    "ConceptCluster",
    "NeuralBlueprint",
    "ConceptProjector",
    "KnowledgeTransplanter",
    "ConceptFlowTracer",
    "FlowTrace",
    "ContaminationMap",
    "DeadZone",
    "DeadZoneAnalysis",
    "SparseAutoencoder",
    "get_target_layer",
    "NeuralVisualizer",
    # cartography
    "LoRACartographer",
    "ConceptMap",
    "ConceptFrame",
    "CartographyAlignment",
    "ConceptAlignmentMatrix",
    "LayerwiseAlignment",
    "InterferenceReport",
    "TransplantReport",
    # evidence
    "EvidenceCollector",
    "EvidencePackage",
]
