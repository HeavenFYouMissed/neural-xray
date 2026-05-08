"""
evidence.py — Evidence Packaging (AutoPsy: Forensics)
=====================================================

Captures a complete snapshot of model behavior on a given input:
logit lens + trace graph + diagnostics. Packages everything into
a JSON artifact that can be saved, compared, or shared.

Usage:
    from antroslammer.autopsy.evidence import EvidenceCollector

    collector = EvidenceCollector(loader, arch_map)
    package = collector.capture("The cat sat on the mat")
    collector.save(package, "evidence/snapshot_001.json")
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .tracer import ConceptFlowTracer
from .diagnostics import ModelDiagnostics


@dataclass
class EvidencePackage:
    """A complete behavioral snapshot of a model on one input."""
    id: str
    created_at: str
    model_id: str
    sentence: str
    logit_lens: Optional[Dict[str, Any]] = None
    trace_graph: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "sentence": self.sentence,
            "logit_lens": self.logit_lens,
            "trace_graph": self.trace_graph,
            "diagnostics": self.diagnostics,
            "annotations": self.annotations,
        }


class EvidenceCollector:
    """
    Captures logit lens + trace graph + diagnostics in one call.
    Packages them into a serializable EvidencePackage.
    """

    def __init__(self, loader, arch_map, concept_vectors=None):
        self.loader = loader
        self.arch_map = arch_map
        self.concept_vectors = concept_vectors or {}

    def capture(self, sentence: str, model_id: str = "") -> EvidencePackage:
        """
        Run all analyses and package results.

        Args:
            sentence: Input text to analyze
            model_id: Label for the model (for metadata)

        Returns:
            EvidencePackage with all results
        """
        pkg = EvidencePackage(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
            model_id=model_id or getattr(self.loader, 'model_name', 'unknown'),
            sentence=sentence,
        )

        # 1. Logit Lens
        try:
            tracer = ConceptFlowTracer(
                loader=self.loader,
                arch_map=self.arch_map,
                concept_vectors=self.concept_vectors,
            )
            lens_result = tracer.logit_lens(sentence)
            pkg.logit_lens = lens_result.to_dict()
        except Exception as e:
            pkg.annotations["logit_lens_error"] = str(e)

        # 2. Trace Graph
        try:
            tracer = ConceptFlowTracer(
                loader=self.loader,
                arch_map=self.arch_map,
                concept_vectors=self.concept_vectors,
            )
            graph_result = tracer.trace_sentence(sentence)
            pkg.trace_graph = graph_result.to_dict()
        except Exception as e:
            pkg.annotations["trace_graph_error"] = str(e)

        # 3. Diagnostics
        try:
            diag = ModelDiagnostics(
                loader=self.loader,
                arch_map=self.arch_map,
            )
            report = diag.run_checks(sentence)
            pkg.diagnostics = report.to_dict()
        except Exception as e:
            pkg.annotations["diagnostics_error"] = str(e)

        return pkg

    def save(self, package: EvidencePackage, path: str) -> Path:
        """Save an evidence package to disk as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(package.to_dict(), indent=2), encoding="utf-8")
        return out

    @staticmethod
    def load(path: str) -> EvidencePackage:
        """Load an evidence package from disk."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return EvidencePackage(
            id=data["id"],
            created_at=data["created_at"],
            model_id=data["model_id"],
            sentence=data["sentence"],
            logit_lens=data.get("logit_lens"),
            trace_graph=data.get("trace_graph"),
            diagnostics=data.get("diagnostics"),
            annotations=data.get("annotations", {}),
        )
