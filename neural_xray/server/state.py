"""
Model Surgery MRI — Shared state for the API server.

Holds loaded models, concept vectors, traces, SAEs, and cartography maps in memory.
All API routes access state through this module.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("model-surgery")


@dataclass
class ModelSession:
    """A loaded model and its derived artifacts."""
    model_id: str
    loader: Any  # ModelLoader
    arch_map: Any  # ArchitectureMap
    concept_vectors: Dict[str, torch.Tensor] = field(default_factory=dict)
    relation_vectors: Dict[str, torch.Tensor] = field(default_factory=dict)
    traces: Dict[str, Any] = field(default_factory=dict)  # concept -> FlowTrace
    contamination_map: Optional[Any] = None
    cluster: Optional[Any] = None  # ConceptCluster
    sae: Optional[Any] = None  # SparseAutoencoder
    cartography_maps: Dict[str, Any] = field(default_factory=dict)  # concept -> ConceptMap
    cartographer: Optional[Any] = None  # LoRACartographer


class AppState:
    """Global mutable state for the MRI server."""

    def __init__(self):
        self.sessions: Dict[str, ModelSession] = {}
        self.lock = threading.Lock()
        # Terminal log buffer for WebSocket streaming
        self.log_lines: List[str] = []
        self.log_event = asyncio.Event()

    def get_session(self, model_id: str) -> ModelSession:
        if model_id not in self.sessions:
            raise KeyError(f"Model '{model_id}' is not loaded")
        return self.sessions[model_id]

    def add_log(self, line: str):
        self.log_lines.append(line)
        # Signal any waiting WebSocket consumers
        try:
            self.log_event.set()
        except RuntimeError:
            pass  # No event loop running yet


# Singleton
state = AppState()


class LogCapture(logging.Handler):
    """Captures log records into the shared state for WebSocket streaming."""

    def emit(self, record):
        msg = self.format(record)
        state.add_log(msg)
