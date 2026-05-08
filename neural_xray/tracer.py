"""
tracer.py — ConceptFlowTracer (AutoPsy Stage 3b: TRACE)
========================================================

The "surgical dye" stage.

Biology: inject radioactive dye → watch it flow through veins/arteries,
         lighting up every structure it passes through.

Here:    inject a concept token → watch how it "contaminates" every layer,
         lighting up related concepts as the signal propagates forward.

This tells you:
    - WHERE in the model a concept lives (which layers "own" it)
    - WHAT it activates along the way (the associative chain)
    - HOW it transforms as it goes deeper (from surface form → semantic → role)
    - WHICH concepts always co-activate (implicit relationships)

Technical basis: Activation patching / causal tracing (ROME, MEMIT, Elhage 2021)

Example output for concept "fire":
    Layer  0 [EMBED ] fire(0.99)  heat(0.62)  burn(0.55)
    Layer  1 [ATTN  ] fire(0.94)  flame(0.71)  light(0.60)
    Layer  4 [MLP   ] heat(0.88)  fire(0.82)  warmth(0.79)    ← concept shift
    Layer  8 [MLP   ] danger(0.85)  damage(0.77)  burn(0.74)  ← deeper abstraction
    Layer 12 [MLP   ] cause(0.81)  trigger(0.76)  effect(0.69) ← causal role

Usage:
    from antroslammer.autopsy.tracer import ConceptFlowTracer

    tracer = ConceptFlowTracer(loader, arch_map, concept_vectors)
    trace = tracer.trace("fire")
    tracer.print_trace(trace)
    tracer.save_trace(trace, "autopsy_output/traces/fire.json")

    # Batch trace all entities:
    traces = tracer.trace_all(entity_names, top_k=5)
    map_ = tracer.build_contamination_map(traces)
    tracer.save_contamination_map(map_, "autopsy_output/contamination_map.json")
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

# SAE import is optional — tracer works without it (falls back to cosine similarity)
try:
    from .sae import SparseAutoencoder
except ImportError:
    SparseAutoencoder = None  # type: ignore


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class LayerActivation:
    """A single layer's activation state — what concepts lit up here."""
    layer_index: int
    layer_type: str          # "embed", "attention", "mlp", "residual"
    module_path: str
    # Top-K concepts that match this activation state (cosine similarity)
    top_concepts: List[Tuple[str, float]] = field(default_factory=list)
    # Raw mean activation norm (signal strength at this layer)
    activation_norm: float = 0.0
    # Dominant concept (top-1)
    dominant: str = ""
    dominant_sim: float = 0.0
    # L2-normalized hidden state direction (CPU tensor) — for dead zone analysis.
    # Not serialized to JSON; populated during trace().
    hidden_dir: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)


# ─── Dead zone analysis data structures ──────────────────────────────────────

@dataclass
class DeadZone:
    """A stretch of layers where concept signal dropped below detection threshold."""
    start_layer: int        # first dead layer index
    end_layer: int          # last dead layer index
    entry_layer: int        # last active layer before this zone
    exit_layer: int         # first active layer after this zone
    entry_concept: str      # dominant concept entering
    exit_concept: str       # dominant concept exiting
    entry_sim: float
    exit_sim: float
    # Trajectory shift: 0 = same direction (passive), 1–2 = changed (active)
    trajectory_shift: float
    # True when trajectory_shift exceeds active_threshold — concept transformed here
    is_active: bool
    # Per-dead-layer: (layer_idx, predicted_concept, predicted_sim) via slerp
    interpolated: List[Tuple[int, str, float]] = field(default_factory=list)
    # The specific layer within the zone with the maximum deviation from slerp path
    causal_layer: Optional[int] = None


@dataclass
class DeadZoneAnalysis:
    """Results of dead zone detection and trajectory interpolation for one trace."""
    concept: str
    total_layers: int
    dead_threshold: float
    zones: List[DeadZone] = field(default_factory=list)
    active_zones: List[DeadZone] = field(default_factory=list)
    # Minimum cut: (layer_idx, layer_type, module_path, shift_score)
    minimum_cut_layers: List[Tuple[int, str, str, float]] = field(default_factory=list)


@dataclass
class LogitLensLayer:
    """Logit lens prediction at a single layer."""
    layer_index: int
    layer_type: str          # "residual", "attention", "mlp"
    module_path: str
    top_tokens: List[Tuple[str, float]]  # (token_str, probability)


@dataclass
class LogitLensResult:
    """Complete logit lens output for a sentence through all layers."""
    sentence: str
    input_tokens: List[str]
    final_prediction: str
    layers: List[LogitLensLayer] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "input_tokens": self.input_tokens,
            "final_prediction": self.final_prediction,
            "layers": [
                {
                    "layer_index": ll.layer_index,
                    "layer_type": ll.layer_type,
                    "module_path": ll.module_path,
                    "top_tokens": [
                        {"token": t, "prob": round(p, 6)} for t, p in ll.top_tokens
                    ],
                }
                for ll in self.layers
            ],
        }


# ─── Sentence Trace (Per-Token) data structures ─────────────────────────────

@dataclass
class TraceNode:
    """A single node in the trace graph: one token at one layer."""
    id: str               # e.g. "b3_mlp_t2"
    block: int            # transformer block index
    layer_type: str       # "attention", "mlp", "residual"
    token_idx: int        # position in input sequence
    token_str: str        # decoded token text
    norm: float           # activation L2 norm at this point
    top_tokens: List[Tuple[str, float]] = field(default_factory=list)  # logit lens predictions


@dataclass
class TraceEdge:
    """An edge in the trace graph: connection between two nodes."""
    source: str           # source node id
    target: str           # target node id
    weight: float         # edge strength
    edge_type: str        # "attention" or "residual"
    head: Optional[int] = None  # attention head index (if attention edge)


@dataclass
class TracePath:
    """A high-weight path through the trace graph (beam search result)."""
    id: str
    nodes: List[str]      # ordered list of node ids
    score: float          # cumulative path weight
    method: str = "beam"


@dataclass
class TraceGraph:
    """
    Complete per-token trace through all model layers.
    This is the data structure that feeds the tournament bracket visualization.
    """
    sentence: str
    input_tokens: List[str]
    num_blocks: int
    nodes: List[TraceNode] = field(default_factory=list)
    edges: List[TraceEdge] = field(default_factory=list)
    paths: List[TracePath] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "input_tokens": self.input_tokens,
            "num_blocks": self.num_blocks,
            "nodes": [
                {
                    "id": n.id,
                    "block": n.block,
                    "layer_type": n.layer_type,
                    "token_idx": n.token_idx,
                    "token_str": n.token_str,
                    "norm": round(n.norm, 4),
                    "top_tokens": [
                        {"token": t, "prob": round(p, 6)} for t, p in n.top_tokens
                    ],
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": round(e.weight, 6),
                    "edge_type": e.edge_type,
                    **({"head": e.head} if e.head is not None else {}),
                }
                for e in self.edges
            ],
            "paths": [
                {"id": p.id, "nodes": p.nodes, "score": round(p.score, 6), "method": p.method}
                for p in self.paths
            ],
        }


@dataclass
class FlowTrace:
    """Complete dye-trace of one concept through all model layers."""
    concept: str
    num_layers: int
    # Full per-layer activation record
    layers: List[LayerActivation] = field(default_factory=list)
    # contamination_map[concept_name] = list of layer indices where it appeared in top-K
    contamination_map: Dict[str, List[int]] = field(default_factory=dict)
    # At what layer does a NEW concept first become dominant? (concept shift points)
    shift_points: List[Tuple[int, str, str]] = field(default_factory=list)  # (layer, from, to)
    # The complete associative chain: sequence of dominant concepts over depth
    dominant_chain: List[str] = field(default_factory=list)
    # Dead zone analysis — populated by analyze_dead_zones() (not serialized)
    dead_zones: Optional["DeadZoneAnalysis"] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "num_layers": self.num_layers,
            "dominant_chain": self.dominant_chain,
            "shift_points": [
                {"layer": s[0], "from": s[1], "to": s[2]} for s in self.shift_points
            ],
            "contamination_map": self.contamination_map,
            "layers": [
                {
                    "layer_index": la.layer_index,
                    "layer_type": la.layer_type,
                    "module_path": la.module_path,
                    "dominant": la.dominant,
                    "dominant_sim": round(la.dominant_sim, 4),
                    "activation_norm": round(la.activation_norm, 4),
                    "top_concepts": [(c, round(s, 4)) for c, s in la.top_concepts],
                }
                for la in self.layers
            ],
        }


@dataclass
class ContaminationMap:
    """
    Cross-concept contamination summary.

    For every concept pair (A, B): at how many layers did running
    concept A also light up concept B? High score = strong association.
    """
    concepts: List[str]
    # co_activation[A][B] = fraction of layers where B appeared when A was injected
    co_activation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # For each concept, its "influence radius" (how many other concepts it activates)
    influence_radius: Dict[str, int] = field(default_factory=dict)
    # The viral spreaders: concepts with the highest influence radius
    top_spreaders: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "concepts": self.concepts,
            "top_spreaders": self.top_spreaders,
            "influence_radius": self.influence_radius,
            "co_activation": self.co_activation,
        }


# ─── Core tracer ──────────────────────────────────────────────────────────────

class ConceptFlowTracer:
    """
    Inject a concept and trace how it propagates (contaminates) through
    every layer of the model, lighting up related concepts as it goes.

    Args:
        loader:           Loaded ModelLoader instance
        arch_map:         ArchitectureMap from the mapper stage
        concept_vectors:  Dict[concept_name → tensor(hidden_dim)] from extractor
        top_k:            How many concepts to report per layer (default 5)
        sim_threshold:    Min cosine similarity to count as "contaminated" (default 0.50)
        device:           "cuda" / "cpu"
        sae:              Optional trained SparseAutoencoder. When provided, concept
                          decomposition switches from raw cosine similarity to SAE
                          feature decomposition — giving monosemantic, clean results.
    """

    def __init__(
        self,
        loader,
        arch_map,
        concept_vectors: Dict[str, torch.Tensor],
        top_k: int = 5,
        sim_threshold: float = 0.50,
        device: Optional[str] = None,
        sae=None,
    ):
        self.loader = loader
        self.arch_map = arch_map
        self.top_k = top_k
        self.sim_threshold = sim_threshold
        self.sae = sae  # Optional[SparseAutoencoder]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Pre-build concept matrix for fast cosine similarity: [N, D]
        # Normalize once, then trace is a single matmul per layer
        self._concept_names = list(concept_vectors.keys())
        self._concept_vectors = concept_vectors      # kept for SAE alignment
        if concept_vectors:
            vecs = torch.stack([
                F.normalize(v.float(), dim=-1) for v in concept_vectors.values()
            ])  # [N, D]
            self.concept_matrix = vecs.to(self.device)  # kept on device
        else:
            self.concept_matrix = None  # logit lens mode — no concepts needed

        # If SAE provided, pre-build feature → concept alignment
        self._feature_to_concept: Optional[Dict[int, List[Tuple[str, float]]]] = None
        if sae is not None:
            print("[Tracer] SAE detected — pre-building feature→concept alignment…")
            self._feature_to_concept = sae.align_features_to_concepts(
                concept_vectors, top_k=top_k
            )
            print(f"[Tracer] Ready (SAE mode). {len(self._concept_names)} concepts, "
                  f"{sae.n_features} features, sim_threshold={sim_threshold}")
        else:
            print(f"[Tracer] Ready (cosine mode). {len(self._concept_names)} concepts, "
                  f"sim_threshold={sim_threshold}")

    # ── Public API ────────────────────────────────────────────────────────────

    def trace(self, concept: str) -> FlowTrace:
        """
        Run the full dye trace for one concept.
        Registers hooks on every layer, does one forward pass, collects results.
        """
        model = self.loader.model
        tokenizer = self.loader.tokenizer

        # ── 1. Tokenize concept ────────────────────────────────────────────
        prompt = f"The concept of {concept} means"   # richer prompt = better activation
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # ── 2. Register hooks using arch_map (architecture-agnostic) ────
        #
        # Use the ArchitectureMapper's discovered layer info — exact module
        # paths for every attention, MLP, and residual checkpoint regardless
        # of model family (GPT-2, Llama, Qwen, Falcon, Phi, T5, Mistral …).
        activations: List[Tuple[int, str, str, torch.Tensor]] = []
        # (layer_index, layer_type, module_path, activation_tensor)
        handles = []

        def _make_hook(layer_idx, ltype, path):
            def _hook(module, inp, out):
                # out may be a tuple (e.g. attention returns (out, weights, ...))
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                    # Average over sequence dim → [hidden_size]
                    pooled = tensor.detach().float().mean(dim=-2)
                    # If batch dim still present, take first
                    if pooled.ndim == 2:
                        pooled = pooled[0]
                    activations.append((layer_idx, ltype, path, pooled.to(self.device)))
            return _hook

        def _resolve_module(path: str):
            """Walk dotted path to get the actual nn.Module."""
            parts = path.split(".")
            mod = model
            for p in parts:
                if p.isdigit():
                    mod = mod[int(p)]
                else:
                    mod = getattr(mod, p, None)
                if mod is None:
                    return None
            return mod

        layer_counter = 0
        for layer_info in self.arch_map.layers:
            # Residual (full block output)
            block_mod = _resolve_module(layer_info.name)
            if block_mod is not None:
                h = block_mod.register_forward_hook(
                    _make_hook(layer_counter, "residual", layer_info.name))
                handles.append(h)
                layer_counter += 1

            # Attention
            if layer_info.attn_path:
                attn_mod = _resolve_module(layer_info.attn_path)
                if attn_mod is not None:
                    h = attn_mod.register_forward_hook(
                        _make_hook(layer_counter, "attention", layer_info.attn_path))
                    handles.append(h)
                    layer_counter += 1

            # MLP
            if layer_info.mlp_path:
                mlp_mod = _resolve_module(layer_info.mlp_path)
                if mlp_mod is not None:
                    h = mlp_mod.register_forward_hook(
                        _make_hook(layer_counter, "mlp", layer_info.mlp_path))
                    handles.append(h)
                    layer_counter += 1

        # ── 3. Forward pass ───────────────────────────────────────────────
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # ── 4. Build FlowTrace from collected activations ─────────────────
        trace = FlowTrace(concept=concept, num_layers=len(activations))

        prev_dominant = concept
        for layer_idx, ltype, path, act in activations:
            top_concepts, norm = self._find_top_concepts(act)
            dominant, dominant_sim = (top_concepts[0] if top_concepts
                                      else (concept, 0.0))
            la = LayerActivation(
                layer_index=layer_idx,
                layer_type=ltype,
                module_path=path,
                top_concepts=top_concepts,
                activation_norm=norm,
                dominant=dominant,
                dominant_sim=dominant_sim,
                hidden_dir=F.normalize(act.float(), dim=-1).cpu(),
            )
            trace.layers.append(la)
            trace.dominant_chain.append(dominant)

            # Record shift points (where dominant concept changes)
            if dominant != prev_dominant:
                trace.shift_points.append((layer_idx, prev_dominant, dominant))
                prev_dominant = dominant

            # Update contamination map
            for c, sim in top_concepts:
                if sim >= self.sim_threshold:
                    if c not in trace.contamination_map:
                        trace.contamination_map[c] = []
                    trace.contamination_map[c].append(layer_idx)

        return trace

    def trace_all(
        self,
        concepts: List[str],
        verbose: bool = True,
    ) -> Dict[str, FlowTrace]:
        """Trace all concepts. Returns {concept: FlowTrace}."""
        traces = {}
        n = len(concepts)
        for i, concept in enumerate(concepts):
            if verbose and i % 20 == 0:
                print(f"  [Tracer] {i}/{n} — tracing '{concept}'...")
            try:
                traces[concept] = self.trace(concept)
            except Exception as e:
                print(f"  [Tracer] WARNING: failed to trace '{concept}': {e}")
        print(f"  [Tracer] Done. Traced {len(traces)}/{n} concepts.")
        return traces

    def build_contamination_map(self, traces: Dict[str, FlowTrace]) -> ContaminationMap:
        """
        Build cross-concept contamination matrix from a set of traces.

        co_activation[A][B] = fraction of A's layers where B also appeared in top-K.
        High value = B is consistently activated when A is processed = strong association.
        """
        concepts = list(traces.keys())
        co = {c: {} for c in concepts}
        influence_radius = {}

        for src_concept, trace in traces.items():
            n_layers = max(len(trace.layers), 1)
            # Count how many layers each co-activated concept appeared in
            contamination = trace.contamination_map
            for activated_concept, layer_list in contamination.items():
                if activated_concept == src_concept:
                    continue
                fraction = len(layer_list) / n_layers
                co[src_concept][activated_concept] = round(fraction, 4)

            # Influence radius = number of OTHER concepts activated above threshold
            influence_radius[src_concept] = len([
                c for c, layers in contamination.items()
                if c != src_concept and len(layers) >= 2   # appeared in at least 2 layers
            ])

        top_spreaders = sorted(
            influence_radius.items(), key=lambda x: x[1], reverse=True
        )[:50]

        return ContaminationMap(
            concepts=concepts,
            co_activation=co,
            influence_radius=influence_radius,
            top_spreaders=top_spreaders,
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def print_trace(self, trace: FlowTrace, condensed: bool = True):
        """Print a human-readable trace. condensed=True only shows shift points."""
        print(f"\n{'='*65}")
        print(f"  DYE TRACE: '{trace.concept}'")
        print(f"{'='*65}")
        print(f"  Layers tracked:  {trace.num_layers}")
        print(f"  Concepts lit up: {len(trace.contamination_map)}")
        print(f"  Shift points:    {len(trace.shift_points)}")
        print()

        if condensed:
            # Show dominant chain summary
            chain = trace.dominant_chain
            # Compress consecutive same-concept runs
            compressed = []
            prev = None
            for c in chain:
                if c != prev:
                    compressed.append(c)
                    prev = c
            print(f"  DOMINANT FLOW: {' → '.join(compressed)}")
            print()

            # Show each shift point
            print(f"  SHIFT POINTS (where concept changes):")
            for layer_idx, from_c, to_c in trace.shift_points:
                la = trace.layers[layer_idx]
                top_str = ", ".join(f"{c}({s:.2f})" for c, s in la.top_concepts[:3])
                print(f"    Layer {layer_idx:3d} [{la.layer_type:8s}] "
                      f"{from_c} → {to_c}  | top: {top_str}")

            # Show contamination summary
            print()
            print(f"  CONTAMINATED CONCEPTS (appeared ≥2 layers):")
            heavy = {c: layers for c, layers in trace.contamination_map.items()
                     if len(layers) >= 2 and c != trace.concept}
            for c, layers in sorted(heavy.items(), key=lambda x: -len(x[1]))[:20]:
                bar = "█" * min(len(layers), 30)
                print(f"    {c:25s} {bar} ({len(layers)} layers)")
        else:
            # Full layer-by-layer
            for la in trace.layers:
                top = ", ".join(f"{c}({s:.2f})" for c, s in la.top_concepts)
                print(f"  L{la.layer_index:3d} [{la.layer_type:8s}] "
                      f"norm={la.activation_norm:.2f}  {top}")

        print(f"{'='*65}\n")

    def print_contamination_map(self, cmap: ContaminationMap, top_n: int = 20):
        """Print the top viral spreaders from the contamination map."""
        print(f"\n{'='*65}")
        print(f"  VIRAL SPREADERS (concepts that contaminate the most others)")
        print(f"{'='*65}")
        for concept, radius in cmap.top_spreaders[:top_n]:
            bar = "█" * min(radius // 2, 40)
            print(f"  {concept:25s} {bar} ({radius} others activated)")
        print()

    # ── Dead Zone Analysis ────────────────────────────────────────────────────

    def analyze_dead_zones(
        self,
        trace: FlowTrace,
        dead_threshold: float = 0.12,
        active_threshold: float = 0.25,
    ) -> DeadZoneAnalysis:
        """
        Find stretches where concept signal goes underground and interpolate the
        trajectory to see what *should* have been there — then flag the spots
        where the concept direction actually shifted during the silence.

        The "connect the dots" approach:
            - 1–3 active … [4–5 dead] … 6–10 active
            - Slerp from the entry direction to the exit direction
            - Compare each dead layer's actual hidden state to the slerp path
            - Big deviation = that layer is doing real work (surgery target)
            - No deviation   = passive carrier (concept just rode through)

        Args:
            dead_threshold:   Max dominant_sim below which a layer is "dead" (default 0.12)
            active_threshold: Min trajectory_shift to call a zone "active" (default 0.25)

        Returns:
            DeadZoneAnalysis with zones, active_zones, minimum_cut_layers
        """
        layers = trace.layers
        n = len(layers)

        alive = [la.dominant_sim >= dead_threshold for la in layers]

        zones: List[DeadZone] = []
        i = 0
        while i < n:
            if not alive[i]:
                # Find the end of this dead stretch
                j = i
                while j < n and not alive[j]:
                    j += 1

                entry_idx = i - 1
                exit_idx  = j   # first live layer after the gap

                if entry_idx >= 0 and exit_idx < n:
                    entry_la = layers[entry_idx]
                    exit_la  = layers[exit_idx]

                    shift        = 0.0
                    interpolated : List[Tuple[int, str, float]] = []
                    causal_layer : Optional[int] = None

                    if (entry_la.hidden_dir is not None
                            and exit_la.hidden_dir is not None):

                        entry_vec = F.normalize(entry_la.hidden_dir.float(), dim=-1)
                        exit_vec  = F.normalize(exit_la.hidden_dir.float(), dim=-1)

                        # Align to concept matrix dim (some layers have different hidden size)
                        cdim = self.concept_matrix.shape[-1]
                        entry_vec = self._align_dim(entry_vec, cdim)
                        exit_vec  = self._align_dim(exit_vec,  cdim)

                        dot   = (entry_vec * exit_vec).sum().clamp(-1.0, 1.0).item()
                        shift = max(0.0, 1.0 - dot)

                        # Slerp through the gap and score each dead layer
                        dead_count    = j - i
                        max_deviation = -1.0

                        concept_mat = self.concept_matrix.cpu()

                        for k, layer_k in enumerate(range(i, j)):
                            t = (k + 1) / (dead_count + 1)
                            interp_vec = self._slerp(entry_vec, exit_vec, t)  # unit vec

                            # Predicted concept at this interpolated position
                            sims      = (concept_mat @ interp_vec).tolist()
                            top_idx   = int(max(range(len(sims)), key=lambda x: sims[x]))
                            pred_conc = self._concept_names[top_idx]
                            pred_sim  = float(sims[top_idx])
                            interpolated.append((layer_k, pred_conc, round(pred_sim, 4)))

                            # How far does the actual hidden state deviate from slerp?
                            actual_la = layers[layer_k]
                            if actual_la.hidden_dir is not None:
                                actual_vec  = F.normalize(
                                    self._align_dim(actual_la.hidden_dir.float(), cdim),
                                    dim=-1)
                                adherence   = (actual_vec * interp_vec).sum().item()
                                deviation   = 1.0 - adherence
                                if deviation > max_deviation:
                                    max_deviation = deviation
                                    causal_layer  = layer_k

                    zones.append(DeadZone(
                        start_layer  = i,
                        end_layer    = j - 1,
                        entry_layer  = entry_idx,
                        exit_layer   = exit_idx,
                        entry_concept= entry_la.dominant,
                        exit_concept = exit_la.dominant,
                        entry_sim    = round(entry_la.dominant_sim, 4),
                        exit_sim     = round(exit_la.dominant_sim, 4),
                        trajectory_shift = round(shift, 4),
                        is_active    = shift > active_threshold,
                        interpolated = interpolated,
                        causal_layer = causal_layer,
                    ))
                i = j
            else:
                i += 1

        active_zones = [z for z in zones if z.is_active]

        # Minimum cut: the causal layer from each active zone, sorted by shift score
        min_cut: List[Tuple[int, str, str, float]] = []
        for z in active_zones:
            if z.causal_layer is not None:
                la = layers[z.causal_layer]
                min_cut.append((z.causal_layer, la.layer_type,
                                 la.module_path, z.trajectory_shift))
        min_cut.sort(key=lambda x: x[3], reverse=True)

        dza = DeadZoneAnalysis(
            concept              = trace.concept,
            total_layers         = n,
            dead_threshold       = dead_threshold,
            zones                = zones,
            active_zones         = active_zones,
            minimum_cut_layers   = min_cut,
        )
        trace.dead_zones = dza
        return dza

    def _slerp(
        self, v0: torch.Tensor, v1: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Spherical linear interpolation between two unit vectors on CPU."""
        v0 = F.normalize(v0.float().cpu(), dim=-1)
        v1 = F.normalize(v1.float().cpu(), dim=-1)
        dot   = (v0 * v1).sum().clamp(-1.0, 1.0)
        theta = torch.acos(dot)
        if theta.abs().item() < 1e-4:
            return F.normalize(v0 * (1.0 - t) + v1 * t, dim=-1)
        sin_theta = torch.sin(theta)
        t_tensor  = torch.tensor(t, dtype=torch.float32)
        w0 = torch.sin((1.0 - t_tensor) * theta) / sin_theta
        w1 = torch.sin(t_tensor * theta) / sin_theta
        return F.normalize(w0 * v0 + w1 * v1, dim=-1)

    @staticmethod
    def _align_dim(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Truncate or zero-pad a vector to target_dim."""
        d = vec.shape[-1]
        if d == target_dim:
            return vec
        if d > target_dim:
            return vec[:target_dim]
        return F.pad(vec, (0, target_dim - d))

    def print_dead_zones(self, trace: FlowTrace, dead_threshold: float = 0.12):
        """
        Print dead-zone analysis — the "connect the dots" view.

        For each gap where signal disappeared: shows where it went in,
        what the slerp predicts should have been there, and which specific
        layer caused a direction change (the surgery target).
        """
        dza = trace.dead_zones
        if dza is None or dza.dead_threshold != dead_threshold:
            dza = self.analyze_dead_zones(trace, dead_threshold=dead_threshold)

        print(f"\n{'='*65}")
        print(f"  DEAD ZONE MAP: '{trace.concept}'")
        print(f"{'='*65}")
        print(f"  Threshold:      sim < {dza.dead_threshold}  → layer goes dark")
        print(f"  Total gaps:     {len(dza.zones)}")
        print(f"  Active (causal):{len(dza.active_zones)}  ← concept transformed inside")
        print(f"  Surgery targets:{len(dza.minimum_cut_layers)}")
        print()

        if not dza.zones:
            print("  No dead zones — signal was continuous across all layers.\n")
            print(f"{'='*65}\n")
            return

        for z in dza.zones:
            gap   = z.end_layer - z.start_layer + 1
            label = "*** ACTIVE" if z.is_active else "    passive"
            print(f"  Layers {z.entry_layer:3d}→[{z.start_layer:3d}…{z.end_layer:3d}]→{z.exit_layer:3d}"
                  f"  ({gap} dark)  "
                  f"shift={z.trajectory_shift:.3f}  {label}")
            print(f"    {z.entry_concept}({z.entry_sim:.2f}) ──── "
                  f"{z.exit_concept}({z.exit_sim:.2f})")

            if z.interpolated:
                path = " → ".join(
                    f"L{l}:{c}({s:.2f})" for l, c, s in z.interpolated[:6]
                )
                print(f"    predicted path: {path}")

            if z.is_active and z.causal_layer is not None:
                la = trace.layers[z.causal_layer]
                print(f"    ▶ causal layer:  {z.causal_layer} [{la.layer_type}]  {la.module_path}")
            print()

        if dza.minimum_cut_layers:
            print(f"  MINIMUM CUT — layers to transplant for this capability:")
            for layer, ltype, path, shift in dza.minimum_cut_layers:
                bar = "█" * min(int(shift * 20), 20)
                print(f"    L{layer:3d} [{ltype:9s}] {bar} shift={shift:.3f}  ({path})")
        print(f"{'='*65}\n")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save_trace(self, trace: FlowTrace, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)

    def save_contamination_map(self, cmap: ContaminationMap, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cmap.to_dict(), f, indent=2)

    @staticmethod
    def load_trace(path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _find_top_concepts(
        self, activation: torch.Tensor
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        Given an activation vector, find the Top-K matching concepts.

        Mode A — SAE (monosemantic, preferred when sae is loaded):
            Decode activation → sparse feature vector → pick active features
            → translate each active feature to its closest concept via decoder
            direction alignment. Returns deduplicated top-K by max feature value.

        Mode B — Raw cosine similarity (fallback, polysemantic):
            Direct cosine similarity against the normalized concept matrix.

        Returns: ([(concept, sim), ...], activation_norm)
        """
        norm = activation.norm().item()
        if norm < 1e-8:
            return [], 0.0

        if self.sae is not None and self._feature_to_concept is not None:
            return self._find_top_concepts_sae(activation, norm)
        return self._find_top_concepts_cosine(activation, norm)

    def _find_top_concepts_sae(
        self, activation: torch.Tensor, norm: float
    ) -> Tuple[List[Tuple[str, float]], float]:
        """
        SAE-mode concept identification — monosemantic, clean.

        Maps top active SAE features → concept names via the decoder column
        directions pre-computed in align_features_to_concepts().
        """
        sae = self.sae
        act = activation.to(sae.b_pre.device)

        # Dimension guard: SAE is trained on a specific layer's hidden dim
        if act.shape[-1] != sae.hidden_dim:
            # Fallback to cosine if shapes are incompatible
            return self._find_top_concepts_cosine(activation, norm)

        indices, values = sae.get_active_features(act)
        if len(indices) == 0:
            return [], norm

        # Walk active features, accumulate concept → max_feature_value
        concept_scores: Dict[str, float] = {}
        for feat_i, val in zip(indices.tolist(), values.tolist()):
            for concept_name, concept_sim in self._feature_to_concept.get(feat_i, []):
                # Only count concepts whose decoder direction actually aligns
                if concept_sim >= self.sim_threshold:
                    if concept_name not in concept_scores:
                        concept_scores[concept_name] = 0.0
                    # Use max across all features that point to this concept
                    concept_scores[concept_name] = max(concept_scores[concept_name],
                                                       val * concept_sim)

        if not concept_scores:
            # No feature passed the threshold — fall back to cosine
            return self._find_top_concepts_cosine(activation, norm)

        top_k = min(self.top_k, len(concept_scores))
        top_concepts = sorted(concept_scores.items(), key=lambda x: -x[1])[:top_k]
        # Normalize scores to [0, 1] range using the max value
        max_score = top_concepts[0][1] if top_concepts else 1.0
        return [(c, round(s / max(max_score, 1e-8), 4)) for c, s in top_concepts], norm

    def _find_top_concepts_cosine(
        self, activation: torch.Tensor, norm: float
    ) -> Tuple[List[Tuple[str, float]], float]:
        """Raw cosine-similarity concept matching (original approach, polysemantic)."""
        act = activation
        if act.shape[-1] != self.concept_matrix.shape[-1]:
            target_dim = self.concept_matrix.shape[-1]
            if act.shape[-1] > target_dim:
                act = act[:target_dim]
            else:
                act = F.pad(act, (0, target_dim - act.shape[-1]))

        act_norm = F.normalize(act.unsqueeze(0), dim=-1)  # [1, D]
        sims = torch.matmul(act_norm, self.concept_matrix.T).squeeze(0)  # [N]

        top_k = min(self.top_k, len(self._concept_names))
        top_sims, top_indices = torch.topk(sims, top_k)

        results = []
        for idx, sim in zip(top_indices.tolist(), top_sims.tolist()):
            results.append((self._concept_names[idx], float(sim)))
        return results, norm

    # ── Attribution Patching ──────────────────────────────────────────────────

    def attribution_patch(
        self,
        concept: str,
        target_layer: Optional[int] = None,
        n_samples: int = 8,
    ) -> Dict[int, float]:
        """
        Estimate each layer's causal contribution to the concept representation
        using gradient-based attribution patching.

        Method (approximation of full activation patching at 1000x less compute):
            attr_layer = mean(|grad_output · activation|) at each layer

        This tells you which layers *causally* determine the concept's meaning,
        rather than just which layers have high cosine similarity.

        Args:
            concept:      Concept name to analyze
            target_layer: If set, only patches up to this layer
            n_samples:    Number of perturbation runs to average (more = more stable)

        Returns:
            Dict[layer_index → attribution_score] sorted by score
        """
        model = self.loader.model
        tokenizer = self.loader.tokenizer

        prompt = f"The concept of {concept} means"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        concept_vec = self._get_concept_vector(concept)
        if concept_vec is None:
            raise ValueError(f"Concept '{concept}' not found in concept vectors.")
        concept_vec = F.normalize(concept_vec.float(), dim=-1).to(self.device)

        activations_storage: Dict[str, torch.Tensor] = {}
        grad_storage: Dict[str, torch.Tensor] = {}
        handles = []

        def _fwd_hook(name):
            def _h(module, inp, out):
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                    pooled = tensor.float().mean(dim=-2)
                    if pooled.ndim == 2:
                        pooled = pooled[0]
                    activations_storage[name] = pooled.to(self.device)
                    pooled.retain_grad()
            return _h

        layer_names = []
        layer_counter = 0
        for layer_info in self.arch_map.layers:
            if target_layer is not None and layer_counter > target_layer:
                break
            # Hook block, attention, and MLP — same as trace()
            for ltype, path in [
                ("residual", layer_info.name),
                ("attention", layer_info.attn_path),
                ("mlp", layer_info.mlp_path),
            ]:
                if path is None:
                    continue
                if target_layer is not None and layer_counter > target_layer:
                    break
                parts = path.split(".")
                mod = model
                for p in parts:
                    mod = mod[int(p)] if p.isdigit() else getattr(mod, p, None)
                    if mod is None:
                        break
                if mod is None:
                    continue
                h = mod.register_forward_hook(_fwd_hook(path))
                handles.append(h)
                layer_names.append(path)
                layer_counter += 1

        attributions: Dict[int, float] = {}

        try:
            # Enable grad for attribution
            with torch.enable_grad():
                outputs = model(**inputs)
                # Compute similarity between last hidden state and concept vector
                last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") \
                    else outputs[0]  # [1, T, D]
                if last_hidden is not None:
                    mean_hidden = last_hidden.float().mean(dim=1)  # [1, D]
                    if mean_hidden.shape[-1] == concept_vec.shape[-1]:
                        score = torch.dot(
                            F.normalize(mean_hidden.squeeze(0), dim=-1),
                            concept_vec
                        )
                        score.backward()

                        for i, lname in enumerate(layer_names):
                            act = activations_storage.get(lname)
                            if act is not None and act.grad is not None:
                                attr = float((act.grad.abs() * act.abs()).mean().item())
                                attributions[i] = attr

        except Exception as e:
            print(f"  [Tracer] attribution_patch warning: {e}")
        finally:
            for h in handles:
                h.remove()

        return dict(sorted(attributions.items(), key=lambda x: -x[1]))

    # ── Feature Steering ─────────────────────────────────────────────────────

    def steer(
        self,
        concept: str,
        alpha: float = 10.0,
        target_layer: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Steering intervention: clamp the concept vector direction in residual stream
        at the target layer during a forward pass, then decode the output.

        Inspired by Anthropic "Golden Gate Bridge" experiment — clamping a single
        SAE feature to 10x its normal value causes the model to respond AS IF it is
        that concept. Useful for verifying that a concept is where you think it is.

        Args:
            concept:      Concept to amplify
            alpha:        Steering magnitude (default 10.0; use 0.0 to ablate)
            target_layer: Which layer to intervene at. If None, uses the SAE's
                          trained layer (or the mapper's fact_layer_start).
            prompt:       Prompt to run. Defaults to a generic probe.

        Returns:
            Decoded text after steering (first generated token + logit shift info)
        """
        model = self.loader.model
        tokenizer = self.loader.tokenizer

        concept_vec = self._get_concept_vector(concept)
        if concept_vec is None:
            raise ValueError(f"Concept '{concept}' not found in concept vectors.")
        concept_dir = F.normalize(concept_vec.float(), dim=-1).to(self.device)

        if prompt is None:
            prompt = f"Tell me about {concept}"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # Determine steering layer
        if target_layer is None:
            if self.sae is not None and self.sae.target_layer >= 0:
                target_layer = self.sae.target_layer
            else:
                target_layer = self.arch_map.fact_layer_start

        # Find the module to hook
        target_path = None
        if self.arch_map.layers and target_layer < len(self.arch_map.layers):
            target_path = self.arch_map.layers[target_layer].name

        if target_path is None:
            raise ValueError(f"Cannot find layer {target_layer} in arch_map.")

        target_module = None
        for name, module in model.named_modules():
            if name == target_path:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Module '{target_path}' not found.")

        pre_logits = []
        post_logits = []

        def _steer_hook(module, inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(tensor, torch.Tensor):
                return out
            # Add alpha * concept_dir to every token position at this layer
            steering = alpha * concept_dir  # [D]
            steered = tensor + steering.unsqueeze(0).unsqueeze(0)  # broadcast
            if isinstance(out, (tuple, list)):
                return (steered,) + tuple(out[1:])
            return steered

        handle = target_module.register_forward_hook(_steer_hook)
        result_text = ""
        try:
            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits if hasattr(out, "logits") else out[0]
                # Greedy decode next token
                next_token_id = logits[0, -1, :].argmax().item()
                result_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
        finally:
            handle.remove()

        return result_text

    # ── Logit Lens ────────────────────────────────────────────────────────

    def logit_lens(
        self,
        sentence: str,
        top_k: int = 10,
        token_position: int = -1,
    ) -> LogitLensResult:
        """
        Run the logit lens: at each layer, project the hidden state through
        the model's unembedding matrix (lm_head) to see what token the model
        would predict if it stopped processing at that layer.

        This is the "disassembly view" — decoded meaning at every execution step.

        Args:
            sentence:       Input text to analyze
            top_k:          How many top predictions per layer
            token_position: Which token position to inspect (-1 = last token)

        Returns:
            LogitLensResult with per-layer top-k token predictions
        """
        model = self.loader.model
        tokenizer = self.loader.tokenizer

        inputs = tokenizer(sentence, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        input_tokens = [tokenizer.decode([tid]) for tid in input_ids]

        # Find the unembedding matrix (lm_head or tied embeddings)
        unembed = self._get_unembed_matrix(model)
        if unembed is None:
            raise RuntimeError(
                "Cannot find unembedding matrix (lm_head). "
                "Model may not be a causal LM."
            )

        # Collect hidden states at each layer via hooks
        hidden_states: List[Tuple[int, str, str, torch.Tensor]] = []
        handles = []

        def _make_hook(layer_idx, ltype, path):
            def _hook(module, inp, out):
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
                    # Keep full sequence: [batch, seq, hidden]
                    hidden_states.append(
                        (layer_idx, ltype, path, tensor.detach().float())
                    )
            return _hook

        layer_counter = 0
        for layer_info in self.arch_map.layers:
            # Register attn and mlp BEFORE the block — they fire first
            # (block output hook fires after its children complete)
            if layer_info.attn_path:
                attn_mod = self._resolve_module(layer_info.attn_path)
                if attn_mod is not None:
                    h = attn_mod.register_forward_hook(
                        _make_hook(layer_counter, "attention", layer_info.attn_path))
                    handles.append(h)
                    layer_counter += 1

            if layer_info.mlp_path:
                mlp_mod = self._resolve_module(layer_info.mlp_path)
                if mlp_mod is not None:
                    h = mlp_mod.register_forward_hook(
                        _make_hook(layer_counter, "mlp", layer_info.mlp_path))
                    handles.append(h)
                    layer_counter += 1

            block_mod = self._resolve_module(layer_info.name)
            if block_mod is not None:
                h = block_mod.register_forward_hook(
                    _make_hook(layer_counter, "residual", layer_info.name))
                handles.append(h)
                layer_counter += 1

        # Forward pass
        try:
            with torch.no_grad():
                output = model(**inputs)
        finally:
            for h in handles:
                h.remove()

        # Get final prediction for reference
        final_logits = output.logits if hasattr(output, "logits") else output[0]
        final_token_id = final_logits[0, -1, :].argmax().item()
        final_prediction = tokenizer.decode([final_token_id])

        # Resolve token position
        seq_len = input_ids.shape[0]
        pos = token_position if token_position >= 0 else seq_len + token_position

        # Build logit lens result
        result = LogitLensResult(
            sentence=sentence,
            input_tokens=input_tokens,
            final_prediction=final_prediction,
        )

        for layer_idx, ltype, path, hidden in hidden_states:
            # Extract hidden state at the target token position: [hidden_dim]
            h = hidden[0, pos, :]  # batch=0, pos=target, all dims

            # Project through unembedding: [hidden_dim] @ [hidden_dim, vocab] → [vocab]
            logits = h @ unembed.T.float()

            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Top-k
            topk_probs, topk_ids = torch.topk(probs, min(top_k, probs.shape[0]))
            top_tokens = [
                (tokenizer.decode([tid.item()]), prob.item())
                for tid, prob in zip(topk_ids, topk_probs)
            ]

            result.layers.append(LogitLensLayer(
                layer_index=layer_idx,
                layer_type=ltype,
                module_path=path,
                top_tokens=top_tokens,
            ))

        return result

    # ── Per-Token Sentence Tracing ────────────────────────────────────────

    def trace_sentence(
        self,
        sentence: str,
        top_k_tokens: int = 5,
        attention_threshold: float = 0.05,
        num_paths: int = 5,
    ) -> TraceGraph:
        """
        Trace every token's hidden state through every layer, capturing
        attention edges and residual connections. This is the "step through"
        mode — you see how each token's representation evolves.

        Args:
            sentence:            Input text to trace
            top_k_tokens:        Top-k logit lens predictions per node
            attention_threshold: Minimum attention weight to create an edge
            num_paths:           How many top paths to extract via beam search

        Returns:
            TraceGraph with per-token nodes, attention/residual edges, and paths
        """
        model = self.loader.model
        tokenizer = self.loader.tokenizer

        inputs = tokenizer(sentence, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        input_tokens = [tokenizer.decode([tid]) for tid in input_ids]
        seq_len = len(input_tokens)

        unembed = self._get_unembed_matrix(model)

        # ── 1. Collect per-token hidden states at each block ──────────────
        # We hook residual stream (block outputs) for per-token states
        block_hidden: Dict[int, torch.Tensor] = {}  # block_idx → [seq, hidden]
        handles = []

        def _make_block_hook(block_idx):
            def _hook(module, inp, out):
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
                    block_hidden[block_idx] = tensor[0].detach().float()  # [seq, hidden]
            return _hook

        for layer_info in self.arch_map.layers:
            block_mod = self._resolve_module(layer_info.name)
            if block_mod is not None:
                h = block_mod.register_forward_hook(_make_block_hook(layer_info.index))
                handles.append(h)

        # ── 2. Forward pass with attention outputs ────────────────────────
        # Need eager attention implementation to get attention weights
        # (SDPA doesn't support output_attentions)
        orig_attn_impl = getattr(model.config, "_attn_implementation", None)
        try:
            model.config._attn_implementation = "eager"
            with torch.no_grad():
                output = model(**inputs, output_attentions=True)
        finally:
            if orig_attn_impl is not None:
                model.config._attn_implementation = orig_attn_impl
            for h in handles:
                h.remove()

        # Extract attention weights: list of [batch, heads, seq, seq] per layer
        attentions = getattr(output, "attentions", None)
        if attentions is not None and len(attentions) == 0:
            attentions = None

        num_blocks = len(block_hidden)
        graph = TraceGraph(
            sentence=sentence,
            input_tokens=input_tokens,
            num_blocks=num_blocks,
        )

        # ── 3. Build nodes (one per token per block) ─────────────────────
        for block_idx in sorted(block_hidden.keys()):
            hidden = block_hidden[block_idx]  # [seq, hidden]
            for tok_idx in range(seq_len):
                h = hidden[tok_idx]  # [hidden]
                norm = h.norm().item()

                # Logit lens at this node
                top_tokens_list = []
                if unembed is not None:
                    logits = h @ unembed.T.float()
                    probs = F.softmax(logits, dim=-1)
                    topk_p, topk_i = torch.topk(probs, min(top_k_tokens, probs.shape[0]))
                    top_tokens_list = [
                        (tokenizer.decode([i.item()]), p.item())
                        for i, p in zip(topk_i, topk_p)
                    ]

                node_id = f"b{block_idx}_res_t{tok_idx}"
                graph.nodes.append(TraceNode(
                    id=node_id,
                    block=block_idx,
                    layer_type="residual",
                    token_idx=tok_idx,
                    token_str=input_tokens[tok_idx],
                    norm=norm,
                    top_tokens=top_tokens_list,
                ))

        # ── 4. Build residual edges (consecutive blocks, same token) ─────
        sorted_blocks = sorted(block_hidden.keys())
        for i in range(len(sorted_blocks) - 1):
            b_from = sorted_blocks[i]
            b_to = sorted_blocks[i + 1]
            h_from = block_hidden[b_from]  # [seq, hidden]
            h_to = block_hidden[b_to]      # [seq, hidden]

            for tok_idx in range(seq_len):
                # Residual edge weight = cosine similarity of hidden states
                cos_sim = F.cosine_similarity(
                    h_from[tok_idx].unsqueeze(0),
                    h_to[tok_idx].unsqueeze(0),
                ).item()
                graph.edges.append(TraceEdge(
                    source=f"b{b_from}_res_t{tok_idx}",
                    target=f"b{b_to}_res_t{tok_idx}",
                    weight=cos_sim,
                    edge_type="residual",
                ))

        # ── 5. Build attention edges (cross-token within each block) ─────
        if attentions is not None:
            for block_idx, attn_weights in enumerate(attentions):
                # attn_weights: [batch, heads, seq, seq]
                aw = attn_weights[0].detach().float()  # [heads, seq, seq]
                num_heads = aw.shape[0]

                # Average across heads for edge weights, but keep per-head
                # for the strongest connections
                avg_attn = aw.mean(dim=0)  # [seq, seq]

                for target_tok in range(seq_len):
                    for source_tok in range(seq_len):
                        w = avg_attn[target_tok, source_tok].item()
                        if w >= attention_threshold:
                            # Find which head contributes most
                            head_weights = aw[:, target_tok, source_tok]
                            best_head = head_weights.argmax().item()

                            graph.edges.append(TraceEdge(
                                source=f"b{block_idx}_res_t{source_tok}",
                                target=f"b{block_idx}_res_t{target_tok}",
                                weight=w,
                                edge_type="attention",
                                head=best_head,
                            ))

        # ── 6. Extract top-k paths via beam search ───────────────────────
        graph.paths = self._extract_paths(graph, num_paths, seq_len)

        return graph

    def _extract_paths(
        self,
        graph: TraceGraph,
        num_paths: int,
        seq_len: int,
    ) -> List[TracePath]:
        """
        Beam search for the strongest paths from last token at first block
        to last token at final block (following residual + attention edges).
        """
        # Build adjacency: node_id → list of (target_id, weight)
        adj: Dict[str, List[Tuple[str, float]]] = {}
        for e in graph.edges:
            if e.source not in adj:
                adj[e.source] = []
            adj[e.source].append((e.target, e.weight))

        if not adj:
            return []

        # Start from the last token at block 0
        last_tok = seq_len - 1
        start_id = f"b0_res_t{last_tok}"
        if start_id not in adj:
            # Try first available node
            if graph.nodes:
                start_id = graph.nodes[0].id
            else:
                return []

        # Beam search
        beam_width = num_paths * 3  # wider beam for diversity
        # Each beam entry: (cumulative_score, [node_ids])
        beams = [(0.0, [start_id])]

        max_steps = graph.num_blocks * 3  # prevent infinite loops
        for _ in range(max_steps):
            candidates = []
            for score, path in beams:
                last_node = path[-1]
                if last_node not in adj:
                    candidates.append((score, path))  # dead end — keep as-is
                    continue
                for next_node, weight in adj[last_node]:
                    if next_node not in path:  # no cycles
                        candidates.append((score + weight, path + [next_node]))
            if not candidates:
                break
            # Keep top beam_width
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

            # Check if all beams reached the final block
            all_done = all(
                any(f"b{graph.num_blocks - 1}" in n for n in path)
                for _, path in beams
            )
            if all_done:
                break

        # Return top num_paths
        paths = []
        for i, (score, node_list) in enumerate(beams[:num_paths]):
            paths.append(TracePath(
                id=f"path_{i}",
                nodes=node_list,
                score=score,
                method=f"beam_k={beam_width}",
            ))
        return paths

    def _get_unembed_matrix(self, model: nn.Module) -> Optional[torch.Tensor]:
        """
        Find the unembedding (output projection) weight matrix.
        Handles: lm_head.weight, embed_out.weight, tied embeddings.
        """
        # Direct lm_head (GPT-2, Llama, Qwen, Mistral, Phi, etc.)
        if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
            return model.lm_head.weight.data

        # GPT-NeoX style
        if hasattr(model, "embed_out") and hasattr(model.embed_out, "weight"):
            return model.embed_out.weight.data

        # Tied embeddings fallback — check transformer.wte (GPT-2 tied path)
        for name in ["transformer.wte", "model.embed_tokens"]:
            mod = self._resolve_module(name)
            if mod is not None and hasattr(mod, "weight"):
                return mod.weight.data

        return None

    def _resolve_module(self, path: str) -> Optional[nn.Module]:
        """Walk dotted path to get the actual nn.Module."""
        parts = path.split(".")
        mod = self.loader.model
        for p in parts:
            if p.isdigit():
                mod = mod[int(p)]
            else:
                mod = getattr(mod, p, None)
            if mod is None:
                return None
        return mod

    def _get_concept_vector(self, concept: str) -> Optional[torch.Tensor]:
        """Retrieve the raw concept vector by name."""
        if concept in self._concept_vectors:
            return self._concept_vectors[concept]
        # Case-insensitive fallback
        lower = concept.lower()
        for k, v in self._concept_vectors.items():
            if k.lower() == lower:
                return v
        return None

    def _classify_module(self, name: str, module: nn.Module) -> Optional[str]:
        """
        Classify a named module into a layer type for tracing.
        Returns None for modules we should skip (too fine-grained or non-transformer).
        """
        if not name:
            return None

        depth = name.count(".")
        name_lower = name.lower()

        # Skip raw parameter wrapping / norm layers (too many, too noisy)
        skip_patterns = [
            "layernorm", "layer_norm", "rmsnorm", "norm",
            "dropout", "embed_tokens",  # embedding handled separately
            "lm_head", "wte", "wpe",    # skip output/positional embeds
        ]
        for pat in skip_patterns:
            if name_lower.endswith(pat) or name_lower.endswith(f".{pat}"):
                return None

        # Only hook "meaningful" granularity layers:
        # - Full transformer blocks (depth ≤ 3, contains digits = layer index)
        # - Attention modules (named "self_attn", "attn", "attention")
        # - MLP modules (named "mlp", "feed_forward", "ffn")
        #
        # IMPORTANT: Only match the LAST path component to avoid hooking sub-modules.
        # e.g. "transformer.h.0.attn" ✓  but "transformer.h.0.attn.c_attn" ✗
        parts = name.split(".")
        last_part = parts[-1].lower()

        # Check if this is an attention-level module (top-level only)
        if last_part in ("self_attn", "attention", "attn", "multi_head"):
            return "attention"

        # Check if this is an MLP/FFN module (top-level only)
        if last_part in ("mlp", "feed_forward", "ffn", "feedforward"):
            return "mlp"

        # Top-level layer blocks (e.g. "model.layers.4", "transformer.h.4")
        # These are the residual stream checkpoints
        if len(parts) >= 2 and parts[-1].isdigit() and depth <= 3:
            return "residual"

        return None
