"""
cartography.py — LoRACartographer (AutoPsy Stage 10: CARTOGRAPHY)
=================================================================

The concept fingerprinting stage.

During LoRA fine-tuning on concept-specific sentences, adapter matrices A and
B learn the *minimal* weight change that encodes that concept at each layer.
The dominant singular vector of A @ B tells us exactly which direction in
that layer's representation space the model is "writing to".

This is fundamentally different from activation hooks (which measure what
FIRES) — cartography measures what the model LEARNS, giving a gradient-level
map of where each concept lives in weight space.

Key insight:
    (A @ B).svd().U[:, 0] at layer L is the "concept direction" in that
    layer's input space. If this direction is the same (up to rotation) for
    "fire" across GPT-2, LLaMA-2, and Qwen2, then that direction IS the
    Platonic "fire" coordinate in transformer weight space.

Closed-loop verification:
    1. Cartographer trains map  →  "fire lives at L4, L9 with these directions"
    2. DeadZoneAnalyzer verifies  →  confirms causal layers match
    3. Alignment matrix computed  →  rotation R maps model_A geometry → model_B
    4. Transplant writes rotated directions into target model
    5. Post-transplant dead zone confirm graft stability

Usage:
    from antroslammer.autopsy.cartography import LoRACartographer

    cartog = LoRACartographer(loader, arch_map)

    fire_map = cartog.train_concept(
        concept="fire",
        sentences=[
            "Fire produces heat and light.",
            "Fire requires oxygen to burn.",
            "Fire can spread rapidly through dry material.",
            "Campfires provide warmth in cold weather.",
        ],
        epochs=3,
        rank=4,
    )
    cartog.print_map(fire_map)
    cartog.save_map(fire_map, "autopsy_output/cartography/fire.json")

    # Platonic Representation test — compare same concept across two models:
    align = LoRACartographer.compare_maps(fire_map_gpt2, fire_map_qwen)
    print(f"Global alignment: {align.global_alignment:.3f}")
    # → 0.7+ = strong Platonic convergence
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ConceptFrame:
    """Snapshot of a concept's direction in one layer at one training step."""

    step: int
    layer_name: str
    # L2-normalized dominant direction in that layer's input space. CPU tensor.
    direction: torch.Tensor
    # Dominant singular value of A@B — higher = layer is more involved.
    magnitude: float

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "layer_name": self.layer_name,
            "direction": self.direction.tolist(),
            "magnitude": self.magnitude,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptFrame":
        return cls(
            step=d["step"],
            layer_name=d["layer_name"],
            direction=torch.tensor(d["direction"]),
            magnitude=float(d["magnitude"]),
        )


@dataclass
class ConceptMap:
    """Full cartographic map of a concept across all layers.

    Attributes:
        concept:          The concept name (e.g. "fire")
        model_name:       The model this map was trained on
        frames_by_layer:  Dict[layer_name → List[ConceptFrame]] — trajectory
        final_directions: Dict[layer_name → unit_vector] — rank-1 dominant direction
        layer_ownership:  [(layer_name, score)] sorted by score descending
        final_svd:        Dict[layer_name → {U, S, Vh, rank}] — full rank-k SVD
                          U [in×k], S [k], Vh [k×out] — the concept subspace, not just
                          its dominant direction.  Used for rank-k transplant.
    """

    concept: str
    model_name: str
    frames_by_layer: Dict[str, List[ConceptFrame]] = field(default_factory=dict)
    final_directions: Dict[str, torch.Tensor] = field(default_factory=dict)
    # Sorted descending by ownership score
    layer_ownership: List[Tuple[str, float]] = field(default_factory=list)
    # Rank-k SVD data per layer (populated by train_concept / fast_map)
    final_svd: Dict[str, dict] = field(default_factory=dict)

    def dominant_layers(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return the top-k layers that most strongly encode this concept."""
        return self.layer_ownership[:top_k]

    def to_dict(self) -> dict:
        svd_serial = {}
        for layer, svd in self.final_svd.items():
            svd_serial[layer] = {
                "U": svd["U"].tolist(),
                "S": svd["S"].tolist(),
                "Vh": svd["Vh"].tolist(),
                "rank": int(svd["rank"]),
            }
        return {
            "concept": self.concept,
            "model_name": self.model_name,
            "frames_by_layer": {
                k: [f.to_dict() for f in frames]
                for k, frames in self.frames_by_layer.items()
            },
            "final_directions": {
                k: v.tolist() for k, v in self.final_directions.items()
            },
            "layer_ownership": [[k, v] for k, v in self.layer_ownership],
            "final_svd": svd_serial,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptMap":
        obj = cls(concept=d["concept"], model_name=d["model_name"])
        obj.frames_by_layer = {
            k: [ConceptFrame.from_dict(f) for f in frames]
            for k, frames in d.get("frames_by_layer", {}).items()
        }
        obj.final_directions = {
            k: torch.tensor(v) for k, v in d.get("final_directions", {}).items()
        }
        obj.layer_ownership = [
            (row[0], float(row[1])) for row in d.get("layer_ownership", [])
        ]
        obj.final_svd = {
            layer: {
                "U": torch.tensor(v["U"]),
                "S": torch.tensor(v["S"]),
                "Vh": torch.tensor(v["Vh"]),
                "rank": int(v["rank"]),
            }
            for layer, v in d.get("final_svd", {}).items()
        }
        return obj


@dataclass
class LayerwiseAlignment:
    """Per-depth-bucket orthogonal rotation matrices mapping model A → model B.

    Instead of one global rotation (which fails across architectures where
    different layers own concepts at different depths), this stores a separate
    R matrix per normalized depth bucket.

    Example: GPT-2 stores 'arithmetic' at layer 6/12 = depth 0.5.
             LLaMA stores 'arithmetic' at layer 20/32 = depth 0.625.
    The per-depth R at each bucket accounts for this positional drift.
    """

    model_a: str
    model_b: str
    shared_concepts: List[str]
    n_bins: int
    # Depth bucket midpoint (str) → R matrix [dim × dim]
    R_per_depth: Dict[str, torch.Tensor]
    # Per-depth Frobenius residuals
    residuals_per_depth: Dict[str, float]

    @property
    def global_residual(self) -> float:
        if not self.residuals_per_depth:
            return 0.0
        return sum(self.residuals_per_depth.values()) / len(self.residuals_per_depth)

    def rotate(self, vec: torch.Tensor, normalized_depth: float) -> torch.Tensor:
        """Rotate a direction using the R matrix nearest to normalized_depth."""
        if not self.R_per_depth:
            return F.normalize(vec.float().cpu(), dim=0)
        nearest = min(self.R_per_depth.keys(), key=lambda d: abs(float(d) - normalized_depth))
        R = self.R_per_depth[nearest]
        v = vec.float().cpu()
        dim = R.shape[0]
        if v.shape[0] > dim:
            v = v[:dim]
        elif v.shape[0] < dim:
            v = torch.cat([v, torch.zeros(dim - v.shape[0])])
        return F.normalize(v @ R, dim=0)

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "shared_concepts": self.shared_concepts,
            "n_bins": self.n_bins,
            "R_per_depth": {k: v.tolist() for k, v in self.R_per_depth.items()},
            "residuals_per_depth": self.residuals_per_depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayerwiseAlignment":
        return cls(
            model_a=d["model_a"],
            model_b=d["model_b"],
            shared_concepts=d["shared_concepts"],
            n_bins=int(d["n_bins"]),
            R_per_depth={k: torch.tensor(v) for k, v in d["R_per_depth"].items()},
            residuals_per_depth=d["residuals_per_depth"],
        )


@dataclass
class InterferenceReport:
    """Pre-transplant safety scan — checks if new concept overwrites existing ones.

    After rotating the donor concept into target space, compute cosine similarity
    against every existing concept in the target model's map.  High similarity
    means you're about to overwrite that concept's weight direction.

    Ratings:
        safe    — max cosine < 0.4   (new direction, clean write)
        caution — max cosine 0.4–0.7 (partial overlap, proceed carefully)
        abort   — max cosine > 0.7   (would corrupt an existing concept)
    """

    concept: str
    # (layer_name, conflicting_concept, cosine_similarity)
    conflicts: List[Tuple[str, str, float]]
    safe_layers: List[str]
    risky_layers: List[str]
    max_cosine: float
    rating: str  # "safe" | "caution" | "abort"

    def summary(self) -> str:
        return (
            f"[Interference:{self.rating.upper()}] '{self.concept}'  "
            f"max_cos={self.max_cosine:.3f}  "
            f"safe={len(self.safe_layers)}  risky={len(self.risky_layers)}"
        )


@dataclass
class TransplantReport:
    """Full result of a transplant_concept() call, including safety data."""

    concept: str
    layers_edited: int
    total_delta_norm: float
    edit_norms: Dict[str, float]
    interference: Optional["InterferenceReport"]
    # Populated by probe_after_transplant() if called
    post_probe_alignment: Optional[float] = None

    def summary(self) -> str:
        rating = self.interference.rating if self.interference else "unchecked"
        post = (
            f"  post-probe alignment: {self.post_probe_alignment:.3f}"
            if self.post_probe_alignment is not None else ""
        )
        return (
            f"[Transplant] '{self.concept}'  "
            f"layers={self.layers_edited}  Δ={self.total_delta_norm:.5f}  "
            f"safety={rating}{post}"
        )


@dataclass
class ConceptAlignmentMatrix:
    """Orthogonal rotation matrix mapping model A's concept geometry → model B's.

    Computed via the orthogonal Procrustes problem over a shared concept set.
    This is the "translation dictionary" between two models' internal coordinate
    systems.  Multiply any concept direction from model A by R to get the
    equivalent direction in model B's weight space.

    Math: given A_dirs [n_concepts × dim_a] and B_dirs [n_concepts × dim_b],
          R = argmin  ||A_dirs @ R - B_dirs||_F
          Solution:   U, S, Vh = svd(A_dirs.T @ B_dirs);  R = U @ Vh
    """

    model_a: str
    model_b: str
    # Concepts used to compute the alignment
    shared_concepts: List[str]
    # Orthogonal rotation matrix [min_dim × min_dim] (CPU float32)
    R: torch.Tensor
    # Frobenius residual after alignment — lower = better
    residual: float

    def rotate(self, vec: torch.Tensor) -> torch.Tensor:
        """Rotate a concept direction from model A's space into model B's."""
        v = vec.float().cpu()
        dim = self.R.shape[0]
        if v.shape[0] > dim:
            v = v[:dim]
        elif v.shape[0] < dim:
            pad = torch.zeros(dim - v.shape[0])
            v = torch.cat([v, pad])
        return F.normalize(v @ self.R, dim=0)

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "shared_concepts": self.shared_concepts,
            "R": self.R.tolist(),
            "residual": self.residual,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptAlignmentMatrix":
        return cls(
            model_a=d["model_a"],
            model_b=d["model_b"],
            shared_concepts=d["shared_concepts"],
            R=torch.tensor(d["R"]),
            residual=float(d["residual"]),
        )


@dataclass
class CartographyAlignment:
    """Comparison of how two models store the same concept.

    A global_alignment score approaching 1.0 means the two models have
    converged to the same geometric encoding — direct evidence for the
    Platonic Representation Hypothesis.
    """

    concept: str
    model_a: str
    model_b: str
    # Keyed by normalized depth string e.g. "depth_0.33"
    cosine_per_layer: Dict[str, float]
    # Mean cosine across all matched depth buckets
    global_alignment: float

    def to_dict(self) -> dict:
        return {
            "concept": self.concept,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "cosine_per_layer": self.cosine_per_layer,
            "global_alignment": self.global_alignment,
        }


# ─── LoRA layer wrapper ────────────────────────────────────────────────────────

def _is_conv1d(module: nn.Module) -> bool:
    """Detect GPT-2's Conv1D whose weight is [in, out] not [out, in]."""
    return type(module).__name__ == "Conv1D"


class _LoRALinear(nn.Module):
    """Frozen linear layer + trainable rank-r LoRA adapter.

    Works transparently with:
      - nn.Linear (weight: [out_features, in_features])
      - GPT-2 Conv1D (weight: [in_features, out_features])

    LoRA forward: output = base(x) + x @ lora_A @ lora_B
    Delta matrix: lora_A @ lora_B  shape [in_features, out_features]
    """

    def __init__(self, base: nn.Module, rank: int = 4):
        super().__init__()
        self.base = base
        self._is_conv1d = _is_conv1d(base)

        if self._is_conv1d:
            in_features, out_features = base.weight.shape    # [in, out]
        else:
            out_features, in_features = base.weight.shape    # [out, in]

        self.in_features = in_features
        self.out_features = out_features

        # Match device of the base layer's weight
        dev = base.weight.device
        # LoRA params are ALWAYS float32 — they're tiny (rank=4) and fp16
        # causes NaN in optimizer states and matmul for diagnostic use.
        dt = torch.float32

        # Both initialized small-random so gradients flow to both A and B
        # from step 1.  Standard LoRA uses one=zeros which blocks gradient
        # to the zero side. For diagnostic "dye" we need both trainable now.
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank, device=dev, dtype=dt) * 1e-3
        )
        self.lora_B = nn.Parameter(
            torch.randn(rank, out_features, device=dev, dtype=dt) * 1e-3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA path in float32 (params are float32), cast result back
        lora_out = x.float() @ self.lora_A @ self.lora_B
        return self.base(x) + lora_out.to(x.dtype)

    def get_delta(self) -> torch.Tensor:
        """Full LoRA delta [in_features, out_features], as float32 on CPU."""
        with torch.no_grad():
            # Cast to float32 BEFORE matmul to avoid fp16 overflow
            return (self.lora_A.float() @ self.lora_B.float()).detach().cpu()

    def get_direction(self) -> Tuple[torch.Tensor, float]:
        """Dominant left singular vector of the delta + its singular value."""
        delta = self.get_delta()
        if delta.norm() < 1e-10:
            return torch.zeros(self.in_features), 0.0
        try:
            U, S, _Vh = torch.linalg.svd(delta, full_matrices=False)
        except Exception:
            return torch.zeros(self.in_features), 0.0
        return F.normalize(U[:, 0], dim=0), float(S[0].item())

    def get_directions_k(self, k: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-k SVD of the LoRA delta: returns (U [in×k], S [k], Vh [k×out]).

        This is the full rank-k concept subspace, not just the dominant direction.
        Use for rank-k transplant — the concept lives in a k-dimensional subspace
        at 1B+ scale, not just a single vector.
        """
        delta = self.get_delta()
        if delta.norm() < 1e-10:
            k_ = min(k, delta.shape[0], delta.shape[1])
            return (
                torch.zeros(self.in_features, k_),
                torch.zeros(k_),
                torch.zeros(k_, self.out_features),
            )
        try:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        except Exception:
            k_ = min(k, delta.shape[0], delta.shape[1])
            return (
                torch.zeros(self.in_features, k_),
                torch.zeros(k_),
                torch.zeros(k_, self.out_features),
            )
        k_ = min(k, S.shape[0])
        return U[:, :k_], S[:k_], Vh[:k_, :]


# ─── Main class ───────────────────────────────────────────────────────────────

class LoRACartographer:
    """Train LoRA adapters on concept sentences and extract concept maps.

    The concept map tells you WHERE in weight space each concept lives —
    which layers, which directions, how strongly — giving you a surgical
    address for every concept the model has learned.

    Args:
        loader:   A loaded ModelLoader instance
        arch_map: An ArchitectureMap from ArchitectureMapper
        rank:     Default LoRA rank (default 4 is enough for concept geometry)
    """

    # Module path fragments that identify MLP-only linear layers
    # NOTE: c_proj and proj intentionally omitted — they appear in attention too
    _MLP_FRAGMENTS = {"mlp", "ffn", "feed_forward", "c_fc", "fc"}
    # Attention path fragments — modules matching these are always excluded
    _ATTN_FRAGMENTS = {"attn", "attention", "self_attn"}

    def __init__(self, loader, arch_map, rank: int = 4):
        self.loader = loader
        self.arch_map = arch_map
        self.rank = rank
        # Maps injected module path → (parent_module, child_attr_name, original_module)
        self._injected: Dict[str, Tuple[nn.Module, str, nn.Module]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def train_concept(
        self,
        concept: str,
        sentences: List[str],
        epochs: int = 3,
        rank: Optional[int] = None,
        lr: float = 5e-4,
        log_every_n_steps: int = 5,
        verbose: bool = True,
    ) -> ConceptMap:
        """Train LoRA on concept sentences, logging concept directions per layer.

        For each optimizer step, records the dominant singular vector of each
        LoRA delta (lora_A @ lora_B), producing a frame-by-frame movie of
        how the concept builds up across layers over training.

        Args:
            concept:           The concept name (e.g. "fire")
            sentences:         Training sentences expressing this concept
            epochs:            Passes over the sentence list
            rank:              LoRA rank override (default: self.rank)
            lr:                Learning rate for LoRA parameters
            log_every_n_steps: How often to snapshot directions
            verbose:           Print progress

        Returns:
            ConceptMap with per-layer concept directions across all steps
        """
        if not sentences:
            raise ValueError(f"Need at least one sentence for concept '{concept}'")

        effective_rank = rank if rank is not None else self.rank
        model = self.loader.model
        tokenizer = self.loader.tokenizer
        device = next(model.parameters()).device

        if verbose:
            print(
                f"\n[Cartography] Mapping concept: '{concept}'  "
                f"({len(sentences)} sentences × {epochs} epochs, rank={effective_rank})"
            )

        # ── Inject LoRA into MLP layers in the fact zone ─────────────────────
        lora_modules = self._inject_lora(effective_rank)
        if verbose:
            print(f"  LoRA injected into {len(lora_modules)} layers")
            print(
                f"  Fact zone: layers "
                f"{self.arch_map.fact_layer_start}–{self.arch_map.fact_layer_end}"
            )

        if not lora_modules:
            self._remove_lora()
            raise RuntimeError(
                "No LoRA injection targets found. "
                "Check that arch_map is populated and model layers are accessible."
            )

        # ── Freeze base params; train only LoRA ──────────────────────────────
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad_(False)

        lora_params = []
        for lora_layer in lora_modules.values():
            lora_layer.lora_A.requires_grad_(True)
            lora_layer.lora_B.requires_grad_(True)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)

        # ── Training loop ─────────────────────────────────────────────────────
        frames_by_layer: Dict[str, List[ConceptFrame]] = {n: [] for n in lora_modules}
        step = 0

        for epoch in range(epochs):
            for sentence in sentences:
                enc = tokenizer(
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=False,
                )
                input_ids = enc["input_ids"].to(device)

                if input_ids.shape[1] < 2:
                    continue  # Need at least one token to predict

                try:
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss
                except Exception:
                    continue

                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                # Clip gradient norms to keep training stable
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                step += 1

                # Log concept directions at this step
                if step == 1 or step % log_every_n_steps == 0:
                    for layer_name, lora_layer in lora_modules.items():
                        direction, magnitude = lora_layer.get_direction()
                        if magnitude > 1e-8:
                            frames_by_layer[layer_name].append(
                                ConceptFrame(
                                    step=step,
                                    layer_name=layer_name,
                                    direction=direction,
                                    magnitude=magnitude,
                                )
                            )

        # ── Extract final converged directions + rank-k SVD ─────────────────
        final_directions: Dict[str, torch.Tensor] = {}
        final_svd: Dict[str, dict] = {}
        for layer_name, lora_layer in lora_modules.items():
            direction, magnitude = lora_layer.get_direction()
            if magnitude > 1e-8:
                final_directions[layer_name] = direction
                U_k, S_k, Vh_k = lora_layer.get_directions_k(k=self.rank)
                final_svd[layer_name] = {
                    "U": U_k,
                    "S": S_k,
                    "Vh": Vh_k,
                    "rank": int(S_k.shape[0]),
                }

        # ── Layer ownership = mean magnitude over training ────────────────────
        layer_ownership_dict: Dict[str, float] = {}
        for layer_name, frames in frames_by_layer.items():
            if frames:
                layer_ownership_dict[layer_name] = (
                    sum(f.magnitude for f in frames) / len(frames)
                )
        ownership_sorted = sorted(
            layer_ownership_dict.items(), key=lambda x: x[1], reverse=True
        )

        if verbose:
            print(f"  Training complete: {step} optimizer steps")
            if ownership_sorted:
                print("  Top concept-owning layers:")
                for lname, score in ownership_sorted[:5]:
                    bar = "█" * max(1, int(score * 15))
                    print(f"    {lname:<50} {score:>7.4f}  {bar}")

        # ── Restore model ─────────────────────────────────────────────────────
        self._remove_lora()
        for name, param in model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad_(original_requires_grad[name])

        return ConceptMap(
            concept=concept,
            model_name=self.loader.model_name,
            frames_by_layer={k: v for k, v in frames_by_layer.items() if v},
            final_directions=final_directions,
            layer_ownership=ownership_sorted,
            final_svd=final_svd,
        )

    def fast_map(
        self,
        concept: str,
        sentences: List[str],
        verbose: bool = False,
    ) -> ConceptMap:
        """Map a concept using gradient attribution — no training, milliseconds.

        Computes ∂loss/∂W for each MLP weight matrix W in the fact zone.
        SVD of that gradient gives the dominant direction the model would need
        to move to better encode the concept — equivalent to what LoRA converges
        to after many steps, but in a single backward pass.

        Advantages over train_concept():
            - ~100-1000x faster (no optimizer loop)
            - Safe on quantized models (gradients still flow through hooks)
            - Good first-pass scan for hundreds of concepts at once

        Disadvantages:
            - Higher noise — single-pass gradient is raw, not time-averaged
            - Magnitude scores less discriminative (no learning dynamics)

        Args:
            concept:   The concept name (e.g. "fire")
            sentences: Sentences expressing this concept (averaged over)
            verbose:   Print progress

        Returns:
            ConceptMap in the same format as train_concept()
        """
        if not sentences:
            raise ValueError(f"Need at least one sentence for concept '{concept}'")

        model = self.loader.model
        tokenizer = self.loader.tokenizer
        device = next(model.parameters()).device

        if verbose:
            print(f"\n[Cartography:fast] '{concept}'  ({len(sentences)} sentences)")

        # Collect target MLP modules in fact zone
        target_modules: Dict[str, nn.Module] = {}
        for name, module in model.named_modules():
            is_linear = isinstance(module, nn.Linear)
            is_gpconv = _is_conv1d(module)
            if not (is_linear or is_gpconv):
                continue
            if not self._module_is_mlp(name):
                continue
            if not self._module_in_fact_zone(name):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue
            try:
                if module.weight.data.dtype not in (
                    torch.float32, torch.float16, torch.bfloat16
                ):
                    continue
            except Exception:
                continue
            target_modules[name] = module

        if not target_modules:
            raise RuntimeError("No target MLP modules found for fast_map.")

        # Accumulate gradients across all sentences
        grad_acc: Dict[str, torch.Tensor] = {}

        original_grads = {}
        for name, module in target_modules.items():
            if module.weight.requires_grad:
                original_grads[name] = True
            else:
                module.weight.requires_grad_(True)
                original_grads[name] = False

        for sentence in sentences:
            enc = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=False,
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue

            # Zero grads on target weights
            for module in target_modules.values():
                if module.weight.grad is not None:
                    module.weight.grad.zero_()

            try:
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
            except Exception:
                continue

            if not torch.isfinite(loss):
                continue

            loss.backward()

            for name, module in target_modules.items():
                if module.weight.grad is None:
                    continue
                g = module.weight.grad.detach().float().cpu()
                if name in grad_acc:
                    grad_acc[name] = grad_acc[name] + g
                else:
                    grad_acc[name] = g.clone()

        # Restore requires_grad
        for name, module in target_modules.items():
            if not original_grads.get(name, True):
                module.weight.requires_grad_(False)

        # SVD of accumulated gradient → concept direction per layer
        final_directions: Dict[str, torch.Tensor] = {}
        layer_ownership_dict: Dict[str, float] = {}
        final_svd: Dict[str, dict] = {}

        for name, grad in grad_acc.items():
            if grad.norm() < 1e-12:
                continue
            try:
                # grad shape: [out, in] for Linear or [in, out] for Conv1D
                # We want dominant left singular vector in input space (dim = in_features)
                # For Linear [out, in]: svd → U [out, r], S [r], Vh [r, in]  → Vh[0] is in-space
                # For Conv1D [in, out]: svd → U [in, r], S [r], Vh [r, out] → U[:,0] is in-space
                module = target_modules[name]
                if _is_conv1d(module):
                    U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
                    direction = F.normalize(U[:, 0], dim=0)
                else:
                    # grad: [out, in] — transpose to get [in, out], then U col 0
                    U, S, Vh = torch.linalg.svd(grad.T, full_matrices=False)
                    direction = F.normalize(U[:, 0], dim=0)
                magnitude = float(S[0].item())
                final_directions[name] = direction
                layer_ownership_dict[name] = magnitude
                # Store rank-k SVD for rank-k transplant
                k_ = min(self.rank, S.shape[0])
                final_svd[name] = {
                    "U": U[:, :k_],
                    "S": S[:k_],
                    "Vh": Vh[:k_, :],
                    "rank": int(k_),
                }
            except Exception:
                continue

        ownership_sorted = sorted(
            layer_ownership_dict.items(), key=lambda x: x[1], reverse=True
        )

        if verbose:
            print(f"  Done: {len(final_directions)} layers mapped")
            if ownership_sorted:
                for lname, score in ownership_sorted[:5]:
                    bar = "█" * max(1, int(min(score / max(v for _, v in ownership_sorted), 1.0) * 18))
                    print(f"    {lname:<50} {score:>10.2f}  {bar}")

        return ConceptMap(
            concept=concept,
            model_name=self.loader.model_name,
            frames_by_layer={},   # no step-by-step logging in fast mode
            final_directions=final_directions,
            layer_ownership=ownership_sorted,
            final_svd=final_svd,
        )

    def fast_map_batch(
        self,
        concepts: Dict[str, List[str]],
        verbose: bool = True,
    ) -> Dict[str, ConceptMap]:
        """Rapidly map many concepts.  concepts = {name: [sentence, ...]}

        Scans an entire concept vocabulary in seconds.  Returns a dict of
        ConceptMaps that can be compared, saved, and used for transplants.
        """
        results: Dict[str, ConceptMap] = {}
        total = len(concepts)
        for i, (concept, sentences) in enumerate(concepts.items()):
            if verbose:
                print(f"  fast_map [{i+1}/{total}]: '{concept}' ...", end=" ", flush=True)
            try:
                cmap = self.fast_map(concept, sentences)
                results[concept] = cmap
                if verbose:
                    top = cmap.layer_ownership[0][0].split(".")[-2:] if cmap.layer_ownership else ["?"]
                    print(f"→ {'.'.join(top)}")
            except Exception as e:
                if verbose:
                    print(f"→ SKIP ({e})")
        return results

    @staticmethod
    def build_alignment_matrix(
        maps_a: Dict[str, ConceptMap],
        maps_b: Dict[str, ConceptMap],
        max_dim: int = 768,
    ) -> ConceptAlignmentMatrix:
        """Compute the orthogonal rotation aligning model A's geometry to model B.

        Uses the orthogonal Procrustes problem over all shared concept directions
        at matched normalized layer depths.

        Given:
            A_dirs [n × d]  — concept directions in model A
            B_dirs [n × d]  — same concepts in model B
        Finds:
            R = argmin ||A_dirs @ R − B_dirs||_F  subject to R^T R = I
        Solution (SVD):
            M = A_dirs.T @ B_dirs
            U, S, Vh = svd(M)
            R = U @ Vh

        With R, you can rotate any concept direction from model A into model B's
        coordinate system — enabling weight transplant without re-training.

        Args:
            maps_a:   Dict of ConceptMaps from the donor model
            maps_b:   Dict of ConceptMaps from the target model
            max_dim:  Cap vector dimension (trim to min(dim_a, dim_b, max_dim))

        Returns:
            ConceptAlignmentMatrix with rotation R and residual error
        """
        shared = sorted(set(maps_a.keys()) & set(maps_b.keys()))
        if len(shared) < 3:
            raise ValueError(
                f"Need ≥ 3 shared concepts to compute alignment, got {len(shared)}. "
                f"Model A has {list(maps_a.keys())}, B has {list(maps_b.keys())}"
            )

        def _mean_direction(cmap: ConceptMap, dim: int) -> Optional[torch.Tensor]:
            """Averaged concept direction across all layers, truncated to dim."""
            dirs = list(cmap.final_directions.values())
            if not dirs:
                return None
            vecs = []
            for d in dirs:
                d = d.float().cpu()
                if d.shape[0] >= dim:
                    vecs.append(d[:dim])
                else:
                    pad = torch.zeros(dim - d.shape[0])
                    vecs.append(torch.cat([d, pad]))
            mean_vec = torch.stack(vecs).mean(0)
            return F.normalize(mean_vec, dim=0)

        # Determine common dimension
        all_dims_a = [
            next(iter(maps_a[c].final_directions.values())).shape[0]
            for c in shared if maps_a[c].final_directions
        ]
        all_dims_b = [
            next(iter(maps_b[c].final_directions.values())).shape[0]
            for c in shared if maps_b[c].final_directions
        ]
        if not all_dims_a or not all_dims_b:
            raise ValueError("Concept maps have no final directions.")

        dim = min(
            max(all_dims_a),  # use dominant dim from A
            max(all_dims_b),
            max_dim,
        )

        rows_a, rows_b, used = [], [], []
        for concept in shared:
            va = _mean_direction(maps_a[concept], dim)
            vb = _mean_direction(maps_b[concept], dim)
            if va is None or vb is None:
                continue
            rows_a.append(va)
            rows_b.append(vb)
            used.append(concept)

        if len(used) < 3:
            raise ValueError(f"Only {len(used)} concepts had valid directions.")

        A_mat = torch.stack(rows_a)  # [n, dim]
        B_mat = torch.stack(rows_b)  # [n, dim]

        # Orthogonal Procrustes: R = U @ Vh where M = A.T @ B
        M = A_mat.T @ B_mat          # [dim, dim]
        U, _S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh                   # [dim, dim]  — orthogonal rotation

        # Residual: how well does A @ R approximate B?
        residual = float(torch.norm(A_mat @ R - B_mat, p="fro").item())

        return ConceptAlignmentMatrix(
            model_a=next(iter(maps_a.values())).model_name,
            model_b=next(iter(maps_b.values())).model_name,
            shared_concepts=used,
            R=R,
            residual=residual,
        )

    @staticmethod
    def build_layerwise_alignment(
        maps_a: Dict[str, "ConceptMap"],
        maps_b: Dict[str, "ConceptMap"],
        n_bins: int = 4,
        max_dim: int = 768,
    ) -> "LayerwiseAlignment":
        """Per-depth Procrustes — one rotation matrix per depth bucket.

        Unlike the global build_alignment_matrix(), this computes a separate
        R for each depth slice of the network.  This handles the fact that
        'arithmetic' lives at depth 0.5 in GPT-2 but depth 0.625 in LLaMA —
        the rotation needed at mid-layers differs from early or late layers.

        Args:
            maps_a:   Concept maps from donor model (concept → ConceptMap)
            maps_b:   Concept maps from target model (same concepts)
            n_bins:   Number of depth buckets (default 4)
            max_dim:  Cap vector dimension

        Returns:
            LayerwiseAlignment with one R matrix per depth bucket
        """
        shared = sorted(set(maps_a.keys()) & set(maps_b.keys()))
        if len(shared) < 3:
            raise ValueError(
                f"Need ≥ 3 shared concepts to compute layerwise alignment, "
                f"got {len(shared)}."
            )

        # Resolve common vector dimension
        all_dims: List[int] = []
        for concept in shared:
            for cmap in [maps_a[concept], maps_b[concept]]:
                for v in cmap.final_directions.values():
                    all_dims.append(int(v.shape[0]))
        if not all_dims:
            raise ValueError("No final directions found in concept maps.")
        dim = min(max(all_dims), max_dim)

        def _dirs_by_depth(cmap: "ConceptMap") -> Dict[float, torch.Tensor]:
            layers = sorted(cmap.final_directions.keys())
            n = len(layers)
            if n == 0:
                return {}
            out: Dict[float, torch.Tensor] = {}
            for i, lname in enumerate(layers):
                depth = round(i / max(n - 1, 1), 2)
                v = cmap.final_directions[lname].float().cpu()
                if v.shape[0] > dim:
                    v = v[:dim]
                elif v.shape[0] < dim:
                    v = torch.cat([v, torch.zeros(dim - v.shape[0])])
                out[depth] = F.normalize(v, dim=0)
            return out

        edges = [i / n_bins for i in range(n_bins + 1)]
        mids = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        R_per_depth: Dict[str, torch.Tensor] = {}
        residuals_per_depth: Dict[str, float] = {}

        for lo, hi, mid in zip(edges[:-1], edges[1:], mids):
            mid_str = f"{mid:.2f}"
            rows_a: List[torch.Tensor] = []
            rows_b: List[torch.Tensor] = []

            for concept in shared:
                dirs_a = _dirs_by_depth(maps_a[concept])
                dirs_b = _dirs_by_depth(maps_b[concept])
                for depth, va in dirs_a.items():
                    in_bin = (lo <= depth < hi) or (hi == 1.0 and depth == 1.0)
                    if not in_bin:
                        continue
                    if not dirs_b:
                        continue
                    d_b = min(dirs_b.keys(), key=lambda d: abs(d - depth))
                    if abs(d_b - depth) > 0.15:
                        continue
                    rows_a.append(va)
                    rows_b.append(dirs_b[d_b])

            if len(rows_a) < 2:
                R_per_depth[mid_str] = torch.eye(dim)
                residuals_per_depth[mid_str] = 0.0
                continue

            A_mat = torch.stack(rows_a)  # [n, dim]
            B_mat = torch.stack(rows_b)  # [n, dim]
            M = A_mat.T @ B_mat          # [dim, dim]
            U, _S, Vh = torch.linalg.svd(M, full_matrices=False)
            R = U @ Vh
            residual = float(torch.norm(A_mat @ R - B_mat, p="fro").item())
            R_per_depth[mid_str] = R
            residuals_per_depth[mid_str] = residual

        return LayerwiseAlignment(
            model_a=next(iter(maps_a.values())).model_name,
            model_b=next(iter(maps_b.values())).model_name,
            shared_concepts=list(shared),
            n_bins=n_bins,
            R_per_depth=R_per_depth,
            residuals_per_depth=residuals_per_depth,
        )

    def transplant_concept(
        self,
        concept_map: ConceptMap,
        alignment: "Union[ConceptAlignmentMatrix, LayerwiseAlignment]",
        rank_k: int = 4,
        scale: float = 0.05,
        existing_maps: Optional[Dict[str, ConceptMap]] = None,
        verbose: bool = True,
    ) -> TransplantReport:
        """Write a concept from a donor model into this model's MLP weights.

        For each layer, rotates the donor concept's rank-k SVD delta into the
        target coordinate system using conjugation (R.T @ Δ @ R), then writes
        the result into the target weight.  Falls back to rank-1 outer product
        if no stored SVD data is available.

        Args:
            concept_map:    ConceptMap from the donor model
            alignment:      ConceptAlignmentMatrix or LayerwiseAlignment
            rank_k:         How many singular components to transplant (default 4)
            scale:          Weight-edit strength (default 0.05)
            existing_maps:  If provided, run interference check before transplant
            verbose:        Print per-layer edit sizes

        Returns:
            TransplantReport with edit norms, interference info, alignment info
        """
        model = self.loader.model
        edit_norms: Dict[str, float] = {}

        # --- Pre-transplant interference check ---
        interference: Optional[InterferenceReport] = None
        if existing_maps is not None:
            interference = self.check_interference(concept_map, alignment, existing_maps)
            if verbose:
                print(interference.summary())
            if interference.rating == "abort":
                if verbose:
                    print("  [Transplant ABORTED] Interference too high — concept overlap detected.")
                return TransplantReport(
                    concept=concept_map.concept,
                    layers_edited=0,
                    total_delta_norm=0.0,
                    edit_norms={},
                    interference=interference,
                )

        if verbose:
            print(f"\n[Cartography:transplant] '{concept_map.concept}'")
            print(f"  Donor: {concept_map.model_name}  →  Target: {self.loader.model_name}")
            res = getattr(alignment, "residual", None) or getattr(alignment, "global_residual", 0.0)
            print(f"  Alignment residual: {res:.4f}  "
                  f"(shared concepts: {len(alignment.shared_concepts)})")
            print(f"  Scale: {scale}  |  rank_k={rank_k}  |  Layers to edit: {len(concept_map.final_directions)}")

        module_map = {n: m for n, m in model.named_modules()}
        sorted_layers = sorted(concept_map.final_directions.keys())
        n_layers = max(len(sorted_layers) - 1, 1)

        def _get_R(layer_name: str) -> torch.Tensor:
            """Return the R matrix appropriate for this layer's depth."""
            if isinstance(alignment, LayerwiseAlignment):
                idx = sorted_layers.index(layer_name) if layer_name in sorted_layers else 0
                depth = idx / n_layers
                nearest = min(
                    alignment.R_per_depth.keys(),
                    key=lambda d: abs(float(d) - depth),
                )
                return alignment.R_per_depth[nearest]
            return alignment.R  # ConceptAlignmentMatrix

        for layer_name, donor_dir in concept_map.final_directions.items():
            target_module = module_map.get(layer_name)
            if target_module is None:
                target_module = self._find_nearest_mlp_by_depth(
                    layer_name, concept_map.model_name, module_map
                )
            if target_module is None:
                continue
            if not hasattr(target_module, "weight") or target_module.weight is None:
                continue
            try:
                weight_data = target_module.weight.data
            except Exception:
                continue

            is_gpconv = _is_conv1d(target_module)
            R = _get_R(layer_name).float().cpu()
            dim = R.shape[0]

            # ── Rank-k delta via conjugation ──────────────────────────────────
            svd_data = concept_map.final_svd.get(layer_name)
            if svd_data is not None:
                U_full = svd_data["U"]
                S_full = svd_data["S"]
                Vh_full = svd_data["Vh"]
                k = min(rank_k, int(svd_data["rank"]))

                if isinstance(U_full, list):
                    U_full = torch.tensor(U_full)
                    S_full = torch.tensor(S_full)
                    Vh_full = torch.tensor(Vh_full)

                U_full = U_full.float().cpu()
                S_full = S_full.float().cpu()
                Vh_full = Vh_full.float().cpu()

                U_k = U_full[:, :k]          # [in, k]
                S_k = S_full[:k]             # [k]
                Vh_k = Vh_full[:k, :]        # [k, out]

                # donor_delta [in, out] = U_k @ diag(S_k) @ Vh_k
                donor_delta = (U_k * S_k.unsqueeze(0)) @ Vh_k  # [in, out]

                if is_gpconv:
                    # Conv1D weight [in, out] — delta orientation matches
                    d_in, d_out = donor_delta.shape
                else:
                    # Linear weight [out, in] — need weight-space orientation
                    donor_delta = donor_delta.T  # [out, in]
                    d_in, d_out = donor_delta.shape

                # Pad / trim donor_delta to [dim, dim] for conjugation
                if d_in < dim:
                    donor_delta = torch.cat(
                        [donor_delta, torch.zeros(dim - d_in, d_out)], dim=0
                    )
                elif d_in > dim:
                    donor_delta = donor_delta[:dim, :]
                d_in = donor_delta.shape[0]

                if d_out < dim:
                    donor_delta = torch.cat(
                        [donor_delta, torch.zeros(d_in, dim - d_out)], dim=1
                    )
                elif d_out > dim:
                    donor_delta = donor_delta[:, :dim]

                # Conjugation: Δ_target = R.T @ Δ_donor @ R
                rotated = R.T @ donor_delta @ R  # [dim, dim]

                def _fit_to_weight(rot: torch.Tensor, out_sz: int, in_sz: int) -> torch.Tensor:
                    """Pad rotated [dim,dim] delta to exactly [out_sz, in_sz]."""
                    r_out = min(out_sz, rot.shape[0])
                    r_in  = min(in_sz,  rot.shape[1])
                    if r_out == out_sz and r_in == in_sz:
                        return rot[:out_sz, :in_sz]
                    full = torch.zeros(out_sz, in_sz, dtype=rot.dtype, device=rot.device)
                    full[:r_out, :r_in] = rot[:r_out, :r_in]
                    return full

                if is_gpconv:
                    # Conv1D weight [in, out]
                    actual_in, actual_out = weight_data.shape[0], weight_data.shape[1]
                    delta_final = _fit_to_weight(rotated, actual_in, actual_out)
                else:
                    # Linear weight [out, in]
                    actual_out, actual_in = weight_data.shape[0], weight_data.shape[1]
                    delta_final = _fit_to_weight(rotated, actual_out, actual_in)

            else:
                # ── Rank-1 fallback (no SVD data stored) ─────────────────────
                v = F.normalize(donor_dir.float().cpu() @ R, dim=0)

                if is_gpconv:
                    in_dim = weight_data.shape[0]
                else:
                    in_dim = weight_data.shape[1]

                if v.shape[0] != in_dim:
                    v = v[:in_dim] if v.shape[0] > in_dim else torch.cat([v, torch.zeros(in_dim - v.shape[0])])
                v = F.normalize(v, dim=0)
                v_dev = v

                if is_gpconv:
                    out_dim = weight_data.shape[1]
                    v_out = v_dev[:out_dim] if v_dev.shape[0] >= out_dim else F.pad(v_dev, (0, out_dim - v_dev.shape[0]))
                    delta_final = torch.outer(v_dev, v_out)
                else:
                    out_dim = weight_data.shape[0]
                    v_out = v_dev[:out_dim] if v_dev.shape[0] >= out_dim else F.pad(v_dev, (0, out_dim - v_dev.shape[0]))
                    delta_final = torch.outer(v_out, v_dev)

            delta_final = delta_final.to(device=weight_data.device, dtype=weight_data.dtype)
            edit_norm = float((scale * delta_final).norm().item())
            with torch.no_grad():
                weight_data.add_(scale * delta_final)

            edit_norms[layer_name] = edit_norm

            if verbose:
                bar = "▓" * max(1, int(min(edit_norm * 200, 20)))
                print(f"    {layer_name:<50}  Δ={edit_norm:.5f}  {bar}")

        total_delta_norm = sum(edit_norms.values())
        if verbose:
            print(f"\n  Transplant complete. Total weight change: {total_delta_norm:.5f}")
            print(f"  Run probe_after_transplant() to verify the graft took.")

        return TransplantReport(
            concept=concept_map.concept,
            layers_edited=len(edit_norms),
            total_delta_norm=total_delta_norm,
            edit_norms=edit_norms,
            interference=interference,
        )

    def _find_nearest_mlp_by_depth(
        self,
        donor_layer_name: str,
        donor_model_name: str,
        module_map: Dict[str, nn.Module],
    ) -> Optional[nn.Module]:
        """Find the closest MLP layer in this model by normalized depth position."""
        # Extract depth from donor layer name (look for numeric parts)
        donor_parts = donor_layer_name.split(".")
        donor_idx = None
        for p in donor_parts:
            if p.isdigit():
                donor_idx = int(p)
                break
        if donor_idx is None:
            return None

        donor_depth = donor_idx / max(self.arch_map.num_layers - 1, 1)

        # Find all MLP modules in this model
        candidates = []
        for name, module in module_map.items():
            is_lin = isinstance(module, nn.Linear)
            is_gpconv = _is_conv1d(module)
            if not (is_lin or is_gpconv):
                continue
            if not self._module_is_mlp(name):
                continue
            # Extract layer index
            parts = name.split(".")
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    depth = idx / max(self.arch_map.num_layers - 1, 1)
                    candidates.append((abs(depth - donor_depth), name, module))
                    break

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][2]

    def check_interference(
        self,
        concept_map: ConceptMap,
        alignment: "Union[ConceptAlignmentMatrix, LayerwiseAlignment]",
        existing_maps: Dict[str, ConceptMap],
        threshold_abort: float = 0.7,
        threshold_caution: float = 0.4,
    ) -> InterferenceReport:
        """Scan for concept collisions before performing a transplant.

        Rotates each donor direction into target space, then checks cosine
        similarity against every existing concept's directions at the same
        layer.  High cosine means the transplant would corrupt an existing concept.

        Args:
            concept_map:       ConceptMap to transplant (donor model)
            alignment:         Rotation mapping donor → target space
            existing_maps:     Concept maps already in the target model
            threshold_abort:   cosine > this → rating "abort"
            threshold_caution: cosine > this → rating "caution"

        Returns:
            InterferenceReport with per-layer conflict list and overall rating
        """
        sorted_layers = sorted(concept_map.final_directions.keys())
        n_layers = max(len(sorted_layers) - 1, 1)

        def _rotate(layer_name: str, vec: torch.Tensor) -> torch.Tensor:
            if isinstance(alignment, LayerwiseAlignment):
                idx = sorted_layers.index(layer_name) if layer_name in sorted_layers else 0
                depth = idx / n_layers
                return alignment.rotate(vec, depth)
            return alignment.rotate(vec)

        conflicts: List[Tuple[str, str, float]] = []
        safe_layers: List[str] = []
        risky_layers: List[str] = []

        for layer_name, donor_dir in concept_map.final_directions.items():
            rotated = _rotate(layer_name, donor_dir).float()
            max_cos_here = 0.0
            worst_concept = ""

            for existing_concept, existing_map in existing_maps.items():
                if existing_concept == concept_map.concept:
                    continue
                if layer_name not in existing_map.final_directions:
                    continue
                existing_dir = existing_map.final_directions[layer_name].float().cpu()
                min_dim = min(rotated.shape[0], existing_dir.shape[0])
                cos = abs(float(
                    F.cosine_similarity(
                        rotated[:min_dim].unsqueeze(0),
                        existing_dir[:min_dim].unsqueeze(0),
                    ).item()
                ))
                if cos > max_cos_here:
                    max_cos_here = cos
                    worst_concept = existing_concept

            if max_cos_here >= threshold_caution:
                conflicts.append((layer_name, worst_concept, max_cos_here))
                risky_layers.append(layer_name)
            else:
                safe_layers.append(layer_name)

        max_cosine = max((c for _, _, c in conflicts), default=0.0)
        if max_cosine >= threshold_abort:
            rating = "abort"
        elif max_cosine >= threshold_caution:
            rating = "caution"
        else:
            rating = "safe"

        return InterferenceReport(
            concept=concept_map.concept,
            conflicts=conflicts,
            safe_layers=safe_layers,
            risky_layers=risky_layers,
            max_cosine=max_cosine,
            rating=rating,
        )

    def probe_after_transplant(
        self,
        concept: str,
        sentences: List[str],
        donor_map: ConceptMap,
        verbose: bool = True,
    ) -> float:
        """Verify that a transplant took by re-mapping the concept post-surgery.

        Runs fast_map() on the concept in THIS model (after transplant) and
        compares the resulting directions against the donor's directions.
        A high alignment score (>0.5) means the graft was successful.

        Args:
            concept:    The concept name that was transplanted
            sentences:  The same (or similar) sentences used for the donor map
            donor_map:  The original ConceptMap from the donor model
            verbose:    Print alignment score

        Returns:
            global_alignment score in [0, 1]
        """
        post_map = self.fast_map(concept, sentences, verbose=False)
        alignment = LoRACartographer.compare_maps(donor_map, post_map)

        if verbose:
            print(
                f"[probe_after_transplant] '{concept}'  "
                f"post-surgery alignment = {alignment.global_alignment:.3f}"
            )

        return alignment.global_alignment

    def save_map(self, concept_map: ConceptMap, path: str) -> None:
        """Save a ConceptMap to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(concept_map.to_dict(), f, indent=2)
        print(f"[Cartography] Saved: {out}")

    @staticmethod
    def load_map(path: str) -> ConceptMap:
        """Load a ConceptMap from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ConceptMap.from_dict(data)

    @staticmethod
    def compare_maps(
        map_a: ConceptMap,
        map_b: ConceptMap,
    ) -> CartographyAlignment:
        """Compare how two models store the same concept.

        Assigns each layer a normalized depth position (0.0–1.0) and computes
        cosine similarity between concept directions at matching depth levels.

        High global_alignment (>0.6) is direct evidence for the Platonic
        Representation Hypothesis — independent models converge to the same
        geometric encoding for the same concept.

        If comparing a map with itself, global_alignment should be ≈ 1.0.

        Args:
            map_a: ConceptMap from the first model
            map_b: ConceptMap from the second model

        Returns:
            CartographyAlignment with per-depth cosine similarities
        """

        def _depth_buckets(
            cmap: ConceptMap,
        ) -> Dict[float, torch.Tensor]:
            """Normalized depth → averaged concept direction."""
            layers = sorted(cmap.final_directions.keys())
            n = len(layers)
            if n == 0:
                return {}
            buckets: Dict[float, List[torch.Tensor]] = {}
            for i, layer_name in enumerate(layers):
                depth = round(i / max(n - 1, 1), 2)
                buckets.setdefault(depth, []).append(
                    cmap.final_directions[layer_name]
                )
            return {
                d: F.normalize(torch.stack(vecs).float().mean(0), dim=0)
                for d, vecs in buckets.items()
            }

        buckets_a = _depth_buckets(map_a)
        buckets_b = _depth_buckets(map_b)

        depths_a = sorted(buckets_a)
        depths_b = sorted(buckets_b)

        cosine_per_layer: Dict[str, float] = {}

        for d_a in depths_a:
            if not depths_b:
                break
            # Nearest depth in model B (within 15% tolerance)
            d_b = min(depths_b, key=lambda d: abs(d - d_a))
            if abs(d_a - d_b) > 0.15:
                continue

            vec_a = buckets_a[d_a].float()
            vec_b = buckets_b[d_b].float()

            # Handle cross-architecture dimension mismatch
            min_dim = min(vec_a.shape[0], vec_b.shape[0])
            cos = float(
                F.cosine_similarity(
                    vec_a[:min_dim].unsqueeze(0),
                    vec_b[:min_dim].unsqueeze(0),
                ).item()
            )
            cosine_per_layer[f"depth_{d_a:.2f}"] = cos

        global_alignment = (
            sum(cosine_per_layer.values()) / len(cosine_per_layer)
            if cosine_per_layer
            else 0.0
        )

        return CartographyAlignment(
            concept=map_a.concept,
            model_a=map_a.model_name,
            model_b=map_b.model_name,
            cosine_per_layer=cosine_per_layer,
            global_alignment=global_alignment,
        )

    def print_map(self, concept_map: ConceptMap, top_k: int = 8) -> None:
        """Print a human-readable concept map summary."""
        width = 72
        print(f"\n{'='*width}")
        print(f"  CONCEPT MAP: '{concept_map.concept}'")
        print(f"  Model: {concept_map.model_name}")
        print(f"{'='*width}")
        print(f"  {'LAYER':<50} {'OWNERSHIP':>10}  {'STEPS':>6}")
        print(f"  {'-'*(width-2)}")

        for lname, score in concept_map.layer_ownership[:top_k]:
            n_steps = len(concept_map.frames_by_layer.get(lname, []))
            bar = "█" * max(1, int(score * 18))
            print(f"  {lname:<50} {score:>8.4f}  {n_steps:>5}  {bar}")

        total = len(concept_map.final_directions)
        print(f"{'='*width}")
        print(f"  Total concept-owning layers: {total}")
        print(f"  Dominant layers: "
              f"{[n for n, _ in concept_map.layer_ownership[:3]]}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fact_zone_indices(self) -> set:
        """Layer indices in the fact-storage zone."""
        return set(
            range(self.arch_map.fact_layer_start, self.arch_map.fact_layer_end + 1)
        )

    def _module_in_fact_zone(self, module_path: str) -> bool:
        """True if the module path contains a layer index in the fact zone."""
        fact_indices = self._fact_zone_indices()
        for part in module_path.split("."):
            if part.isdigit() and int(part) in fact_indices:
                return True
        # If no numeric index found in path, include it (conservative)
        return True

    def _module_is_mlp(self, module_path: str) -> bool:
        """True if the module path refers to an MLP-style linear layer.

        Explicitly excludes attention sub-paths (e.g. GPT-2 attn.c_proj).
        """
        low = module_path.lower()
        # Exclude any attention sub-module
        if any(frag in low for frag in self._ATTN_FRAGMENTS):
            return False
        return any(frag in low for frag in self._MLP_FRAGMENTS)

    def _inject_lora(self, rank: int) -> Dict[str, _LoRALinear]:
        """Replace eligible linear layers with LoRA wrappers.

        Returns a dict mapping module path → LoRA layer for all injected modules.
        """
        model = self.loader.model
        self._injected = {}
        lora_modules: Dict[str, _LoRALinear] = {}

        for name, module in list(model.named_modules()):
            is_linear = isinstance(module, nn.Linear)
            is_gpconv = _is_conv1d(module)
            if not (is_linear or is_gpconv):
                continue

            if not self._module_is_mlp(name):
                continue

            if not self._module_in_fact_zone(name):
                continue

            if not hasattr(module, "weight") or module.weight is None:
                continue

            # Skip bitsandbytes quantized layers (can't wrap them)
            cls_name = type(module).__name__.lower()
            if "4bit" in cls_name or "8bit" in cls_name or "bnb" in cls_name:
                continue

            # Verify weight is accessible (not quantized/offloaded)
            try:
                if module.weight.data.dtype not in (
                    torch.float32, torch.float16, torch.bfloat16
                ):
                    continue
            except Exception:
                continue

            try:
                lora_layer = _LoRALinear(module, rank=rank)
            except Exception:
                continue

            # Find parent and replace child attribute
            parent_path, _, child_name = name.rpartition(".")
            parent = (
                model
                if parent_path == ""
                else self._get_submodule(model, parent_path)
            )
            if parent is None:
                continue

            setattr(parent, child_name, lora_layer)
            self._injected[name] = (parent, child_name, module)
            lora_modules[name] = lora_layer

        return lora_modules

    def _remove_lora(self) -> None:
        """Restore all original modules, removing LoRA wrappers."""
        for _name, (parent, child_name, original) in self._injected.items():
            setattr(parent, child_name, original)
        self._injected = {}

    @staticmethod
    def _get_submodule(model: nn.Module, path: str) -> Optional[nn.Module]:
        """Walk dot-separated path to a submodule; returns None if missing."""
        m = model
        for part in path.split("."):
            if not hasattr(m, part):
                return None
            m = getattr(m, part)
        return m
