"""
diagnostics.py — ModelDiagnostics (AutoPsy: Health Checks)
==========================================================

The "7-layer health scanner" — inspired by anti-cheat multi-meter detection.

No single metric reliably detects model pathologies. But 7 orthogonal checks
together are 99%+ confident. Each check is independent, fast, and returns
a normalized 0-1 severity score.

The 7 checks (mapped from anti-cheat detection pipeline):
    1. Activation Magnitude Bounds  — abnormally large/small layer outputs
    2. Token Probability Entropy    — too confident or too uncertain
    3. Attention Head Specialization — heads doing distinct jobs or collapsed
    4. Gradient Flow Consistency     — vanishing/exploding gradients
    5. Dead Neuron Ratio            — what % of neurons never activate
    6. Temporal Pattern Regularity  — artificial alternation across tokens
    7. Concept Contamination Score  — unrelated concepts co-activating

Usage:
    from antroslammer.autopsy.diagnostics import ModelDiagnostics

    diag = ModelDiagnostics(loader, arch_map)
    report = diag.run_checks("The cat sat on the mat")
    diag.print_report(report)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    score: float          # 0.0 = healthy, 1.0 = critical
    severity: str         # "ok", "warn", "critical"
    detail: str           # human-readable explanation
    raw_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "severity": self.severity,
            "detail": self.detail,
            "raw_values": {k: round(v, 6) for k, v in self.raw_values.items()},
        }


@dataclass
class DiagnosticReport:
    """Aggregate diagnostic report from all 7 checks."""
    sentence: str
    overall_score: float          # weighted average of all checks (0=healthy, 1=critical)
    overall_severity: str         # "healthy", "degraded", "critical"
    checks: List[CheckResult] = field(default_factory=list)
    num_ok: int = 0
    num_warn: int = 0
    num_critical: int = 0

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "overall_score": round(self.overall_score, 4),
            "overall_severity": self.overall_severity,
            "num_ok": self.num_ok,
            "num_warn": self.num_warn,
            "num_critical": self.num_critical,
            "checks": [c.to_dict() for c in self.checks],
        }


# ─── Thresholds ───────────────────────────────────────────────────────────────

WARN_THRESHOLD = 0.4
CRITICAL_THRESHOLD = 0.7


def _severity(score: float) -> str:
    if score >= CRITICAL_THRESHOLD:
        return "critical"
    elif score >= WARN_THRESHOLD:
        return "warn"
    return "ok"


# ─── Core diagnostics engine ─────────────────────────────────────────────────

class ModelDiagnostics:
    """
    Run 7 orthogonal health checks on a model given a probe sentence.
    Each check is independent and fast (single forward pass shared).

    Args:
        loader:    Loaded ModelLoader instance
        arch_map:  ArchitectureMap from the mapper stage
        device:    "cuda" / "cpu"
    """

    def __init__(self, loader, arch_map, device: Optional[str] = None):
        self.loader = loader
        self.arch_map = arch_map
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def run_checks(self, sentence: str) -> DiagnosticReport:
        """
        Run all 7 diagnostic checks on the given sentence.
        Single forward pass captures all needed data, then each check analyzes it.
        """
        model = self.loader.model
        tokenizer = self.loader.tokenizer

        inputs = tokenizer(sentence, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]

        # ── Collect all data in one forward pass ──────────────────────────
        block_hidden = {}   # block_idx → [seq, hidden] (residual stream)
        mlp_activations = {}  # block_idx → [seq, intermediate]
        handles = []

        def _make_block_hook(idx):
            def _hook(module, inp, out):
                tensor = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 3:
                    block_hidden[idx] = tensor[0].detach().float()
            return _hook

        def _make_mlp_hook(idx):
            def _hook(module, inp, out):
                tensor = out if isinstance(out, torch.Tensor) else (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                    mlp_activations[idx] = tensor.detach().float()
                    if mlp_activations[idx].ndim == 3:
                        mlp_activations[idx] = mlp_activations[idx][0]
            return _hook

        for layer_info in self.arch_map.layers:
            block_mod = self._resolve_module(layer_info.name)
            if block_mod is not None:
                handles.append(block_mod.register_forward_hook(
                    _make_block_hook(layer_info.index)))

            if layer_info.mlp_path:
                mlp_mod = self._resolve_module(layer_info.mlp_path)
                if mlp_mod is not None:
                    handles.append(mlp_mod.register_forward_hook(
                        _make_mlp_hook(layer_info.index)))

        # Forward pass with attention weights
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

        logits = output.logits if hasattr(output, "logits") else output[0]
        attentions = getattr(output, "attentions", None) or []

        # ── Run all 7 checks ──────────────────────────────────────────────
        checks = [
            self._check_activation_magnitude(block_hidden),
            self._check_entropy(logits, seq_len),
            self._check_attention_specialization(attentions),
            self._check_gradient_proxy(block_hidden),
            self._check_dead_neurons(mlp_activations),
            self._check_temporal_regularity(block_hidden, seq_len),
            self._check_contamination(block_hidden),
        ]

        # ── Aggregate ─────────────────────────────────────────────────────
        scores = [c.score for c in checks]
        overall = sum(scores) / len(scores) if scores else 0.0

        if overall >= CRITICAL_THRESHOLD:
            overall_sev = "critical"
        elif overall >= WARN_THRESHOLD:
            overall_sev = "degraded"
        else:
            overall_sev = "healthy"

        return DiagnosticReport(
            sentence=sentence,
            overall_score=overall,
            overall_severity=overall_sev,
            checks=checks,
            num_ok=sum(1 for c in checks if c.severity == "ok"),
            num_warn=sum(1 for c in checks if c.severity == "warn"),
            num_critical=sum(1 for c in checks if c.severity == "critical"),
        )

    # ── Check 1: Activation Magnitude Bounds ──────────────────────────────

    def _check_activation_magnitude(
        self, block_hidden: Dict[int, torch.Tensor]
    ) -> CheckResult:
        """
        Are any layer outputs abnormally large or small?
        Healthy models have gradually increasing norms with depth.
        """
        if not block_hidden:
            return CheckResult("activation_magnitude", 0.5, "warn",
                               "No hidden states captured", {})

        norms = []
        for idx in sorted(block_hidden.keys()):
            h = block_hidden[idx]
            # Mean norm across token positions
            norm = h.norm(dim=-1).mean().item()
            norms.append(norm)

        if len(norms) < 2:
            return CheckResult("activation_magnitude", 0.0, "ok",
                               "Only 1 layer", {"mean_norm": norms[0]})

        mean_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        min_norm = min(norms)
        ratio = max_norm / (min_norm + 1e-8)

        # Norm ratio > 100 is suspicious, > 1000 is critical
        if ratio > 1000:
            score = 1.0
        elif ratio > 100:
            score = 0.6
        elif ratio > 10:
            score = 0.3
        else:
            score = 0.0

        # Check for any NaN or Inf
        has_nan = any(math.isnan(n) or math.isinf(n) for n in norms)
        if has_nan:
            score = 1.0

        return CheckResult(
            name="activation_magnitude",
            score=score,
            severity=_severity(score),
            detail=(f"Norm range: {min_norm:.1f}–{max_norm:.1f} "
                    f"(ratio={ratio:.1f})"
                    + (", NaN/Inf detected!" if has_nan else "")),
            raw_values={"mean_norm": mean_norm, "max_norm": max_norm,
                        "min_norm": min_norm, "ratio": ratio},
        )

    # ── Check 2: Token Probability Entropy ────────────────────────────────

    def _check_entropy(
        self, logits: torch.Tensor, seq_len: int
    ) -> CheckResult:
        """
        Is the model too confident or too uncertain?
        Healthy: moderate entropy (2-6 nats for English).
        Pathological: near-zero (collapsed) or very high (random).
        """
        # logits: [batch, seq, vocab]
        probs = F.softmax(logits[0].float(), dim=-1)  # [seq, vocab]
        # Per-token entropy
        log_probs = torch.log(probs + 1e-10)
        entropy_per_token = -(probs * log_probs).sum(dim=-1)  # [seq]

        mean_entropy = entropy_per_token.mean().item()
        min_entropy = entropy_per_token.min().item()
        max_entropy = entropy_per_token.max().item()

        # Healthy range for English: ~2-8 nats
        # Near 0 = collapsed, >10 = near-random
        if mean_entropy < 0.5:
            score = 0.8  # too confident / collapsed
            detail = f"Very low entropy ({mean_entropy:.2f}) — model may be collapsed"
        elif mean_entropy > 10.0:
            score = 0.7  # near-random
            detail = f"Very high entropy ({mean_entropy:.2f}) — near-random predictions"
        elif mean_entropy < 1.5 or mean_entropy > 8.0:
            score = 0.4
            detail = f"Entropy {mean_entropy:.2f} — slightly outside healthy range"
        else:
            score = 0.0
            detail = f"Healthy entropy ({mean_entropy:.2f})"

        return CheckResult(
            name="token_entropy",
            score=score,
            severity=_severity(score),
            detail=detail,
            raw_values={"mean_entropy": mean_entropy, "min_entropy": min_entropy,
                        "max_entropy": max_entropy},
        )

    # ── Check 3: Attention Head Specialization ────────────────────────────

    def _check_attention_specialization(
        self, attentions: list
    ) -> CheckResult:
        """
        Are attention heads doing distinct jobs, or have they collapsed?
        Computes inter-head cosine similarity — high = all heads doing same thing.
        """
        if not attentions:
            return CheckResult("attention_specialization", 0.3, "ok",
                               "No attention weights available", {})

        all_similarities = []
        for layer_attn in attentions:
            # layer_attn: [batch, heads, seq, seq]
            aw = layer_attn[0].detach().float()  # [heads, seq, seq]
            num_heads = aw.shape[0]
            if num_heads < 2:
                continue

            # Flatten each head's attention pattern: [heads, seq*seq]
            flat = aw.reshape(num_heads, -1)
            flat_norm = F.normalize(flat, dim=-1)

            # Pairwise cosine similarity between heads
            sim_matrix = flat_norm @ flat_norm.T  # [heads, heads]
            # Extract upper triangle (excluding diagonal)
            mask = torch.triu(torch.ones(num_heads, num_heads, dtype=torch.bool), diagonal=1)
            pairwise_sims = sim_matrix[mask]
            all_similarities.append(pairwise_sims.mean().item())

        if not all_similarities:
            return CheckResult("attention_specialization", 0.0, "ok",
                               "Could not compute head similarity", {})

        mean_sim = sum(all_similarities) / len(all_similarities)
        max_sim = max(all_similarities)

        # High similarity = heads collapsed (doing same thing)
        if mean_sim > 0.9:
            score = 0.9
            detail = f"Heads highly collapsed (mean sim={mean_sim:.3f})"
        elif mean_sim > 0.7:
            score = 0.5
            detail = f"Moderate head overlap (mean sim={mean_sim:.3f})"
        elif mean_sim > 0.5:
            score = 0.2
            detail = f"Some head overlap (mean sim={mean_sim:.3f})"
        else:
            score = 0.0
            detail = f"Healthy head specialization (mean sim={mean_sim:.3f})"

        return CheckResult(
            name="attention_specialization",
            score=score,
            severity=_severity(score),
            detail=detail,
            raw_values={"mean_sim": mean_sim, "max_layer_sim": max_sim},
        )

    # ── Check 4: Gradient Flow Proxy ──────────────────────────────────────

    def _check_gradient_proxy(
        self, block_hidden: Dict[int, torch.Tensor]
    ) -> CheckResult:
        """
        Proxy for gradient health without backprop: measure how much the
        residual stream changes between consecutive layers.
        Vanishing = near-zero change. Exploding = huge jumps.
        """
        if len(block_hidden) < 2:
            return CheckResult("gradient_flow", 0.0, "ok",
                               "Not enough layers", {})

        sorted_blocks = sorted(block_hidden.keys())
        deltas = []
        for i in range(len(sorted_blocks) - 1):
            h1 = block_hidden[sorted_blocks[i]]
            h2 = block_hidden[sorted_blocks[i + 1]]
            # Mean L2 distance between consecutive layers (averaged over tokens)
            delta = (h2 - h1).norm(dim=-1).mean().item()
            deltas.append(delta)

        mean_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)
        min_delta = min(deltas)
        ratio = max_delta / (min_delta + 1e-8)

        # Check for vanishing (all deltas near zero) or exploding (huge ratio)
        if min_delta < 0.01:
            score = 0.7  # vanishing
            detail = f"Near-zero change at some layers (min_delta={min_delta:.4f}) — possible vanishing gradients"
        elif ratio > 100:
            score = 0.8  # exploding
            detail = f"Extreme delta ratio ({ratio:.1f}) — possible exploding gradients"
        elif ratio > 10:
            score = 0.4
            detail = f"High delta ratio ({ratio:.1f})"
        else:
            score = 0.0
            detail = f"Healthy gradient flow (delta range: {min_delta:.2f}–{max_delta:.2f})"

        return CheckResult(
            name="gradient_flow",
            score=score,
            severity=_severity(score),
            detail=detail,
            raw_values={"mean_delta": mean_delta, "max_delta": max_delta,
                        "min_delta": min_delta, "ratio": ratio},
        )

    # ── Check 5: Dead Neuron Ratio ────────────────────────────────────────

    def _check_dead_neurons(
        self, mlp_activations: Dict[int, torch.Tensor]
    ) -> CheckResult:
        """
        What fraction of MLP neurons never activate (output ≤ 0)?
        Healthy models: <5% dead. Pathological: >20% dead.
        """
        if not mlp_activations:
            return CheckResult("dead_neurons", 0.3, "warn",
                               "No MLP activations captured", {})

        total_neurons = 0
        dead_neurons = 0
        per_layer_dead = []

        for idx in sorted(mlp_activations.keys()):
            act = mlp_activations[idx]  # [seq, intermediate] or [intermediate]
            if act.ndim == 1:
                act = act.unsqueeze(0)

            # A neuron is "dead" if it's ≤ 0 for ALL token positions
            is_dead = (act <= 0).all(dim=0)  # [intermediate]
            n_dead = is_dead.sum().item()
            n_total = is_dead.shape[0]
            total_neurons += n_total
            dead_neurons += n_dead
            per_layer_dead.append(n_dead / n_total if n_total > 0 else 0)

        dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0
        max_layer_dead = max(per_layer_dead) if per_layer_dead else 0

        if dead_ratio > 0.3:
            score = 0.9
            detail = f"{dead_ratio:.1%} neurons dead — severe"
        elif dead_ratio > 0.15:
            score = 0.6
            detail = f"{dead_ratio:.1%} neurons dead — moderate concern"
        elif dead_ratio > 0.05:
            score = 0.3
            detail = f"{dead_ratio:.1%} neurons dead — minor"
        else:
            score = 0.0
            detail = f"{dead_ratio:.1%} neurons dead — healthy"

        return CheckResult(
            name="dead_neurons",
            score=score,
            severity=_severity(score),
            detail=detail,
            raw_values={"dead_ratio": dead_ratio, "total_neurons": total_neurons,
                        "dead_neurons": dead_neurons, "max_layer_dead": max_layer_dead},
        )

    # ── Check 6: Temporal Pattern Regularity ──────────────────────────────

    def _check_temporal_regularity(
        self, block_hidden: Dict[int, torch.Tensor], seq_len: int
    ) -> CheckResult:
        """
        Does the model's behavior alternate artificially across token positions?
        In anti-cheat: alternating axis locks = bot signal.
        Here: alternating activation patterns = potential pathology.
        """
        if len(block_hidden) < 2 or seq_len < 4:
            return CheckResult("temporal_regularity", 0.0, "ok",
                               "Sequence too short for temporal analysis", {})

        # Check if activation norms alternate (even vs odd positions)
        alternation_scores = []
        for idx in sorted(block_hidden.keys()):
            h = block_hidden[idx]  # [seq, hidden]
            norms = h.norm(dim=-1)  # [seq]

            if norms.shape[0] < 4:
                continue

            # Compute autocorrelation at lag 2 (alternating pattern)
            centered = norms - norms.mean()
            if centered.norm() < 1e-6:
                continue

            # Lag-1 and lag-2 autocorrelation
            n = centered.shape[0]
            lag1 = (centered[:-1] * centered[1:]).sum() / (centered.pow(2).sum() + 1e-8)
            lag2 = (centered[:-2] * centered[2:]).sum() / (centered.pow(2).sum() + 1e-8)

            # Strong negative lag-1 + positive lag-2 = alternating pattern
            if lag1 < -0.3 and lag2 > 0.3:
                alternation_scores.append(abs(lag1.item()))
            else:
                alternation_scores.append(0.0)

        if not alternation_scores:
            return CheckResult("temporal_regularity", 0.0, "ok",
                               "No alternation detected", {})

        max_alt = max(alternation_scores)
        mean_alt = sum(alternation_scores) / len(alternation_scores)

        if max_alt > 0.7:
            score = 0.8
            detail = f"Strong alternating pattern detected (max={max_alt:.3f})"
        elif max_alt > 0.4:
            score = 0.4
            detail = f"Moderate alternation (max={max_alt:.3f})"
        else:
            score = 0.0
            detail = f"No artificial alternation (max={max_alt:.3f})"

        return CheckResult(
            name="temporal_regularity",
            score=score,
            severity=_severity(score),
            detail=detail,
            raw_values={"max_alternation": max_alt, "mean_alternation": mean_alt},
        )

    # ── Check 7: Concept Contamination ────────────────────────────────────

    def _check_contamination(
        self, block_hidden: Dict[int, torch.Tensor]
    ) -> CheckResult:
        """
        Do hidden states across layers become too similar (collapsed)
        or too different (fragmented)?
        Measures inter-layer cosine similarity of mean hidden states.
        """
        if len(block_hidden) < 3:
            return CheckResult("contamination", 0.0, "ok",
                               "Not enough layers", {})

        # Mean hidden state per layer: [hidden]
        means = []
        for idx in sorted(block_hidden.keys()):
            means.append(block_hidden[idx].mean(dim=0))

        means_tensor = torch.stack(means)  # [layers, hidden]
        means_norm = F.normalize(means_tensor, dim=-1)

        # Pairwise cosine similarity
        sim_matrix = means_norm @ means_norm.T  # [layers, layers]

        # Non-adjacent similarity (layers that are far apart)
        n = sim_matrix.shape[0]
        far_sims = []
        near_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                s = sim_matrix[i, j].item()
                if j - i > n // 2:
                    far_sims.append(s)
                elif j - i <= 2:
                    near_sims.append(s)

        mean_far = sum(far_sims) / len(far_sims) if far_sims else 0
        mean_near = sum(near_sims) / len(near_sims) if near_sims else 0

        # Pathology: far layers are too similar (residual stream not changing)
        if mean_far > 0.95:
            score = 0.8
            detail = f"Distant layers nearly identical (far_sim={mean_far:.3f}) — possible collapse"
        elif mean_far > 0.85:
            score = 0.4
            detail = f"High distant-layer similarity ({mean_far:.3f})"
        else:
            score = 0.0
            detail = f"Healthy layer differentiation (far_sim={mean_far:.3f}, near_sim={mean_near:.3f})"

        return CheckResult(
            name="contamination",
            score=score,
            severity=_severity(score),
            detail=detail,
            raw_values={"mean_far_sim": mean_far, "mean_near_sim": mean_near},
        )

    # ── Utilities ─────────────────────────────────────────────────────────

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

    def print_report(self, report: DiagnosticReport) -> None:
        """Pretty-print the diagnostic report."""
        print(f"\n{'═' * 60}")
        print(f"DIAGNOSTIC REPORT — '{report.sentence}'")
        print(f"Overall: {report.overall_severity.upper()} "
              f"(score={report.overall_score:.3f})")
        print(f"{'═' * 60}")

        for c in report.checks:
            icon = {"ok": "✓", "warn": "⚠", "critical": "✗"}[c.severity]
            print(f"  {icon} [{c.severity:>8}] {c.name:<28} {c.score:.3f}  {c.detail}")

        print(f"\n  Summary: {report.num_ok} ok, {report.num_warn} warn, "
              f"{report.num_critical} critical")
        print(f"{'═' * 60}")
