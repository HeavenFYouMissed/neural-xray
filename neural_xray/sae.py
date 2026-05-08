"""
sae.py — Sparse Autoencoder (AutoPsy Stage 3c: DECOMPOSE)
=========================================================

Based on Anthropic "Scaling Monosemanticity" (2024) and Cunningham et al.
(arXiv:2309.08600).

The problem with raw cosine similarity (what tracer.py currently does):
    A single hidden-dim activation mixes dozens of concepts at once — one
    neuron might respond to "fire", "heat", AND "danger" simultaneously.
    This polysemanticity makes cosine similarity noisy and unreliable.

The SAE solution:
    Train an overcomplete dictionary (n_features >> hidden_dim) with L1
    sparsity on the activations. The L1 penalty forces the model to use
    as few features as possible to reconstruct any given activation. The
    result: each feature becomes monosemantic — it activates for exactly
    one coherent concept. ~300 of 8192 features are active per token.

Architecture (Anthropic formulation):
    Pre-bias trick:   b_pre = learned mean of the activations
    Encoder:          f(x) = ReLU(W_enc(x - b_pre) + b_enc)    [n_features]
    Decoder:          x̂   = W_dec · f(x) + b_pre              [hidden_dim]
    Loss:             L    = ||x - x̂||² + λ·||f(x)||₁
    Constraint:       ||W_dec[:, i]||₂ = 1  after each step

Architecture-specific layer targeting (arXiv:2602.06852, Quantum Sieve Tracer):
    Qwen2.x:  layer 7  — "Recall Hub"   (ablating degrades recall)
    Llama2/3: layer 9  — "Interference Suppression" (ablating IMPROVES recall)
    Default:  middle third of layers (falls back to Geva/ROME heuristic)

Usage:
    from antroslammer.autopsy.sae import SparseAutoencoder

    sae = SparseAutoencoder(hidden_dim=1536, n_features=8192)
    sae.train_on_model(loader, arch_map, sentences, epochs=3)
    sae.save("autopsy_output/sae.pt")

    # Decompose a single activation into monosemantic features:
    features = sae.encode(activation)           # [n_features] sparse
    active_idx, active_vals = sae.get_active_features(activation)

    # Align feature directions to concept vocabulary (needed by tracer):
    feature_to_concept = sae.align_features_to_concepts(concept_vectors)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Architecture-specific layer targeting ────────────────────────────────────

# Maps model name substrings → (target_layer, layer_role_description)
# Source: arXiv:2602.06852 (Quantum Sieve Tracer, Feb 2026)
_ARCH_LAYER_MAP = {
    "qwen2":  (7,  "recall_hub"),
    "qwen":   (7,  "recall_hub"),
    "llama":  (9,  "interference_suppression"),
    "mistral": (8, "fact_storage"),
    "phi":    (6,  "fact_storage"),
}


def get_target_layer(model_name: str, num_layers: int) -> Tuple[int, str]:
    """Return the best single layer to hook for SAE training.

    Uses architecture-specific research findings where available,
    falls back to middle-third heuristic.

    Returns:
        (layer_index, role_description)
    """
    lower = model_name.lower()
    for key, (layer_idx, role) in _ARCH_LAYER_MAP.items():
        if key in lower:
            # Safety: clamp to valid range in case the model has fewer layers
            actual = min(layer_idx, num_layers - 1)
            return actual, role

    # Fallback: geometric middle (ROME/Geva heuristic)
    mid = num_layers // 2
    return mid, "fact_storage_middle"


# ─── SAE model ────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """
    Overcomplete sparse autoencoder for decomposing transformer activations
    into monosemantic features.

    Args:
        hidden_dim:   Dimension of model's residual stream (e.g. 1536 for Qwen2.5-1.5B)
        n_features:   Number of SAE features (should be >> hidden_dim, e.g. 8192)
        l1_coeff:     L1 sparsity coefficient λ (controls feature density; 3e-4 is a
                      good default — roughly 300 active features per token on Qwen1.5B)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_features: int = 8192,
        l1_coeff: float = 3e-4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.l1_coeff = l1_coeff

        # Encoder: Linear + bias (acts on x - b_pre)
        self.W_enc = nn.Linear(hidden_dim, n_features, bias=True)
        # Decoder: Linear, no bias (bias handled by b_pre)
        self.W_dec = nn.Linear(n_features, hidden_dim, bias=False)
        # Pre-bias = learned mean of training activations (Anthropic trick)
        self.b_pre = nn.Parameter(torch.zeros(hidden_dim))

        # Decoder columns are kept unit-norm during training
        self._normalize_decoder()

        # Meta: set after training or loading
        self.target_layer: int = -1
        self.target_layer_role: str = "unknown"
        self.model_name: str = ""
        self.train_stats: dict = {}

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Activation tensor [batch, hidden_dim] or [hidden_dim]

        Returns:
            (x_hat, features, l2_loss, l1_loss)
            - x_hat:    [batch, hidden_dim] reconstruction
            - features: [batch, n_features] sparse feature activations
            - l2_loss:  scalar MSE reconstruction loss
            - l1_loss:  scalar L1 sparsity loss (weighted by l1_coeff)
        """
        x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)  # [1, hidden_dim]

        # Subtract pre-bias, encode, apply ReLU
        x_centered = x - self.b_pre.unsqueeze(0)            # [B, hidden_dim]
        features = F.relu(self.W_enc(x_centered))            # [B, n_features]

        # Decode back to hidden dim, re-add pre-bias
        x_hat = self.W_dec(features) + self.b_pre.unsqueeze(0)  # [B, hidden_dim]

        # Losses
        l2_loss = F.mse_loss(x_hat, x)
        l1_loss = self.l1_coeff * features.abs().mean()

        return x_hat, features, l2_loss, l1_loss

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single activation into its sparse feature representation.

        Args:
            x: [hidden_dim] or [batch, hidden_dim]

        Returns:
            features: [n_features] or [batch, n_features] sparse (most values ≈ 0)
        """
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = x.float()
        x_centered = x - self.b_pre.unsqueeze(0)
        features = F.relu(self.W_enc(x_centered))
        return features.squeeze(0) if squeeze else features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to hidden dim space.

        Args:
            features: [n_features] or [batch, n_features]

        Returns:
            x_hat: [hidden_dim] or [batch, hidden_dim]
        """
        squeeze = features.ndim == 1
        if squeeze:
            features = features.unsqueeze(0)
        x_hat = self.W_dec(features.float()) + self.b_pre.unsqueeze(0)
        return x_hat.squeeze(0) if squeeze else x_hat

    def get_active_features(
        self,
        x: torch.Tensor,
        threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return indices + values of active (non-zero) features.

        Args:
            x:         [hidden_dim] activation
            threshold: Minimum feature activation to consider "active"

        Returns:
            (indices, values) sorted by value descending
        """
        features = self.encode(x.to(self.b_pre.device))  # [n_features]
        mask = features > threshold
        indices = mask.nonzero(as_tuple=True)[0]
        values = features[indices]
        # Sort descending by activation strength
        order = values.argsort(descending=True)
        return indices[order], values[order]

    # ── Decoder column normalization ──────────────────────────────────────────

    @torch.no_grad()
    def _normalize_decoder(self):
        """Normalize each decoder column to unit length (Anthropic constraint)."""
        norms = self.W_dec.weight.data.norm(dim=0, keepdim=True)  # [1, n_features]
        norms = norms.clamp(min=1e-8)
        self.W_dec.weight.data = self.W_dec.weight.data / norms

    # ── Feature → Concept alignment ──────────────────────────────────────────

    def align_features_to_concepts(
        self,
        concept_vectors: Dict[str, torch.Tensor],
        top_k: int = 1,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        For each SAE feature (decoder column direction), find the closest concept(s)
        in the concept vocabulary via cosine similarity.

        This bridges the SAE feature space back to human-readable concept names,
        enabling monosemantic concept identification in the tracer.

        Args:
            concept_vectors: Dict[concept_name → tensor(hidden_dim)]
            top_k:           How many concept names to assign per feature

        Returns:
            Dict[feature_index → [(concept_name, similarity), ...]]  sorted by sim desc
        """
        device = self.W_dec.weight.device

        # Build concept matrix [N, D], normalized
        concept_names = list(concept_vectors.keys())
        concept_mat = torch.stack([
            F.normalize(v.float(), dim=-1) for v in concept_vectors.values()
        ]).to(device)  # [N, D]

        # Decoder columns [D, n_features] → transpose → [n_features, D], normalize
        decoder_cols = self.W_dec.weight.data.T  # [n_features, D]
        decoder_cols = F.normalize(decoder_cols, dim=-1)  # already unit norm, but ensure

        # Batch cosine similarity: [n_features, N]
        sims = torch.matmul(decoder_cols, concept_mat.T)  # [n_features, N]

        feature_to_concept: Dict[int, List[Tuple[str, float]]] = {}
        top_vals, top_idxs = torch.topk(sims, min(top_k, len(concept_names)), dim=-1)

        for feat_i in range(self.n_features):
            feature_to_concept[feat_i] = [
                (concept_names[top_idxs[feat_i, k].item()],
                 float(top_vals[feat_i, k].item()))
                for k in range(top_k)
            ]

        return feature_to_concept

    # ── Training ─────────────────────────────────────────────────────────────

    def train_on_model(
        self,
        loader,
        arch_map,
        sentences: List[str],
        epochs: int = 3,
        batch_size: int = 256,
        lr: float = 2e-4,
        warmup_steps: int = 200,
        verbose: bool = True,
    ) -> dict:
        """
        Train the SAE on residual-stream activations collected from the source model.

        Process:
            1. Hook residual stream at the architecture-specific target layer
            2. Run all sentences through the model to harvest activations
            3. Train SAE with Adam optimizer (MSE + L1)
            4. Normalize decoder columns after each step

        Args:
            loader:        Loaded ModelLoader instance
            arch_map:      ArchitectureMap from mapper stage
            sentences:     Training sentences (can be concept probe sentences)
            epochs:        Training epochs over collected activations
            batch_size:    Mini-batch size for SAE training (NOT model batch size)
            lr:            Adam learning rate
            warmup_steps:  Linear warmup over this many optimizer steps
            verbose:       Print progress

        Returns:
            Dict with training stats (final_loss, l2_loss, l1_loss, active_features_avg)
        """
        model = loader.model
        tokenizer = loader.tokenizer

        # Determine which layer to target
        self.model_name = arch_map.model_name
        target_layer, role = get_target_layer(arch_map.model_name, arch_map.num_layers)
        self.target_layer = target_layer
        self.target_layer_role = role

        if verbose:
            print(f"\n[SAE] Training on '{arch_map.model_name}'")
            print(f"  Layer target: {target_layer} ({role})")
            print(f"  hidden_dim={self.hidden_dim}, n_features={self.n_features}")
            print(f"  l1_coeff={self.l1_coeff}, lr={lr}, epochs={epochs}")
            print(f"  Collecting activations from {len(sentences)} sentences…")

        # ── Step 1: Collect activations ───────────────────────────────────────
        all_activations = self._collect_activations(
            model, tokenizer, arch_map, sentences, target_layer, verbose
        )

        if len(all_activations) == 0:
            raise RuntimeError(
                f"No activations collected! Check that layer {target_layer} "
                f"exists in {arch_map.model_name}."
            )

        # Stack into tensor and move to training device
        device = next(self.parameters()).device
        acts = torch.cat(all_activations, dim=0).to(device)  # [T, hidden_dim]

        if verbose:
            print(f"  Collected {acts.shape[0]} token activations, "
                  f"shape={tuple(acts.shape)}")

        # Initialize b_pre as mean of training activations (Anthropic trick)
        with torch.no_grad():
            self.b_pre.data = acts.mean(dim=0)

        # ── Step 2: Train SAE ─────────────────────────────────────────────────
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n = acts.shape[0]
        total_steps = (n // batch_size + 1) * epochs
        step = 0

        stats = {
            "total_tokens": n,
            "epochs": epochs,
            "steps": 0,
            "final_loss": float("inf"),
            "final_l2": float("inf"),
            "final_l1": float("inf"),
            "active_features_avg": 0.0,
        }

        for epoch in range(epochs):
            # Shuffle each epoch
            perm = torch.randperm(n)
            acts_shuffled = acts[perm]

            epoch_loss = epoch_l2 = epoch_l1 = epoch_active = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = acts_shuffled[start: start + batch_size]
                if batch.shape[0] == 0:
                    continue

                # Linear LR warmup
                if step < warmup_steps:
                    scale = (step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr * scale

                optimizer.zero_grad(set_to_none=True)
                _, features, l2_loss, l1_loss = self.forward(batch)
                loss = l2_loss + l1_loss
                loss.backward()
                optimizer.step()

                # Normalize decoder columns after every step (critical!)
                self._normalize_decoder()

                # Track stats
                with torch.no_grad():
                    active = (features > 0).float().sum(dim=-1).mean().item()
                epoch_loss  += loss.item()
                epoch_l2    += l2_loss.item()
                epoch_l1    += l1_loss.item()
                epoch_active += active
                n_batches   += 1
                step        += 1

            if verbose and n_batches > 0:
                avg_loss   = epoch_loss   / n_batches
                avg_l2     = epoch_l2     / n_batches
                avg_l1     = epoch_l1     / n_batches
                avg_active = epoch_active / n_batches
                print(f"  Epoch {epoch+1}/{epochs}  "
                      f"loss={avg_loss:.4f}  l2={avg_l2:.4f}  "
                      f"l1={avg_l1:.4f}  active={avg_active:.1f}")

        # Final pass for stats
        self.eval()
        with torch.no_grad():
            sample = acts[:min(1024, n)]
            _, feats, l2, l1 = self.forward(sample)
            active_avg = (feats > 0).float().sum(dim=-1).mean().item()

        stats.update({
            "steps": step,
            "final_loss": float(l2 + l1),
            "final_l2": float(l2),
            "final_l1": float(l1),
            "active_features_avg": float(active_avg),
            "target_layer": target_layer,
            "target_layer_role": role,
        })
        self.train_stats = stats

        if verbose:
            print(f"\n[SAE] Training complete.")
            print(f"  Final L2={stats['final_l2']:.4f}  L1={stats['final_l1']:.4f}")
            print(f"  Avg active features/token: {stats['active_features_avg']:.1f} "
                  f"(target: ~300)")
            if stats["active_features_avg"] > 500:
                print(f"  NOTE: too many active features — try increasing l1_coeff")
            elif stats["active_features_avg"] < 50:
                print(f"  NOTE: too few active features — try decreasing l1_coeff")

        return stats

    def _collect_activations(
        self,
        model,
        tokenizer,
        arch_map,
        sentences: List[str],
        target_layer: int,
        verbose: bool,
    ) -> List[torch.Tensor]:
        """Run sentences through model and collect residual stream at target_layer."""

        # Find the module path for the target layer's residual output
        # Strategy: hook the layer container's target_layer child module
        target_path = self._find_layer_path(arch_map, target_layer)
        if verbose:
            print(f"  Hooking module: {target_path}")

        all_acts: List[torch.Tensor] = []

        def _hook(module, inp, out):
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                # [batch, seq, hidden] → detach, move to CPU to save VRAM
                acts_cpu = tensor.detach().float().cpu()
                if acts_cpu.ndim == 3:
                    # Flatten batch+seq: [B*T, D]
                    B, T, D = acts_cpu.shape
                    acts_cpu = acts_cpu.view(B * T, D)
                elif acts_cpu.ndim == 2:
                    pass  # already [T, D] or [B, D]
                all_acts.append(acts_cpu)

        # Locate the target module in the model
        target_module = None
        for name, module in model.named_modules():
            if name == target_path:
                target_module = module
                break

        if target_module is None:
            raise RuntimeError(
                f"Could not find module '{target_path}' in model. "
                f"Check that target_layer={target_layer} is valid."
            )

        handle = target_module.register_forward_hook(_hook)

        device_str = "cuda" if next(model.parameters()).is_cuda else "cpu"

        model.eval()
        try:
            with torch.no_grad():
                for i, sent in enumerate(sentences):
                    if verbose and i % 200 == 0 and i > 0:
                        print(f"    {i}/{len(sentences)} sentences…")
                    try:
                        inputs = tokenizer(
                            sent,
                            return_tensors="pt",
                            truncation=True,
                            max_length=64,
                        ).to(device_str)
                        model(**inputs)
                    except Exception:
                        continue  # skip malformed sentences
        finally:
            handle.remove()

        return all_acts

    def _find_layer_path(self, arch_map, target_layer: int) -> str:
        """Find the module path string for the target layer in the arch map."""
        if arch_map.layers and target_layer < len(arch_map.layers):
            return arch_map.layers[target_layer].name  # e.g. "model.layers.7"

        # Fallback: try to guess from the first layer's name pattern
        if arch_map.layers:
            first_name = arch_map.layers[0].name  # e.g. "model.layers.0"
            parts = first_name.rsplit(".", 1)
            if len(parts) == 2:
                prefix, _ = parts
                return f"{prefix}.{target_layer}"

        raise RuntimeError(
            f"Could not determine module path for layer {target_layer}. "
            f"Run the MAP stage first."
        )

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save SAE weights + hyperparameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_dim": self.hidden_dim,
                "n_features": self.n_features,
                "l1_coeff": self.l1_coeff,
                "target_layer": self.target_layer,
                "target_layer_role": self.target_layer_role,
                "model_name": self.model_name,
                "train_stats": self.train_stats,
            },
            path,
        )
        print(f"[SAE] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "SparseAutoencoder":
        """Load a previously saved SAE.

        Args:
            path:   Path to .pt file saved by SparseAutoencoder.save()
            device: Target device ("cuda" / "cpu"). Auto-detects if None.

        Returns:
            Loaded SparseAutoencoder (in eval mode)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=device, weights_only=True)

        sae = cls(
            hidden_dim=checkpoint["hidden_dim"],
            n_features=checkpoint["n_features"],
            l1_coeff=checkpoint["l1_coeff"],
        )
        sae.load_state_dict(checkpoint["state_dict"])
        sae.target_layer      = checkpoint.get("target_layer", -1)
        sae.target_layer_role = checkpoint.get("target_layer_role", "unknown")
        sae.model_name        = checkpoint.get("model_name", "")
        sae.train_stats       = checkpoint.get("train_stats", {})
        sae.to(device)
        sae.eval()

        print(f"[SAE] Loaded from {path}")
        print(f"  Layer {sae.target_layer} ({sae.target_layer_role}), "
              f"n_features={sae.n_features}, "
              f"avg_active={sae.train_stats.get('active_features_avg', '?'):.1f}")
        return sae

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def reconstruction_quality(
        self,
        activations: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute quality metrics on a batch of activations.

        Args:
            activations: [batch, hidden_dim]

        Returns:
            Dict with mse, explained_variance, avg_active_features, l0_sparsity
        """
        self.eval()
        with torch.no_grad():
            x = activations.float().to(self.b_pre.device)
            x_hat, features, l2, l1 = self.forward(x)

            # Explained variance (higher = better, 1.0 = perfect)
            var_x = x.var(dim=0).mean().item()
            var_err = (x - x_hat).var(dim=0).mean().item()
            explained_var = 1.0 - (var_err / (var_x + 1e-8))

            avg_active = (features > 0).float().sum(dim=-1).mean().item()
            l0 = avg_active / self.n_features  # fraction of features active

        return {
            "mse": float(l2),
            "explained_variance": float(explained_var),
            "avg_active_features": float(avg_active),
            "l0_sparsity": float(l0),
        }

    def print_top_features(
        self,
        x: torch.Tensor,
        feature_to_concept: Optional[Dict[int, List[Tuple[str, float]]]] = None,
        top_n: int = 10,
    ):
        """Print the most active features for an activation, with concept labels if available.

        Args:
            x:                  [hidden_dim] activation
            feature_to_concept: Output of align_features_to_concepts() (optional)
            top_n:              How many top features to show
        """
        indices, values = self.get_active_features(x)
        print(f"\nTop {min(top_n, len(indices))} active features "
              f"(of {len(indices)} total active):")
        for rank in range(min(top_n, len(indices))):
            feat_i = int(indices[rank].item())
            val    = float(values[rank].item())
            if feature_to_concept:
                labels = feature_to_concept.get(feat_i, [])
                label_str = ", ".join(f"{c}({s:.2f})" for c, s in labels)
            else:
                label_str = f"feature_{feat_i}"
            print(f"  [{feat_i:5d}] {val:6.3f}  {label_str}")
