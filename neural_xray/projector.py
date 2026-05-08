"""
Stage 5: PROJECT
================
Train a small projection network that maps from source model space
(e.g. Qwen's 2048-dim hidden space) to target model space (our 512-dim space).

This is the bridge between the two worlds. Without it, a Qwen concept vector
is meaningless to our model because the two models use different geometric
spaces to represent meaning.

How we train it:
- We have "anchor" concepts: the 72 foundational concepts that are simple
  and unambiguous (fire, water, ball, rock...).
- We know what those should look like in our target space: they're the
  concepts the target model already encodes from training data.
- We train the projector to map source_vec(anchor) → target_vec(anchor)
- Then apply it to all other concepts to project them into target space.

The projector is tiny (4M params) and trains in minutes on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple


class ProjectionNetwork(nn.Module):
    """Two-layer MLP that maps source_dim → target_dim.

    Architecture:
        source_dim → bottleneck → target_dim → L2 normalize

    The L2 normalization at the end ensures projected vectors
    live on the unit sphere, consistent with how we'll use them
    (cosine similarity comparisons in the target model).
    """

    def __init__(self, source_dim: int, target_dim: int, bottleneck: Optional[int] = None):
        super().__init__()
        if bottleneck is None:
            bottleneck = max(target_dim, (source_dim + target_dim) // 2)

        self.fc1 = nn.Linear(source_dim, bottleneck, bias=True)
        self.fc2 = nn.Linear(bottleneck, target_dim, bias=True)
        self.norm = nn.LayerNorm(target_dim)

        # Initialize to approximate identity-like mapping
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return x


class ConceptProjector:
    """Train and apply a projection from source embedding space to target space.

    Args:
        source_dim: Hidden size of source model (e.g. 2048 for Qwen-1.5B)
        target_dim: Hidden size of target model (e.g. 512 for our 30M model)
        bottleneck: Intermediate dimension (default: average of source and target)
    """

    def __init__(self, source_dim: int, target_dim: int, bottleneck: Optional[int] = None):
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.network = ProjectionNetwork(source_dim, target_dim, bottleneck)
        self._trained = False

    def train(
        self,
        source_vectors: Dict[str, torch.Tensor],
        target_vectors: Dict[str, torch.Tensor],
        epochs: int = 2000,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ) -> "ConceptProjector":
        """Train the projector using anchor concept pairs.

        source_vectors and target_vectors must share some concept keys.
        Those shared concepts act as supervision signal.

        Args:
            source_vectors: Dict[concept_name → source_dim vector]
            target_vectors: Dict[concept_name → target_dim vector]
            epochs: Training steps
            lr: Learning rate
            device: "cpu" or "cuda"
            verbose: Print training progress
        """
        # Find shared concepts (anchors)
        shared = sorted(set(source_vectors.keys()) & set(target_vectors.keys()))
        if len(shared) < 5:
            raise ValueError(
                f"Need at least 5 shared concepts for projection training, got {len(shared)}. "
                f"Source has {len(source_vectors)}, target has {len(target_vectors)}"
            )

        if verbose:
            print(f"\n[AutoPsy:PROJECT] Training projector")
            print(f"  Source dim: {self.source_dim} → Target dim: {self.target_dim}")
            print(f"  Anchor concepts: {len(shared)}")
            print(f"  Training on: {device}")

        # Build training tensors
        src = torch.stack([source_vectors[c].float() for c in shared]).to(device)  # [N, src_dim]
        tgt = torch.stack([target_vectors[c].float() for c in shared]).to(device)  # [N, tgt_dim]

        # Normalize targets
        tgt = F.normalize(tgt, dim=1)

        self.network = self.network.to(device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.network.train()
        best_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            optimizer.zero_grad()
            projected = self.network(src)                          # [N, tgt_dim]
            projected_norm = F.normalize(projected, dim=1)

            # Cosine similarity loss: maximize similarity to target
            cos_loss = 1.0 - (projected_norm * tgt).sum(dim=1).mean()

            # MSE loss: also match magnitudes
            mse_loss = F.mse_loss(projected, tgt)

            loss = cos_loss + 0.1 * mse_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}

            if verbose and (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs}  loss={loss.item():.4f}  cos={cos_loss.item():.4f}")

        # Restore best weights
        if best_state:
            self.network.load_state_dict(best_state)

        self.network.eval()
        self._trained = True

        if verbose:
            print(f"  Best loss: {best_loss:.4f}")

            # Verify on anchors
            with torch.no_grad():
                proj = F.normalize(self.network(src), dim=1)
                sim = (proj * tgt).sum(dim=1).mean().item()
            print(f"  Anchor alignment (cosine sim): {sim:.3f} (1.0 = perfect)")

        return self

    @torch.no_grad()
    def project(self, vectors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Project a dict of source vectors into target space.

        Returns a new dict with the same keys but target-dim vectors.
        """
        if not self._trained:
            raise RuntimeError("Projector not trained — call .train() first")

        self.network.eval()
        results = {}
        for name, vec in vectors.items():
            projected = self.network(vec.float().unsqueeze(0)).squeeze(0)
            results[name] = projected.cpu()
        return results

    def save(self, path: str):
        """Save the trained projector."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.network.cpu().state_dict(),
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "trained": self._trained,
        }, str(p))
        print(f"[AutoPsy:PROJECT] Saved projector → {p}")

    @classmethod
    def load(cls, path: str) -> "ConceptProjector":
        """Load a trained projector from disk."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        proj = cls(data["source_dim"], data["target_dim"])
        proj.network.load_state_dict(data["state_dict"])
        proj._trained = data.get("trained", True)
        proj.network.eval()
        print(f"[AutoPsy:PROJECT] Loaded projector {data['source_dim']}→{data['target_dim']}")
        return proj

    def get_target_anchors(self, target_model, vocab, tok) -> Dict[str, torch.Tensor]:
        """Extract concept vectors from the target model's current embedding layer.

        These become the supervision targets for projection training.
        The target model's embeddings encode what IT thinks each concept means.

        Args:
            target_model: Our AnthroSlammerModel (StructuredAnthroSlammer)
            vocab: ConceptVocabulary
            tok: CortexLangTokenizer

        Returns:
            Dict[concept_name → target_dim vector]
        """
        # Handle both StructuredAnthroSlammer (direct) and base model wrappers
        if hasattr(target_model, 'tok_embeddings'):
            embedding_weight = target_model.tok_embeddings.weight.detach().cpu()
        else:
            embedding_weight = target_model.base_model.tok_embeddings.weight.detach().cpu()
        # entity token_id = entity_base (5000) + concept_id
        entity_base = tok._entity_base  # 5000

        anchors = {}
        for name in vocab._name_to_id:
            concept = vocab.lookup(name)
            if concept is None:
                continue
            token_id = entity_base + concept.concept_id
            if token_id < embedding_weight.shape[0]:
                anchors[name] = embedding_weight[token_id].float()
        return anchors
