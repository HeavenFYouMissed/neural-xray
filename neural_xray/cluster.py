"""
Stage 4a: CLUSTER
=================
After extracting concept vectors, build a semantic map:
- Which concepts cluster together (similar vector directions)?
- What are the main "knowledge regions" in the source model?
- Are "fire" and "heat" close together? Is "water" near "ice"?

This is how you see the model's internal knowledge structure.
It also validates that extraction worked — if "fire" and "steam" 
cluster together but are far from "democracy", the extraction is good.
"""

import torch
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ConceptCluster:
    """Build a semantic map from extracted concept vectors.

    Uses cosine similarity to measure concept proximity.
    Groups concepts into clusters using simple agglomerative approach.

    Args:
        vectors: Dict mapping concept_name → vector [hidden_size]
    """

    def __init__(self, vectors: Dict[str, torch.Tensor]):
        self.vectors = vectors
        self.names: List[str] = list(vectors.keys())
        self._matrix: Optional[torch.Tensor] = None  # cosine similarity matrix

    def build(self) -> "ConceptCluster":
        """Compute the full pairwise cosine similarity matrix."""
        print(f"\n[AutoPsy:CLUSTER] Building similarity matrix for {len(self.names)} concepts")

        # Stack all vectors into a matrix [N, hidden]
        vecs = torch.stack([self.vectors[n] for n in self.names])  # [N, H]

        # L2-normalize for cosine similarity
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = vecs / norms  # [N, H]

        # Cosine similarity matrix [N, N]
        self._matrix = normalized @ normalized.T  # [N, N]
        print(f"  Similarity matrix: {self._matrix.shape}")
        return self

    def nearest(self, concept: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find the top-k most similar concepts to a given concept."""
        if concept not in self.names:
            return []
        if self._matrix is None:
            self.build()

        idx = self.names.index(concept)
        sims = self._matrix[idx]  # [N]

        # Exclude self (similarity = 1.0)
        topk_vals, topk_ids = sims.topk(top_k + 1)
        results = []
        for val, i in zip(topk_vals.tolist(), topk_ids.tolist()):
            if self.names[i] != concept:
                results.append((self.names[i], round(val, 4)))
        return results[:top_k]

    def clusters(self, threshold: float = 0.7) -> List[List[str]]:
        """Group concepts into clusters based on similarity threshold.

        Returns a list of concept groups where all members have
        cosine similarity >= threshold with at least one other member.
        """
        if self._matrix is None:
            self.build()

        n = len(self.names)
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue
            # Find all concepts within threshold distance of this one
            group = [self.names[i]]
            visited.add(i)
            for j in range(n):
                if j != i and j not in visited:
                    if self._matrix[i, j].item() >= threshold:
                        group.append(self.names[j])
                        visited.add(j)
            if len(group) > 1:
                groups.append(sorted(group))

        # Sort by cluster size (largest first)
        return sorted(groups, key=len, reverse=True)

    def similarity(self, concept_a: str, concept_b: str) -> float:
        """Return cosine similarity between two concepts."""
        if concept_a not in self.names or concept_b not in self.names:
            return 0.0
        if self._matrix is None:
            self.build()
        i = self.names.index(concept_a)
        j = self.names.index(concept_b)
        return self._matrix[i, j].item()

    def print_report(self, sample_concepts: Optional[List[str]] = None):
        """Print a human-readable cluster report."""
        if sample_concepts is None:
            # Pick a few interesting ones
            sample_concepts = self.names[:10]

        print(f"\n{'='*60}")
        print(f"Semantic Cluster Report ({len(self.names)} concepts)")
        print(f"{'='*60}")

        for concept in sample_concepts:
            if concept not in self.names:
                continue
            neighbors = self.nearest(concept, top_k=5)
            neighbor_str = ", ".join(f"{n}({s:.2f})" for n, s in neighbors)
            print(f"  {concept:20s} → {neighbor_str}")

        print(f"\nTop semantic clusters (threshold=0.65):")
        clusters = self.clusters(threshold=0.65)
        for i, group in enumerate(clusters[:10]):
            print(f"  [{i+1}] {', '.join(group[:8])}{'...' if len(group) > 8 else ''}")

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        if self._matrix is None:
            self.build()
        # Store top-10 nearest neighbors per concept (not full matrix — too large)
        neighbors_map = {}
        for concept in self.names:
            neighbors_map[concept] = [
                {"concept": n, "similarity": s}
                for n, s in self.nearest(concept, top_k=10)
            ]
        return {
            "concept_count": len(self.names),
            "concepts": self.names,
            "neighbors": neighbors_map,
            "clusters": self.clusters(threshold=0.65),
        }
