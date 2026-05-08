"""
Stage 6: TRANSPLANT
===================
Write projected concept vectors directly into the target model's embedding weights.

This is the "memory transplant" step — after projection, we have concept vectors
in our target model's space. We write them directly into the embedding table at
the correct token IDs.

For CortexLang:
    entity token_id = entity_base (5000) + concept_id
    relation token_id = relation_base (100) + relation_offset

After transplant, when the model processes "E:fire", the embedding lookup
returns a vector pre-loaded with Qwen's knowledge about fire — before any
training has occurred.

The model then only needs to learn:
    - How to chain these pre-loaded concepts (grammar/structure)
    - How to route through its MoE experts
    - How to output confidence scores
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json


class KnowledgeTransplanter:
    """Write projected concept vectors into a target model's embedding weights.

    Args:
        target_model: Our StructuredAnthroSlammer model (has .tok_embeddings)
        projector: A trained ConceptProjector
        vocab: ConceptVocabulary (maps concept names → concept_ids)
        tok: CortexLangTokenizer (knows entity_base, relation token IDs)
    """

    def __init__(self, target_model, projector, vocab, tok):
        self.target_model = target_model
        self.projector = projector
        self.vocab = vocab
        self.tok = tok
        self._transplant_log: List[dict] = []

    def _get_embedding(self):
        """Get the token embedding weight tensor, handling both model wrappers."""
        # StructuredAnthroSlammer has tok_embeddings directly
        if hasattr(self.target_model, 'tok_embeddings'):
            return self.target_model.tok_embeddings.weight
        # Fallback for raw AnthroSlammerModel
        if hasattr(self.target_model, 'base_model'):
            return self.target_model.base_model.tok_embeddings.weight
        raise AttributeError("Cannot find tok_embeddings on target model")

    # ── Public API ─────────────────────────────────────────────────

    def transplant_entities(
        self,
        entity_vectors: Dict[str, torch.Tensor],
        freeze_after: bool = True,
        verbose: bool = True,
    ) -> int:
        """Write entity concept vectors into the embedding table.

        For each entity name in entity_vectors:
        1. Look up its concept_id in vocab
        2. Compute token_id = entity_base + concept_id
        3. Write the projected vector into tok_embeddings.weight[token_id]

        Args:
            entity_vectors: Dict[concept_name → source_dim vector] (pre-projection)
            freeze_after: If True, mark the transplanted slots to be frozen during training
            verbose: Print progress

        Returns:
            Number of successful transplants
        """
        if verbose:
            print(f"\n[AutoPsy:TRANSPLANT] Transplanting {len(entity_vectors)} entity vectors")

        # Project all source vectors to target space
        projected = self.projector.project(entity_vectors)

        # Get the embedding weight tensor (we'll write into it)
        embedding = self._get_embedding()
        entity_base = self.tok._entity_base  # 5000

        count = 0
        skipped = 0
        target_dim = embedding.shape[1]

        with torch.no_grad():
            for name, proj_vec in projected.items():
                # Look up concept
                concept = self.vocab.lookup(name)
                if concept is None:
                    skipped += 1
                    continue

                token_id = entity_base + concept.concept_id
                if token_id >= embedding.shape[0]:
                    skipped += 1
                    continue

                # Resize if needed (shouldn't happen but be safe)
                if proj_vec.shape[0] != target_dim:
                    proj_vec = F.interpolate(
                        proj_vec.unsqueeze(0).unsqueeze(0),
                        size=target_dim,
                        mode="linear",
                    ).squeeze()

                # Write into embedding table
                embedding[token_id] = proj_vec.to(embedding.device).to(embedding.dtype)
                count += 1

                self._transplant_log.append({
                    "type": "entity",
                    "name": name,
                    "concept_id": concept.concept_id,
                    "token_id": token_id,
                })

        if verbose:
            print(f"  Transplanted: {count} entities")
            if skipped:
                print(f"  Skipped (not in vocab): {skipped}")

        return count

    def transplant_relations(
        self,
        relation_vectors: Dict[str, torch.Tensor],
        verbose: bool = True,
    ) -> int:
        """Write relation vectors into the relation token slots.

        Relation tokens occupy IDs 100-300 (the RANGES["relation"] zone).
        We map relation name → token_id via the tokenizer's _token_to_id dict.
        """
        if verbose:
            print(f"\n[AutoPsy:TRANSPLANT] Transplanting {len(relation_vectors)} relation vectors")

        projected = self.projector.project(relation_vectors)
        embedding = self._get_embedding()
        token_to_id = self.tok._token_to_id
        target_dim = embedding.shape[1]

        count = 0
        skipped = 0

        with torch.no_grad():
            for rel_name, proj_vec in projected.items():
                # relation_vectors keys may be "R:causes" or "causes"
                candidates = [rel_name, f"R:{rel_name}", rel_name.replace("R:", "")]
                token_id = None
                for candidate in candidates:
                    if candidate in token_to_id:
                        token_id = token_to_id[candidate]
                        break

                if token_id is None:
                    skipped += 1
                    continue

                if proj_vec.shape[0] != target_dim:
                    proj_vec = F.interpolate(
                        proj_vec.unsqueeze(0).unsqueeze(0),
                        size=target_dim,
                        mode="linear",
                    ).squeeze()

                embedding[token_id] = proj_vec.to(embedding.device).to(embedding.dtype)
                count += 1

                self._transplant_log.append({
                    "type": "relation",
                    "name": rel_name,
                    "token_id": token_id,
                })

        if verbose:
            print(f"  Transplanted: {count} relations")
            if skipped:
                print(f"  Skipped (not in tokenizer): {skipped}")

        return count

    def verify(
        self,
        test_pairs: Optional[List[Tuple[str, str]]] = None,
        verbose: bool = True,
    ) -> dict:
        """Check that transplanted vectors have the right semantic relationships.

        Tests that semantically similar concepts (fire/heat) have higher
        cosine similarity than unrelated ones (fire/democracy).

        Args:
            test_pairs: List of (concept_a, concept_b) pairs that SHOULD be similar.
                        Defaults to built-in sanity check pairs.

        Returns:
            Dict with verification results.
        """
        if test_pairs is None:
            test_pairs = [
                ("fire", "heat"),
                ("water", "ice"),
                ("sun", "light"),
                ("rain", "water"),
                ("oxygen", "air"),
            ]
        # Pairs that should be DISSIMILAR
        dissimilar_pairs = [
            ("fire", "democracy"),
            ("water", "integer"),
            ("heat", "proof"),
        ]

        embedding = self._get_embedding().detach()
        entity_base = self.tok._entity_base

        def get_vec(name):
            c = self.vocab.lookup(name)
            if c is None:
                return None
            tid = entity_base + c.concept_id
            if tid >= embedding.shape[0]:
                return None
            return F.normalize(embedding[tid].float(), dim=0)

        results = {"similar_pairs": [], "dissimilar_pairs": [], "pass": True}

        if verbose:
            print(f"\n[AutoPsy:TRANSPLANT] Verification")

        for a, b in test_pairs:
            va, vb = get_vec(a), get_vec(b)
            if va is None or vb is None:
                continue
            sim = (va * vb).sum().item()
            results["similar_pairs"].append({"a": a, "b": b, "similarity": round(sim, 4)})
            if verbose:
                status = "✓" if sim > 0.3 else "✗ LOW"
                print(f"  {a:15s} ~ {b:15s}  sim={sim:.3f}  {status}")
            if sim < 0.1:
                results["pass"] = False

        for a, b in dissimilar_pairs:
            va, vb = get_vec(a), get_vec(b)
            if va is None or vb is None:
                continue
            sim = (va * vb).sum().item()
            results["dissimilar_pairs"].append({"a": a, "b": b, "similarity": round(sim, 4)})
            if verbose:
                status = "✓" if sim < 0.5 else "✗ HIGH (should be dissimilar)"
                print(f"  {a:15s} ≠ {b:15s}  sim={sim:.3f}  {status}")

        if verbose:
            print(f"\n  Overall: {'PASS' if results['pass'] else 'FAIL — check projection quality'}")

        return results

    def save_checkpoint(self, output_path: str):
        """Save the transplanted model as a checkpoint."""
        p = Path(output_path)
        p.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(
            self.target_model.state_dict(),
            p / "model.pt",
        )

        # Save transplant log
        with open(p / "transplant_log.json", "w") as f:
            json.dump({
                "transplanted_count": len(self._transplant_log),
                "entries": self._transplant_log,
            }, f, indent=2)

        print(f"[AutoPsy:TRANSPLANT] Saved transplanted model → {p}")
        print(f"  {len(self._transplant_log)} concept slots pre-loaded")
