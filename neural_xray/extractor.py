"""
Stage 3: EXTRACT
================
Pull concept vectors from a model's middle MLP layers using activation hooks.

For each concept (e.g. "fire", "water", "causes"), we:
1. Build a short probe prompt: "fire is a type of"
2. Run a forward pass through the model
3. Hook into the MLP output at each "fact zone" layer
4. Average the hidden states across the fact zone layers
5. Save the result as a single concept vector

This gives us a vector that encodes what the model "thinks" about that concept —
its position in the model's internal knowledge geometry.

The key insight (Geva et al 2021): MLP layers act as key-value memories.
The middle layers store factual associations. The output of the MLP at those
layers, when prompted with an entity name, captures the entity's semantic meaning.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager


class ConceptExtractor:
    """Extract concept vectors from a model's middle layers via activation hooks.

    For each concept word, runs a forward pass and captures MLP outputs
    across the fact-storage zone, averaging them into a single dense vector.

    Args:
        loader: A loaded ModelLoader instance
        arch_map: An ArchitectureMap from ArchitectureMapper
    """

    # Prompt templates for extracting concept vectors
    # Multiple templates are averaged to get a more robust representation
    _ENTITY_TEMPLATES = [
        "{concept} is",
        "{concept} refers to",
        "The concept of {concept}",
        "{concept} can be described as",
    ]

    # Neutral baselines — same template structure, neutral word
    # Subtracting these removes the shared template direction (CAA method)
    _BASELINE_TEMPLATES = [
        "something is",
        "something refers to",
        "The concept of something",
        "something can be described as",
    ]

    _RELATION_TEMPLATES = [
        "X {relation} Y means",
        "when something {relation} something else",
        "the relationship {relation} describes",
    ]

    _RELATION_BASELINE_TEMPLATES = [
        "X does Y means",
        "when something does something else",
        "the relationship does describes",
    ]

    def __init__(self, loader, arch_map):
        self.loader = loader
        self.arch_map = arch_map
        self._hooks = []
        self._captured: Dict[int, List[torch.Tensor]] = {}

    # ── Public API ─────────────────────────────────────────────────

    def extract_entities(
        self,
        concepts: List[str],
        batch_size: int = 8,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Extract concept vectors for a list of entity names.

        Args:
            concepts: List of entity names e.g. ["fire", "water", "heat"]
            batch_size: How many concepts to process at once
            verbose: Print progress

        Returns:
            Dict mapping concept_name → vector [hidden_size]
        """
        if verbose:
            print(f"\n[AutoPsy:EXTRACT] Extracting {len(concepts)} entity vectors")
            print(f"  Fact zone: layers {self.arch_map.fact_layer_start}–{self.arch_map.fact_layer_end}")

        results = {}
        total = len(concepts)

        for i in range(0, total, batch_size):
            batch = concepts[i:i + batch_size]
            for concept in batch:
                vec = self._extract_single(concept, self._ENTITY_TEMPLATES)
                results[concept] = vec
            if verbose:
                done = min(i + batch_size, total)
                print(f"  {done}/{total} extracted", end="\r")

        if verbose:
            print(f"  {total}/{total} extracted — done")
        return results

    def extract_relations(
        self,
        relations: List[str],
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Extract vectors for relation types (causes, produces, etc.)"""
        if verbose:
            print(f"\n[AutoPsy:EXTRACT] Extracting {len(relations)} relation vectors")

        results = {}
        for rel in relations:
            # Clean: "R:causes" → "causes"
            clean = rel.replace("R:", "").replace("_", " ")
            vec = self._extract_single(clean, self._RELATION_TEMPLATES)
            results[rel] = vec
            if verbose:
                print(f"  {rel} → {vec.shape}", end="\r")

        if verbose:
            print(f"  {len(relations)} relations extracted — done")
        return results

    # ── Contrastive extraction (CAA) ──────────────────────────────

    def extract_entities_contrastive(
        self,
        concepts: List[str],
        batch_size: int = 8,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Extract concept vectors using Contrastive Activation Addition.

        For each concept, we compute:
            vec = mean(template_activations) - mean(baseline_activations)

        This removes the shared template direction that causes "happy dominance"
        where all concepts look similar due to template structure, not meaning.

        Reference: Rimsky et al. 2023 (arXiv:2312.06681)
        """
        if verbose:
            print(f"\n[AutoPsy:EXTRACT] Contrastive extracting {len(concepts)} entities (CAA)")
            print(f"  Fact zone: layers {self.arch_map.fact_layer_start}–{self.arch_map.fact_layer_end}")

        # Pre-compute baseline (same for all concepts — "something" templates)
        baseline_vec = self._compute_baseline(self._BASELINE_TEMPLATES, verbose)

        results = {}
        total = len(concepts)

        for i in range(0, total, batch_size):
            batch = concepts[i:i + batch_size]
            for concept in batch:
                concept_vec = self._extract_single(concept, self._ENTITY_TEMPLATES)
                # Subtract baseline to isolate concept-specific direction
                results[concept] = concept_vec - baseline_vec
            if verbose:
                done = min(i + batch_size, total)
                print(f"  {done}/{total} extracted", end="\r")

        if verbose:
            print(f"  {total}/{total} extracted — done (contrastive)")
        return results

    def extract_relations_contrastive(
        self,
        relations: List[str],
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Extract contrastive vectors for relation types."""
        if verbose:
            print(f"\n[AutoPsy:EXTRACT] Contrastive extracting {len(relations)} relations (CAA)")

        baseline_vec = self._compute_baseline(self._RELATION_BASELINE_TEMPLATES, verbose)

        results = {}
        for rel in relations:
            clean = rel.replace("R:", "").replace("_", " ")
            concept_vec = self._extract_single(clean, self._RELATION_TEMPLATES)
            results[rel] = concept_vec - baseline_vec
            if verbose:
                print(f"  {rel} → {concept_vec.shape}", end="\r")

        if verbose:
            print(f"  {len(relations)} relations extracted — done (contrastive)")
        return results

    def _compute_baseline(
        self,
        baseline_templates: List[str],
        verbose: bool = False,
    ) -> torch.Tensor:
        """Compute the average baseline activation across neutral templates.

        This is the shared template direction that we subtract out.
        """
        all_vecs = []
        for template in baseline_templates:
            # Baseline templates don't have {concept} — they use "something" directly
            vec = self._forward_and_capture(template)
            if vec is not None:
                all_vecs.append(vec)

        if not all_vecs:
            if verbose:
                print("  WARNING: baseline capture failed, returning zeros")
            return torch.zeros(self.arch_map.hidden_size)

        baseline = torch.stack(all_vecs).mean(dim=0)
        if verbose:
            print(f"  Baseline computed (norm={baseline.norm():.2f})")
        return baseline

    def save(self, vectors: Dict[str, torch.Tensor], path: str):
        """Save extracted vectors to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "vectors": {k: v.cpu() for k, v in vectors.items()},
            "hidden_size": self.arch_map.hidden_size,
            "model_name": self.arch_map.model_name,
            "fact_layer_start": self.arch_map.fact_layer_start,
            "fact_layer_end": self.arch_map.fact_layer_end,
        }, str(p))
        print(f"[AutoPsy:EXTRACT] Saved {len(vectors)} vectors → {p}")

    @staticmethod
    def load(path: str) -> Dict[str, torch.Tensor]:
        """Load previously extracted vectors."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[AutoPsy:EXTRACT] Loaded {len(data['vectors'])} vectors from {path}")
        print(f"  Source model: {data.get('model_name', '?')}")
        print(f"  Hidden size: {data.get('hidden_size', '?')}")
        return data["vectors"]

    # ── Internal ───────────────────────────────────────────────────

    def _extract_single(
        self,
        concept: str,
        templates: List[str],
    ) -> torch.Tensor:
        """Extract and average concept vectors across multiple prompt templates."""
        all_vecs = []

        for template in templates:
            prompt = template.format(concept=concept, relation=concept)
            vec = self._forward_and_capture(prompt)
            if vec is not None:
                all_vecs.append(vec)

        if not all_vecs:
            # Fallback: zero vector
            return torch.zeros(self.arch_map.hidden_size)

        # Average across templates for robustness
        stacked = torch.stack(all_vecs, dim=0)  # [n_templates, hidden]
        return stacked.mean(dim=0)

    def _forward_and_capture(self, prompt: str) -> Optional[torch.Tensor]:
        """Run one forward pass and return the averaged MLP hidden state
        from the fact-storage zone layers."""
        model = self.loader.model
        tokenizer = self.loader.tokenizer

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Capture hidden states via output_hidden_states=True
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple: [embed, layer0, layer1, ..., layerN]
        # Index offset: hidden_states[i+1] = output of layer i
        hidden_states = outputs.hidden_states  # tuple of [1, seq_len, hidden]

        fact_layers = list(range(
            self.arch_map.fact_layer_start,
            self.arch_map.fact_layer_end + 1,
        ))

        collected = []
        for layer_idx in fact_layers:
            hs_idx = layer_idx + 1  # +1 because index 0 is embedding layer output
            if hs_idx < len(hidden_states):
                # Take the last token position (the model's "answer" position)
                h = hidden_states[hs_idx][0, -1, :].float().cpu()
                collected.append(h)

        if not collected:
            return None

        # Average across fact-zone layers
        return torch.stack(collected).mean(dim=0)

    @contextmanager
    def _hook_mlp_outputs(self, fact_layer_indices: List[int]):
        """Context manager that installs forward hooks on MLP output layers."""
        self._captured = {i: [] for i in fact_layer_indices}
        hooks = []

        for layer_info in self.arch_map.fact_layers():
            if layer_info.mlp_path is None:
                continue
            idx = layer_info.index
            mlp_mod = self._get_module(layer_info.mlp_path)
            if mlp_mod is None:
                continue

            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output may be a tensor or tuple
                    h = output[0] if isinstance(output, tuple) else output
                    self._captured[layer_idx].append(h.detach().cpu())
                return hook

            h = mlp_mod.register_forward_hook(make_hook(idx))
            hooks.append(h)

        try:
            yield
        finally:
            for h in hooks:
                h.remove()
            self._captured = {}

    def _get_module(self, path: str) -> Optional[nn.Module]:
        """Traverse model by dot-separated path."""
        mod = self.loader.model
        for part in path.split("."):
            if part.isdigit():
                try:
                    mod = mod[int(part)]
                except (IndexError, TypeError):
                    return None
            else:
                mod = getattr(mod, part, None)
                if mod is None:
                    return None
        return mod
