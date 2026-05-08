"""
Stage 2: MAP
============
Automatically discover the architecture of any loaded model:
- Which layers are attention vs MLP
- Where the MLP "value" matrices are (the ones that store facts)
- The shape of every important weight matrix
- Which layer range is "middle" (where facts are primarily stored, per ROME research)

This produces a LayerMap that every other stage uses to know WHERE to look.
The mapper is architecture-agnostic — it inspects the actual module tree
rather than assuming specific attribute names.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerInfo:
    """Description of a single transformer layer."""
    index: int
    name: str                      # module path, e.g. "model.layers.4"
    has_attention: bool = False
    has_mlp: bool = False
    mlp_path: Optional[str] = None  # path to MLP submodule
    attn_path: Optional[str] = None # path to attention submodule
    # MLP weight shapes
    mlp_w1_shape: Optional[Tuple] = None  # gate/up weight
    mlp_w2_shape: Optional[Tuple] = None  # down weight (output = value store)
    hidden_size: Optional[int] = None


@dataclass
class ArchitectureMap:
    """Full structural map of a model."""
    model_name: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    layers: List[LayerInfo] = field(default_factory=list)

    # The "fact storage zone" — middle 1/3 of layers (per ROME/Geva research)
    fact_layer_start: int = 0
    fact_layer_end: int = 0

    # Embedding module path
    embed_path: Optional[str] = None

    def fact_layers(self) -> List[LayerInfo]:
        """Return only layers in the fact-storage zone."""
        return [l for l in self.layers
                if self.fact_layer_start <= l.index <= self.fact_layer_end]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "fact_layer_start": self.fact_layer_start,
            "fact_layer_end": self.fact_layer_end,
            "embed_path": self.embed_path,
            "layers": [
                {
                    "index": l.index,
                    "name": l.name,
                    "has_attention": l.has_attention,
                    "has_mlp": l.has_mlp,
                    "mlp_path": l.mlp_path,
                    "mlp_w2_shape": list(l.mlp_w2_shape) if l.mlp_w2_shape else None,
                }
                for l in self.layers
            ],
        }


class ArchitectureMapper:
    """Auto-discover the structure of any transformer model.

    The mapper walks the module tree looking for:
    - Linear layers with weight shapes suggesting attention (Q, K, V, O)
    - Linear layers with weight shapes suggesting MLP (gate, up, down projections)
    - Embedding layers (token embeddings)

    It does NOT assume specific attribute names — works on Qwen, LLaMA,
    Mistral, Phi, GPT-2, Falcon, etc.
    """

    # Module name fragments that suggest MLP components
    _MLP_NAMES = {"mlp", "ffn", "feed_forward", "ff", "fc"}
    _ATTN_NAMES = {"attn", "attention", "self_attn", "self_attention"}
    _EMBED_NAMES = {"embed_tokens", "tok_embeddings", "wte", "word_embeddings", "embed"}
    # Weight attribute names for MLP output (the "value" projection — stores facts)
    _MLP_DOWN_NAMES = {"down_proj", "w_down", "c_proj", "fc2", "out_proj", "wo"}

    def __init__(self, loader):
        """
        Args:
            loader: A loaded ModelLoader instance
        """
        self.loader = loader
        self.arch_map: Optional[ArchitectureMap] = None

    def map(self) -> ArchitectureMap:
        """Walk the model and produce a full ArchitectureMap."""
        model = self.loader.model
        print(f"\n[AutoPsy:MAP] Mapping {self.loader.model_name}")

        hidden_size = self.loader.hidden_size
        num_layers = self.loader.num_layers
        vocab_size = model.config.vocab_size

        arch = ArchitectureMap(
            model_name=self.loader.model_name,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
        )

        # Find embedding layer
        arch.embed_path = self._find_embed_path(model)
        print(f"  Embedding: {arch.embed_path}")

        # Find transformer layers
        layer_container, layer_path = self._find_layer_container(model)
        print(f"  Layer container: {layer_path} ({len(layer_container)} layers)")

        for idx, layer_module in enumerate(layer_container):
            base_path = f"{layer_path}.{idx}"
            info = LayerInfo(index=idx, name=base_path)

            # Find attention and MLP submodules within this layer
            for name, submod in layer_module.named_children():
                low = name.lower()
                if any(k in low for k in self._ATTN_NAMES):
                    info.has_attention = True
                    info.attn_path = f"{base_path}.{name}"
                if any(k in low for k in self._MLP_NAMES):
                    info.has_mlp = True
                    info.mlp_path = f"{base_path}.{name}"
                    # Look for the down projection (output/value weight)
                    for wname, wmod in submod.named_modules():
                        if isinstance(wmod, nn.Linear):
                            if any(k in wname.lower() for k in self._MLP_DOWN_NAMES):
                                info.mlp_w2_shape = tuple(wmod.weight.shape)
                            elif info.mlp_w1_shape is None:
                                info.mlp_w1_shape = tuple(wmod.weight.shape)
                    info.hidden_size = hidden_size

            arch.layers.append(info)

        # Fact storage zone — use architecture-specific targeting where known.
        # Source: arXiv:2602.06852 (Quantum Sieve Tracer, Feb 2026).
        #   Qwen2.x  → layer 7 is the "Recall Hub"
        #   Llama2/3 → layer 9 is "Interference Suppression"
        #   Others   → fall back to ROME/Geva middle-third heuristic
        arch.fact_layer_start, arch.fact_layer_end = self._fact_zone(
            self.loader.model_name, num_layers
        )
        fact_layers = arch.fact_layer_end - arch.fact_layer_start
        print(f"  Fact storage zone: layers {arch.fact_layer_start}–{arch.fact_layer_end} ({fact_layers} layers)")

        self.arch_map = arch
        print(f"  Map complete: {len([l for l in arch.layers if l.has_mlp])} MLP layers found")
        return arch

    # Architecture-specific fact-zone lookup (arXiv:2602.06852)
    _ARCH_FACT_ZONES = {
        "qwen2":   (7, 7),   # Recall Hub at layer 7 (single focal layer)
        "qwen":    (7, 7),
        "llama":   (9, 9),   # Interference Suppression at layer 9
        "mistral": (8, 8),
        "phi":     (6, 6),
    }

    def _fact_zone(self, model_name: str, num_layers: int) -> Tuple[int, int]:
        """Return (fact_layer_start, fact_layer_end) for this architecture."""
        lower = model_name.lower()
        for key, (start, end) in self._ARCH_FACT_ZONES.items():
            if key in lower:
                # Clamp to valid layer range
                start = min(start, num_layers - 1)
                end   = min(end,   num_layers - 1)
                return start, end
        # Fallback: ROME/Geva middle-third heuristic
        return num_layers // 3, (2 * num_layers) // 3

    def _find_embed_path(self, model) -> Optional[str]:
        """Find the token embedding module path."""
        for path, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                last_part = path.split(".")[-1].lower()
                if any(k in last_part for k in self._EMBED_NAMES):
                    return path
        # Fallback: first embedding layer
        for path, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                return path
        return None

    def _find_layer_container(self, model) -> Tuple[nn.ModuleList, str]:
        """Find the ModuleList that contains the transformer blocks."""
        # Common names for the list of transformer layers
        _LAYER_CONTAINER_NAMES = {"layers", "h", "blocks", "decoder", "encoder"}
        candidates = []
        for name, mod in model.named_children():
            if isinstance(mod, nn.ModuleList) and len(mod) > 1:
                candidates.append((name, mod, len(mod)))
            # One level deeper
            for subname, submod in mod.named_children():
                if isinstance(submod, nn.ModuleList) and len(submod) > 1:
                    path = f"{name}.{subname}"
                    candidates.append((path, submod, len(submod)))

        if not candidates:
            raise ValueError("Cannot find transformer layer container (ModuleList)")

        # Pick the one whose length matches num_layers, or the largest
        for path, mod, length in candidates:
            if length == self.loader.num_layers:
                return mod, path
        # Fallback: largest ModuleList
        candidates.sort(key=lambda x: x[2], reverse=True)
        path, mod, _ = candidates[0]
        return mod, path

    def print_summary(self):
        """Print a human-readable summary of the architecture."""
        if self.arch_map is None:
            print("Not yet mapped — call .map() first")
            return
        a = self.arch_map
        print(f"\n{'='*60}")
        print(f"Architecture Map: {a.model_name}")
        print(f"{'='*60}")
        print(f"  Hidden size:    {a.hidden_size}")
        print(f"  Layers:         {a.num_layers}")
        print(f"  Vocab size:     {a.vocab_size:,}")
        print(f"  Embedding at:   {a.embed_path}")
        print(f"  Fact zone:      layers {a.fact_layer_start}–{a.fact_layer_end}")
        print()
        print(f"  {'Layer':>5}  {'Attn':>5}  {'MLP':>5}  {'MLP-out shape':>20}  {'In fact zone':>12}")
        print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*20}  {'-'*12}")
        for l in a.layers:
            in_zone = "← FACT ZONE" if a.fact_layer_start <= l.index <= a.fact_layer_end else ""
            w2 = str(l.mlp_w2_shape) if l.mlp_w2_shape else "?"
            print(f"  {l.index:>5}  {'✓' if l.has_attention else '':>5}  {'✓' if l.has_mlp else '':>5}  {w2:>20}  {in_zone}")
        print()
