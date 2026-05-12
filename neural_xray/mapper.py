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
    """Description of a single transformer layer — every sub-component exposed."""
    index: int
    name: str                      # module path, e.g. "model.layers.4"
    has_attention: bool = False
    has_mlp: bool = False
    mlp_path: Optional[str] = None  # path to MLP submodule
    attn_path: Optional[str] = None # path to attention submodule
    # Legacy MLP weight shapes (kept for backwards compat)
    mlp_w1_shape: Optional[Tuple] = None  # gate/up weight
    mlp_w2_shape: Optional[Tuple] = None  # down weight (output = value store)
    hidden_size: Optional[int] = None

    # ── Attention sub-components ──
    num_heads: Optional[int] = None          # total attention heads
    num_kv_heads: Optional[int] = None       # key/value heads (GQA < num_heads)
    head_dim: Optional[int] = None           # dimension per head
    q_proj_shape: Optional[Tuple] = None     # Q projection weight shape
    k_proj_shape: Optional[Tuple] = None     # K projection weight shape
    v_proj_shape: Optional[Tuple] = None     # V projection weight shape
    o_proj_shape: Optional[Tuple] = None     # output projection weight shape
    attn_has_bias: bool = False              # whether projections have bias

    # ── MLP sub-components (separated) ──
    gate_proj_shape: Optional[Tuple] = None  # gate projection (SwiGLU models)
    up_proj_shape: Optional[Tuple] = None    # up/expand projection
    down_proj_shape: Optional[Tuple] = None  # down/contract projection
    activation_fn: Optional[str] = None      # "silu", "gelu", "relu", "gelu_new", etc.

    # ── Normalisation layers ──
    pre_attn_norm_path: Optional[str] = None   # LayerNorm before attention
    post_attn_norm_path: Optional[str] = None  # LayerNorm after attention (if present)
    pre_mlp_norm_path: Optional[str] = None    # LayerNorm before MLP
    norm_type: Optional[str] = None            # "layer_norm" | "rms_norm"

    # ── Full flat module list for this layer (Netron-style) ──
    # List of {name, type, shape} for every sub-module with weights
    module_nodes: List[dict] = field(default_factory=list)


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
                    "attn_path": l.attn_path,
                    "mlp_path": l.mlp_path,
                    # Legacy shapes (backwards compat)
                    "mlp_w1_shape": list(l.mlp_w1_shape) if l.mlp_w1_shape else None,
                    "mlp_w2_shape": list(l.mlp_w2_shape) if l.mlp_w2_shape else None,
                    "hidden_size": l.hidden_size,
                    # Attention detail
                    "num_heads": l.num_heads,
                    "num_kv_heads": l.num_kv_heads,
                    "head_dim": l.head_dim,
                    "q_proj_shape": list(l.q_proj_shape) if l.q_proj_shape else None,
                    "k_proj_shape": list(l.k_proj_shape) if l.k_proj_shape else None,
                    "v_proj_shape": list(l.v_proj_shape) if l.v_proj_shape else None,
                    "o_proj_shape": list(l.o_proj_shape) if l.o_proj_shape else None,
                    "attn_has_bias": l.attn_has_bias,
                    # MLP detail
                    "gate_proj_shape": list(l.gate_proj_shape) if l.gate_proj_shape else None,
                    "up_proj_shape": list(l.up_proj_shape) if l.up_proj_shape else None,
                    "down_proj_shape": list(l.down_proj_shape) if l.down_proj_shape else None,
                    "activation_fn": l.activation_fn,
                    # Norms
                    "pre_attn_norm_path": l.pre_attn_norm_path,
                    "post_attn_norm_path": l.post_attn_norm_path,
                    "pre_mlp_norm_path": l.pre_mlp_norm_path,
                    "norm_type": l.norm_type,
                    # Full module graph
                    "module_nodes": l.module_nodes,
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

    # ── Attention projection name sets ──
    _Q_NAMES   = {"q_proj", "query", "q", "wq", "query_key_value"}
    _K_NAMES   = {"k_proj", "key", "k", "wk"}
    _V_NAMES   = {"v_proj", "value", "v", "wv"}
    _O_NAMES   = {"o_proj", "out_proj", "dense", "wo", "c_proj"}

    # ── MLP projection name sets ──
    _GATE_NAMES = {"gate_proj", "w_gate", "w1"}
    _UP_NAMES   = {"up_proj", "w_up", "w3", "fc1", "c_fc", "dense_h_to_4h"}
    _DOWN_NAMES = {"down_proj", "w_down", "c_proj", "fc2", "out_proj", "wo", "dense_4h_to_h"}

    # ── LayerNorm name sets ──
    _PRE_ATTN_NORM_NAMES  = {"input_layernorm", "ln_1", "norm1", "attention_norm", "pre_attention_layernorm"}
    _POST_ATTN_NORM_NAMES = {"post_attention_layernorm", "ln_2", "norm2", "ffn_norm", "post_feedforward_layernorm"}
    _PRE_MLP_NORM_NAMES   = {"post_attention_layernorm", "ln_2", "norm2", "ffn_norm"}

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

            # ── Pass 1: find top-level attention / MLP / norm submodules ──
            for name, submod in layer_module.named_children():
                low = name.lower()
                if any(k in low for k in self._ATTN_NAMES):
                    info.has_attention = True
                    info.attn_path = f"{base_path}.{name}"
                    self._scan_attention(submod, info, hidden_size, model)
                elif any(k in low for k in self._MLP_NAMES):
                    info.has_mlp = True
                    info.mlp_path = f"{base_path}.{name}"
                    self._scan_mlp(submod, info, model)
                else:
                    self._scan_norm(name, submod, base_path, info)

            # ── Pass 2: build flat module_nodes list (Netron-style) ──
            info.module_nodes = self._build_module_nodes(layer_module, base_path)

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

    # ──────────────────────────────────────────────────────────────
    # Deep scan helpers
    # ──────────────────────────────────────────────────────────────

    def _scan_attention(self, attn_mod: nn.Module, info: 'LayerInfo', hidden_size: int, model) -> None:
        """Extract Q/K/V/O projection shapes and head counts from an attention submodule."""
        cfg = getattr(model, 'config', None)

        # Pull head counts from config
        if cfg:
            info.num_heads = getattr(cfg, 'num_attention_heads',
                             getattr(cfg, 'n_head',
                             getattr(cfg, 'num_heads', None)))
            info.num_kv_heads = getattr(cfg, 'num_key_value_heads',
                                 getattr(cfg, 'num_kv_heads', info.num_heads))
            if info.num_heads:
                info.head_dim = hidden_size // info.num_heads

        # Check for fused QKV (e.g. GPT-2's c_attn: 768 → 2304)
        fused_qkv = None
        for wname, wmod in attn_mod.named_modules():
            if not isinstance(wmod, nn.Linear):
                continue
            low = wname.lower().split('.')[-1]
            w_shape = tuple(wmod.weight.shape)
            info.attn_has_bias = wmod.bias is not None

            # Fused QKV (GPT-2 style: output is 3×hidden or 3×kv_dim)
            if low in {'c_attn', 'qkv', 'query_key_value', 'Wqkv'} or (
                len(w_shape) == 2 and w_shape[0] in {hidden_size * 3, hidden_size + 2 * (hidden_size // (info.num_heads or 1) * (info.num_kv_heads or info.num_heads or 1))}
            ):
                fused_qkv = w_shape
                # Approximate split for display
                out_dim = w_shape[0]
                kv_dim = (out_dim - hidden_size) // 2 if out_dim > hidden_size else hidden_size
                q_dim = out_dim - 2 * kv_dim
                info.q_proj_shape = (q_dim, w_shape[1])
                info.k_proj_shape = (kv_dim, w_shape[1])
                info.v_proj_shape = (kv_dim, w_shape[1])
                continue

            if low in self._Q_NAMES and not fused_qkv:
                info.q_proj_shape = w_shape
            elif low in self._K_NAMES:
                info.k_proj_shape = w_shape
            elif low in self._V_NAMES:
                info.v_proj_shape = w_shape
            elif low in self._O_NAMES:
                info.o_proj_shape = w_shape

        # Fallback: if we only found shapes by position (no matching names)
        if not info.q_proj_shape and not fused_qkv:
            linears = [(n, m) for n, m in attn_mod.named_modules() if isinstance(m, nn.Linear)]
            if len(linears) >= 4:
                info.q_proj_shape = tuple(linears[0][1].weight.shape)
                info.k_proj_shape = tuple(linears[1][1].weight.shape)
                info.v_proj_shape = tuple(linears[2][1].weight.shape)
                info.o_proj_shape = tuple(linears[3][1].weight.shape)
            elif len(linears) == 1 and fused_qkv is None:
                # Single fused matrix
                info.q_proj_shape = tuple(linears[0][1].weight.shape)

    def _scan_mlp(self, mlp_mod: nn.Module, info: 'LayerInfo', model) -> None:
        """Extract gate/up/down projection shapes and activation function."""
        cfg = getattr(model, 'config', None)

        # Get activation from config
        if cfg:
            act = getattr(cfg, 'hidden_act',
                  getattr(cfg, 'activation_function',
                  getattr(cfg, 'hidden_activation', None)))
            if act:
                info.activation_fn = str(act)

        linears_by_name: dict = {}
        for wname, wmod in mlp_mod.named_modules():
            if not isinstance(wmod, nn.Linear):
                continue
            low = wname.lower().split('.')[-1]
            shape = tuple(wmod.weight.shape)
            linears_by_name[low] = shape

            if low in self._GATE_NAMES:
                info.gate_proj_shape = shape
            elif low in self._DOWN_NAMES:
                info.down_proj_shape = shape
                info.mlp_w2_shape = shape  # legacy compat
            elif low in self._UP_NAMES:
                info.up_proj_shape = shape
                if info.mlp_w1_shape is None:
                    info.mlp_w1_shape = shape

        # Fallback: positional assignment for unnamed linears
        if not info.up_proj_shape and not info.gate_proj_shape:
            linears = [(n, m) for n, m in mlp_mod.named_modules() if isinstance(m, nn.Linear)]
            if len(linears) == 2:
                # fc1 → fc2 (BERT / GPT-2 style)
                info.up_proj_shape = tuple(linears[0][1].weight.shape)
                info.mlp_w1_shape  = info.up_proj_shape
                info.down_proj_shape = tuple(linears[1][1].weight.shape)
                info.mlp_w2_shape    = info.down_proj_shape
            elif len(linears) == 3:
                # gate → up → down (LLaMA / Mistral SwiGLU)
                info.gate_proj_shape = tuple(linears[0][1].weight.shape)
                info.up_proj_shape   = tuple(linears[1][1].weight.shape)
                info.mlp_w1_shape    = info.up_proj_shape
                info.down_proj_shape = tuple(linears[2][1].weight.shape)
                info.mlp_w2_shape    = info.down_proj_shape
                if not info.activation_fn:
                    info.activation_fn = "silu"  # SwiGLU/SiLU implied by 3-linear MLP

        # Detect activation module type if still unknown
        if not info.activation_fn:
            for _, submod in mlp_mod.named_modules():
                t = type(submod).__name__.lower()
                if 'gelu' in t:
                    info.activation_fn = 'gelu'
                    break
                elif 'silu' in t or 'swish' in t:
                    info.activation_fn = 'silu'
                    break
                elif 'relu' in t:
                    info.activation_fn = 'relu'
                    break

    def _scan_norm(self, name: str, mod: nn.Module, base_path: str, info: 'LayerInfo') -> None:
        """Identify LayerNorm / RMSNorm submodules and classify them."""
        low = name.lower()
        is_norm = isinstance(mod, nn.LayerNorm) or type(mod).__name__ in {
            'RMSNorm', 'LlamaRMSNorm', 'MistralRMSNorm', 'Qwen2RMSNorm',
            'PhiRMSNorm', 'GemmaRMSNorm', 'FalconRMSNorm', 'T5LayerNorm',
        }
        if not is_norm:
            return

        norm_type = 'rms_norm' if 'rms' in type(mod).__name__.lower() else 'layer_norm'
        if info.norm_type is None:
            info.norm_type = norm_type

        path = f"{base_path}.{name}"
        if low in self._PRE_ATTN_NORM_NAMES:
            info.pre_attn_norm_path = path
        elif low in self._POST_ATTN_NORM_NAMES:
            info.post_attn_norm_path = path
            if info.pre_mlp_norm_path is None:
                info.pre_mlp_norm_path = path

    def _build_module_nodes(self, layer_module: nn.Module, base_path: str) -> List[dict]:
        """Build a flat Netron-style list of all sub-modules with weights in this layer."""
        nodes = []
        seen_paths = set()
        for rel_path, mod in layer_module.named_modules():
            if not rel_path:
                continue  # skip the layer itself
            full_path = f"{base_path}.{rel_path}"
            if full_path in seen_paths:
                continue
            seen_paths.add(full_path)

            node_type = type(mod).__name__
            shape = None
            if isinstance(mod, nn.Linear):
                shape = list(mod.weight.shape)
            elif isinstance(mod, nn.Embedding):
                shape = list(mod.weight.shape)
            elif isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                shape = list(mod.weight.shape) if mod.weight is not None else None
            elif hasattr(mod, 'weight') and mod.weight is not None:
                try:
                    shape = list(mod.weight.shape)
                except Exception:
                    pass

            # Only include modules that have weights or are meaningful ops
            meaningful = isinstance(mod, (
                nn.Linear, nn.Embedding, nn.LayerNorm, nn.GroupNorm,
                nn.MultiheadAttention, nn.Conv1d, nn.Conv2d,
                nn.Dropout, nn.GELU, nn.SiLU, nn.ReLU, nn.Sigmoid, nn.Tanh,
            )) or node_type in {
                'RMSNorm', 'LlamaRMSNorm', 'MistralRMSNorm', 'Qwen2RMSNorm',
                'PhiRMSNorm', 'GemmaRMSNorm', 'FalconRMSNorm', 'T5LayerNorm',
                'RotaryEmbedding', 'LlamaRotaryEmbedding', 'SwiGLU',
            }
            if not meaningful and shape is None:
                continue

            nodes.append({
                "path": full_path,
                "rel_path": rel_path,
                "type": node_type,
                "shape": shape,
                "has_bias": bool(getattr(mod, 'bias', None) is not None and
                                 not isinstance(getattr(mod, 'bias', None), bool)),
            })
        return nodes

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
