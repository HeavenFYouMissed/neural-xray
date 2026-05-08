"""
Stage 1: LOAD
=============
Load any HuggingFace model with automatic quantization to fit in available VRAM.
Detects GPU memory and selects the appropriate quantization level automatically.

Supported models: anything on HuggingFace (Qwen, LLaMA, Mistral, Phi, Gemma, etc.)
"""

import torch
import json
from pathlib import Path
from typing import Optional


class ModelLoader:
    """Load a pretrained HuggingFace model with automatic VRAM management.

    Automatically picks quantization:
        - 6GB  VRAM → 4-bit (NF4)  — fits 7B models
        - 8GB  VRAM → 4-bit or 8-bit
        - 16GB VRAM → 8-bit or float16
        - CPU only  → 8-bit with CPU offload

    Args:
        model_name: HuggingFace model ID, e.g. "Qwen/Qwen2-1.5B"
        device: "auto", "cuda", "cpu"
        force_quantization: None (auto), "4bit", "8bit", "float16", "float32"
        cache_dir: Where to cache downloaded model weights
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        force_quantization: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.force_quantization = force_quantization
        self.cache_dir = cache_dir
        self.hf_token = hf_token

        self.model = None
        self.tokenizer = None
        self.hidden_size: Optional[int] = None
        self.num_layers: Optional[int] = None
        self.quantization_used: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────

    def load(self) -> "ModelLoader":
        """Load the model. Returns self for chaining."""
        print(f"\n[AutoPsy:LOAD] Loading {self.model_name}")

        quant = self.force_quantization or self._pick_quantization()
        print(f"  Quantization: {quant}")
        print(f"  VRAM available: {self._vram_gb():.1f} GB")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(quant)
        self.quantization_used = quant

        # Cache key model dimensions
        self.hidden_size = self._detect_hidden_size()
        self.num_layers = self._detect_num_layers()

        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Loaded OK")
        return self

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[AutoPsy:LOAD] Unloaded {self.model_name}")

    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Internal ───────────────────────────────────────────────────

    def _vram_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        return 0.0

    def _pick_quantization(self) -> str:
        vram = self._vram_gb()

        # Check if bitsandbytes is available (not on Windows typically)
        bnb_available = False
        try:
            import bitsandbytes  # noqa: F401
            bnb_available = True
        except ImportError:
            pass

        if vram == 0:
            return "8bit" if bnb_available else "float32"  # CPU
        elif vram < 8:
            return "4bit" if bnb_available else "float16"
        elif vram < 20:
            return "8bit" if bnb_available else "float16"
        else:
            return "float16"

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("pip install transformers")

        tok = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            token=self.hf_token,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _load_model(self, quant: str):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("pip install transformers")

        kwargs = dict(
            pretrained_model_name_or_path=self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            token=self.hf_token,
            output_hidden_states=True,  # critical — we need all layer outputs
        )

        if quant in ("4bit", "8bit"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError("pip install bitsandbytes")

            if quant == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
        elif quant == "float16":
            kwargs["dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:  # float32
            kwargs["dtype"] = torch.float32
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        model.eval()
        return model

    def _detect_hidden_size(self) -> int:
        """Find the hidden dimension by inspecting model config."""
        cfg = self.model.config
        for attr in ("hidden_size", "d_model", "n_embd", "dim"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise ValueError(f"Cannot detect hidden_size from {type(cfg)}")

    def _detect_num_layers(self) -> int:
        """Find the number of transformer layers."""
        cfg = self.model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
        raise ValueError(f"Cannot detect num_layers from {type(cfg)}")

    def info(self) -> dict:
        """Return a summary dict of the loaded model."""
        return {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "quantization": self.quantization_used,
            "vram_gb": self._vram_gb(),
            "param_count": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
        }
