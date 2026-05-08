"""
neural-xray — open source, no license enforcement.

This module previously implemented tier-based feature gating for a commercial
build. In the open-source release every check is a no-op so all features and
all model sizes are available to everyone.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class License:
    tier: str = "open-source"
    model_size_cap: float = float("inf")
    features: Dict[str, object] = field(default_factory=lambda: {
        "trace": True, "cluster": True, "surgery": True, "sae": True,
        "export": True, "batch": True, "multi_gpu": True, "api_access": True,
    })
    expiry: float = 0
    machine_id: str = ""


TIER_LIMITS = {
    "open-source": {
        "model_size_cap": float("inf"),
        "features": License().features,
    },
}


_license = License()


def get_license() -> License:
    return _license


def load_license() -> License:
    return _license


def check_model_size(num_params: int) -> bool:
    return True


def check_feature(feature: str) -> bool:
    return True


def require_feature(feature: str):
    return None
