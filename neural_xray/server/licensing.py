"""
Model Surgery MRI — License tier enforcement.

Tiers:
  free   — precomputed only (web demo)
  indie  — ≤1B params, basic features
  pro    — ≤13B params, all features including surgery
  enterprise — unlimited
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("model-surgery")

LICENSE_PATH = Path.home() / ".model-surgery" / "license.key"

TIER_LIMITS = {
    "free": {
        "model_size_cap": 0,
        "features": {"trace": "view_only", "cluster": "view_only",
                      "surgery": False, "sae": False, "export": False,
                      "batch": False, "multi_gpu": False, "api_access": False},
    },
    "indie": {
        "model_size_cap": 1_000_000_000,
        "features": {"trace": True, "cluster": True,
                      "surgery": False, "sae": "limited", "export": False,
                      "batch": False, "multi_gpu": False, "api_access": False},
    },
    "pro": {
        "model_size_cap": 13_000_000_000,
        "features": {"trace": True, "cluster": True,
                      "surgery": True, "sae": True, "export": True,
                      "batch": True, "multi_gpu": False, "api_access": False},
    },
    "enterprise": {
        "model_size_cap": float("inf"),
        "features": {"trace": True, "cluster": True,
                      "surgery": True, "sae": True, "export": True,
                      "batch": True, "multi_gpu": True, "api_access": True},
    },
}


@dataclass
class License:
    tier: str
    model_size_cap: int
    features: Dict[str, object]
    expiry: float  # Unix timestamp, 0 = never
    machine_id: str


def _get_machine_id() -> str:
    """Simple machine fingerprint — good enough for v1."""
    import platform
    raw = f"{platform.node()}-{platform.machine()}-{platform.processor()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_license() -> License:
    """Load license from disk. Returns 'pro' tier by default during development."""
    if LICENSE_PATH.exists():
        try:
            data = json.loads(LICENSE_PATH.read_text())
            tier = data.get("tier", "free")
            limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
            return License(
                tier=tier,
                model_size_cap=limits["model_size_cap"],
                features=limits["features"],
                expiry=data.get("expiry", 0),
                machine_id=data.get("machine_id", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to read license: {e}")

    # Development mode — full access
    logger.info("No license file found — running in development mode (pro tier)")
    limits = TIER_LIMITS["pro"]
    return License(
        tier="pro",
        model_size_cap=limits["model_size_cap"],
        features=limits["features"],
        expiry=0,
        machine_id=_get_machine_id(),
    )


# Singleton
_license: Optional[License] = None


def get_license() -> License:
    global _license
    if _license is None:
        _license = load_license()
    return _license


def check_model_size(num_params: int) -> bool:
    """Returns True if the model is within the license tier's size cap."""
    lic = get_license()
    return num_params <= lic.model_size_cap


def check_feature(feature: str) -> bool:
    """Returns True if the feature is enabled for the current tier."""
    lic = get_license()
    val = lic.features.get(feature, False)
    # "limited" and "view_only" count as truthy for basic access
    return bool(val)


def require_feature(feature: str):
    """Raises ValueError if feature is not available."""
    if not check_feature(feature):
        lic = get_license()
        raise ValueError(
            f"Feature '{feature}' requires a higher tier (current: {lic.tier})"
        )
