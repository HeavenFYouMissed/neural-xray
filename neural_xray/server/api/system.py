"""
Model Surgery MRI — /api/system endpoints.

GPU info, health checks, and license status.
"""

import logging
import platform
import sys
from typing import Optional

from fastapi import APIRouter

from ..licensing import get_license, TIER_LIMITS
from ..state import state

logger = logging.getLogger("model-surgery")
router = APIRouter()


@router.get("/gpu")
async def gpu_info():
    """Get GPU information and VRAM status."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_free = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_allocated(0)) / (1024**3)
            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            return {
                "available": True,
                "gpu_name": gpu_name,
                "vram_total_gb": round(vram_total, 2),
                "vram_free_gb": round(vram_free, 2),
                "vram_used_gb": round(vram_used, 2),
                "cuda_version": torch.version.cuda,
            }
        else:
            return {"available": False, "gpu_name": None, "reason": "No CUDA GPU detected"}
    except ImportError:
        return {"available": False, "gpu_name": None, "reason": "PyTorch not installed"}


@router.get("/license")
async def license_info():
    """Get current license tier and feature flags."""
    lic = get_license()
    return {
        "tier": lic.tier,
        "model_size_cap": lic.model_size_cap,
        "features": lic.features,
        "machine_id": lic.machine_id,
    }


@router.get("/info")
async def system_info():
    """Get system information."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": sys.version,
        "platform": platform.machine(),
        "loaded_models": len(state.sessions),
        "model_ids": list(state.sessions.keys()),
    }
