"""
Stage 4b: BLUEPRINT
===================
Saves a complete, human-readable JSON "neural blueprint" of the extracted knowledge.

This is the permanent record — the autopsy report. It contains:
- Source model architecture details
- All extracted concept vectors (as base64-encoded tensors)
- The semantic cluster map
- Metadata for reproducibility

The blueprint is the output artifact. You can load it years later
to transplant the same knowledge into a new target architecture,
without needing the original source model anymore.
"""

import torch
import json
import base64
import io
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class NeuralBlueprint:
    """Save and load the complete output of an autopsy run.

    A blueprint is a self-contained file that records:
    1. What model was dissected
    2. The architecture map
    3. All extracted concept vectors
    4. The semantic cluster analysis
    5. Checksums for verification

    Args:
        output_dir: Directory to save/load blueprint files
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        arch_map,
        entity_vectors: Dict[str, torch.Tensor],
        relation_vectors: Dict[str, torch.Tensor],
        cluster_data: Optional[dict] = None,
        notes: str = "",
    ) -> str:
        """Save a complete blueprint. Returns the path to the blueprint file."""
        print(f"\n[AutoPsy:BLUEPRINT] Saving neural blueprint")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = arch_map.model_name.replace("/", "_").replace("-", "_")
        blueprint_name = f"blueprint_{model_slug}_{timestamp}"
        blueprint_dir = self.output_dir / blueprint_name
        blueprint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save architecture map
        with open(blueprint_dir / "architecture.json", "w") as f:
            json.dump(arch_map.to_dict(), f, indent=2)
        print(f"  Saved architecture map")

        # 2. Save entity vectors
        torch.save(
            {k: v.cpu() for k, v in entity_vectors.items()},
            blueprint_dir / "entity_vectors.pt",
        )
        print(f"  Saved {len(entity_vectors)} entity vectors")

        # 3. Save relation vectors
        torch.save(
            {k: v.cpu() for k, v in relation_vectors.items()},
            blueprint_dir / "relation_vectors.pt",
        )
        print(f"  Saved {len(relation_vectors)} relation vectors")

        # 4. Save cluster analysis
        if cluster_data is not None:
            serializable = cluster_data.to_dict() if hasattr(cluster_data, "to_dict") else cluster_data
            with open(blueprint_dir / "clusters.json", "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"  Saved cluster analysis")

        # 5. Save metadata / manifest
        manifest = {
            "blueprint_name": blueprint_name,
            "created_at": datetime.now().isoformat(),
            "source_model": arch_map.model_name,
            "hidden_size": arch_map.hidden_size,
            "num_layers": arch_map.num_layers,
            "entity_count": len(entity_vectors),
            "relation_count": len(relation_vectors),
            "fact_layer_start": arch_map.fact_layer_start,
            "fact_layer_end": arch_map.fact_layer_end,
            "notes": notes,
            "files": {
                "architecture": "architecture.json",
                "entity_vectors": "entity_vectors.pt",
                "relation_vectors": "relation_vectors.pt",
                "clusters": "clusters.json" if cluster_data else None,
            },
            # Checksum for integrity verification
            "entity_vector_checksum": self._checksum_vectors(entity_vectors),
        }
        with open(blueprint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        blueprint_path = str(blueprint_dir)
        print(f"  Blueprint saved: {blueprint_path}")
        return blueprint_path

    @classmethod
    def load(cls, blueprint_path: str) -> dict:
        """Load a complete blueprint from disk.

        Returns a dict with keys:
            manifest, architecture, entity_vectors, relation_vectors, clusters
        """
        p = Path(blueprint_path)
        print(f"\n[AutoPsy:BLUEPRINT] Loading blueprint from {p.name}")

        with open(p / "manifest.json") as f:
            manifest = json.load(f)

        with open(p / "architecture.json") as f:
            architecture = json.load(f)

        entity_vectors = torch.load(
            p / "entity_vectors.pt", map_location="cpu", weights_only=True
        )

        relation_vectors = torch.load(
            p / "relation_vectors.pt", map_location="cpu", weights_only=True
        )

        clusters = None
        clusters_path = p / "clusters.json"
        if clusters_path.exists():
            with open(clusters_path) as f:
                clusters = json.load(f)

        print(f"  Source: {manifest['source_model']}")
        print(f"  Entities: {manifest['entity_count']}")
        print(f"  Relations: {manifest['relation_count']}")
        print(f"  Created: {manifest['created_at']}")

        return {
            "manifest": manifest,
            "architecture": architecture,
            "entity_vectors": entity_vectors,
            "relation_vectors": relation_vectors,
            "clusters": clusters,
        }

    def list_blueprints(self) -> List[dict]:
        """List all saved blueprints in the output directory."""
        blueprints = []
        for d in sorted(self.output_dir.iterdir()):
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    m = json.load(f)
                blueprints.append({
                    "path": str(d),
                    "name": m.get("blueprint_name"),
                    "source_model": m.get("source_model"),
                    "entity_count": m.get("entity_count"),
                    "created_at": m.get("created_at"),
                })
        return blueprints

    @staticmethod
    def _checksum_vectors(vectors: Dict[str, torch.Tensor]) -> str:
        """Compute a hash of all vectors for integrity checking."""
        buf = io.BytesIO()
        for k in sorted(vectors.keys()):
            buf.write(k.encode())
            buf.write(vectors[k].cpu().numpy().tobytes())
        return hashlib.md5(buf.getvalue()).hexdigest()
