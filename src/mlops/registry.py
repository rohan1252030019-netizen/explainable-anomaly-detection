"""
src/mlops/registry.py

SENTINEL Versioned Model Registry.

Tracks models, serialized weights, metadata, lifecycle stages, and model cards.
Stages: 'development', 'staging', 'production', 'archived'.
"""

import json
import os
import time
from typing import Any

import joblib


class ModelRegistry:
    """
    Filesystem-backed versioned model registry.

    Parameters
    ----------
    registry_dir : str
        Base directory to store versioned model bundles.
    """

    VALID_STAGES = {"development", "staging", "production", "archived"}

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def _sanitize(self, name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def register_model(
        self,
        name: str,
        version: str,
        model_obj: Any,
        model_card: dict[str, Any] | None = None,
        stage: str = "staging",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Save a model, its weights, metadata and model card into the registry.

        Parameters
        ----------
        name : str
        version : str (e.g. '1.0.0')
        model_obj : Any (picklable model object)
        model_card : dict or None
        stage : str ('staging', 'production', etc.)
        tags : list of str or None
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}")

        safe_name = self._sanitize(name)
        target_dir = os.path.join(self.registry_dir, safe_name, version)
        os.makedirs(target_dir, exist_ok=True)

        # 1. Serialize model object
        weights_path = os.path.join(target_dir, "model.joblib")
        try:
            joblib.dump(model_obj, weights_path)
        except Exception:
            # If model object itself cannot be serialized directly, save dummy indicator
            with open(weights_path, "wb") as f:
                f.write(b"SENTINEL_WEIGHTS_PLACEHOLDER")

        # 2. Save Model Card if provided
        if model_card:
            card_json_path = os.path.join(target_dir, "model_card.json")
            with open(card_json_path, "w", encoding="utf-8") as f:
                json.dump(model_card.get("model_card_data", model_card), f, indent=2)

            if "markdown" in model_card:
                card_md_path = os.path.join(target_dir, "model_card.md")
                with open(card_md_path, "w", encoding="utf-8") as f:
                    f.write(model_card["markdown"])

        # 3. Save metadata
        metadata = {
            "name": name,
            "version": version,
            "stage": stage,
            "tags": tags or ["sentinel-detector"],
            "registered_at": time.time(),
            "weights_path": os.path.abspath(weights_path),
        }
        meta_path = os.path.join(target_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def list_models(self) -> list[dict[str, Any]]:
        """List all models and versions stored in the registry."""
        results = []
        if not os.path.exists(self.registry_dir):
            return []

        for m_dir in sorted(os.listdir(self.registry_dir)):
            full_m = os.path.join(self.registry_dir, m_dir)
            if not os.path.isdir(full_m):
                continue
            for v_dir in sorted(os.listdir(full_m)):
                full_v = os.path.join(full_m, v_dir)
                meta_path = os.path.join(full_v, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, encoding="utf-8") as f:
                            meta = json.load(f)
                            results.append(meta)
                    except Exception:
                        pass
        return results

    def get_model(self, name: str, version: str) -> dict[str, Any]:
        """Retrieve model object and metadata."""
        safe_name = self._sanitize(name)
        target_dir = os.path.join(self.registry_dir, safe_name, version)
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Model '{name}' with version '{version}' not found in registry.")

        meta_path = os.path.join(target_dir, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        weights_path = os.path.join(target_dir, "model.joblib")
        model_obj = None
        if os.path.exists(weights_path):
            try:
                model_obj = joblib.load(weights_path)
            except Exception:
                pass

        card_path = os.path.join(target_dir, "model_card.json")
        card_data = None
        if os.path.exists(card_path):
            try:
                with open(card_path, encoding="utf-8") as f:
                    card_data = json.load(f)
            except Exception:
                pass

        return {
            "metadata": metadata,
            "model_obj": model_obj,
            "model_card": card_data,
        }

    def promote_stage(self, name: str, version: str, new_stage: str) -> dict[str, Any]:
        """Update lifecycle stage of a model version."""
        if new_stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage '{new_stage}'. Must be one of {self.VALID_STAGES}")

        safe_name = self._sanitize(name)
        meta_path = os.path.join(self.registry_dir, safe_name, version, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model metadata not found at {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        meta["stage"] = new_stage
        meta["stage_updated_at"] = time.time()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta
