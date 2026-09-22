"""JSON persistence for the small Gaussian NB classifier (no pickle loading)."""

from __future__ import annotations

import json
from pathlib import Path

from .features import FEATURE_NAMES
from .model import OnlineGaussianNB
from .trace import CLASSES


def save_model(path: str | Path, model: OnlineGaussianNB,
               metadata: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "features": list(FEATURE_NAMES),
        "classes": list(CLASSES),
        "variance_floor": model.variance_floor,
        "decay_factor": model.decay_factor,
        "count": model.count,
        "mean": model.mean,
        "m2": model.m2,
        "metadata": metadata,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_model(path: str | Path) -> tuple[OnlineGaussianNB, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (payload.get("format_version") != 1 or
            payload.get("features") != list(FEATURE_NAMES) or
            payload.get("classes") != list(CLASSES)):
        raise ValueError("incompatible classifier artifact")
    model = OnlineGaussianNB(len(FEATURE_NAMES),
                             float(payload["variance_floor"]),
                             float(payload["decay_factor"]))
    for label in CLASSES:
        count = float(payload["count"][label])
        mean = [float(value) for value in payload["mean"][label]]
        m2 = [float(value) for value in payload["m2"][label]]
        if count < 0 or len(mean) != model.n_features or len(m2) != model.n_features:
            raise ValueError("invalid classifier statistics in artifact")
        model.count[label] = count
        model.mean[label] = mean
        model.m2[label] = m2
    return model, payload.get("metadata", {})
