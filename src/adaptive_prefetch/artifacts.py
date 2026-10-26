"""JSON persistence for the small window classifiers (no pickle loading).

Each classifier kind stores a different set of sufficient statistics, so the
payload is tagged with ``classifier`` and the loader dispatches on it. A
payload whose tag, feature list, class list, or version does not match the
running code is rejected rather than silently mis-loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from .features import FEATURE_NAMES
from .model import CLASSIFIERS, OnlineGaussianNB, OnlineQDA
from .trace import CLASSES

FORMAT_VERSION = 2


def _statistics(model) -> dict:
    if isinstance(model, OnlineGaussianNB):
        return {
            "count": model.count,
            "mean": model.mean,
            "m2": model.m2,
        }
    if isinstance(model, OnlineQDA):
        return {
            "count": model.count,
            "mean": model.mean,
            "scatter": model.scatter,
        }
    raise ValueError(f"unsupported classifier type: {type(model).__name__}")


def save_model(path: str | Path, model, metadata: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "classifier": model.name,
        "features": list(FEATURE_NAMES),
        "classes": list(CLASSES),
        "variance_floor": model.variance_floor,
        "decay_factor": model.decay_factor,
        "statistics": _statistics(model),
        "metadata": metadata,
    }
    if isinstance(model, OnlineQDA):
        # Shrinkage changes every QDA prediction. It used to be dropped here,
        # so a model trained at any other value silently reloaded at 0.05.
        payload["shrink"] = model.shrink
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_model(path: str | Path, *, classifier: str | None = None,
               decay_factor: float | None = None, shrink: float | None = None):
    """Return ``(model, metadata)``; raise ``ValueError`` on any mismatch.

    ``classifier``, ``decay_factor`` and ``shrink``, when given, are what the
    caller expects the artifact to contain; a different stored value is
    rejected instead of being silently used.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (payload.get("format_version") != FORMAT_VERSION
            or payload.get("features") != list(FEATURE_NAMES)
            or payload.get("classes") != list(CLASSES)):
        raise ValueError(
            "incompatible classifier artifact: feature set, class list, or "
            "format version changed since it was saved. Regenerate it with "
            "`python -m adaptive_prefetch train-msr-sample`.")
    name = payload.get("classifier")
    if name not in CLASSIFIERS:
        raise ValueError(f"unknown classifier {name!r} in artifact")
    if classifier is not None and name != classifier:
        raise ValueError(
            f"artifact holds a {name!r} classifier, expected {classifier!r}")

    decay = float(payload["decay_factor"])
    floor = float(payload["variance_floor"])
    if not 0 < decay <= 1:
        raise ValueError("artifact decay_factor must be in (0, 1]")
    if floor <= 0:
        raise ValueError("artifact variance_floor must be positive")
    if decay_factor is not None and decay != decay_factor:
        raise ValueError(
            f"artifact decay_factor {decay} does not match the expected "
            f"{decay_factor}; regenerate the artifact")

    if name == "gnb":
        if shrink is not None:
            raise ValueError("shrink applies to qda artifacts only")
        model = OnlineGaussianNB(len(FEATURE_NAMES), floor, decay)
    else:
        if "shrink" not in payload:
            raise ValueError(
                "qda artifact has no stored shrink, so the value it was "
                "trained with is unknown; regenerate the artifact")
        stored = float(payload["shrink"])
        if not 0.0 <= stored < 1.0:
            raise ValueError("artifact shrink must be in [0, 1)")
        if shrink is not None and stored != shrink:
            raise ValueError(
                f"artifact shrink {stored} does not match the expected "
                f"{shrink}; regenerate the artifact")
        model = OnlineQDA(len(FEATURE_NAMES), floor, decay, stored)

    stats = payload.get("statistics")
    if not isinstance(stats, dict):
        raise ValueError("invalid classifier statistics in artifact: no statistics block")
    try:
        for label in CLASSES:
            count = float(stats["count"][label])
            mean = [float(value) for value in stats["mean"][label]]
            if count < 0 or len(mean) != model.n_features:
                raise ValueError(f"invalid statistics for class {label!r}")
            if name == "gnb":
                m2 = [float(value) for value in stats["m2"][label]]
                if len(m2) != model.n_features:
                    raise ValueError(f"invalid statistics for class {label!r}")
                model.count[label] = count
                model.mean[label] = mean
                model.m2[label] = m2
            else:
                width = model.n_features * model.n_features
                scatter = [float(value) for value in stats["scatter"][label]]
                if len(scatter) != width:
                    raise ValueError(f"invalid scatter matrix for class {label!r}")
                model.count[label] = count
                model.mean[label] = mean
                model.scatter[label] = scatter
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid classifier statistics in artifact: {exc}") from exc

    if sum(model.count.values()) <= 0:
        raise ValueError("artifact contains no trained statistics")
    return model, payload.get("metadata", {})
