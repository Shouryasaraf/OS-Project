"""Conservative weak-label adaptation for the tiny unlabelled MSR sample."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .artifacts import save_model
from .benchmark import make_model
from .features import extract
from .trace import CLASSES, load_csv, synthetic_dataset


def synthetic_accuracy(model, seed: int) -> float:
    held_out = synthetic_dataset(40, seed + 100000)
    return sum(model.predict(extract(window))[0] == label
               for window, label in held_out) / len(held_out)


def adapt_msr_sample(trace_path: str | Path, save_path: str | Path,
                     seed: int = 42, window_size: int = 32) -> dict[str, object]:
    if window_size < 8:
        raise ValueError("window_size must be at least 8")
    source = Path(trace_path)
    requests = load_csv(source, "msr")
    model = make_model(seed)
    before = synthetic_accuracy(model, seed)
    complete_windows = len(requests) // window_size
    selected = 0
    for index in range(complete_windows):
        window = requests[index * window_size:(index + 1) * window_size]
        features = extract(window)
        contiguous, repeated_stride, distant_jumps = features[:3]
        read_ratio = features[7]
        # A deliberately narrow proxy label. This does not certify workload
        # ground truth or provide four-class labels for the sample.
        if (read_ratio >= 0.5 and contiguous <= 0.2 and
                repeated_stride <= 0.2 and distant_jumps >= 0.75):
            model.update(features, "random")
            selected += 1
    if not selected:
        raise ValueError("no read-heavy, clearly random-like windows in sample")
    after = synthetic_accuracy(model, seed)
    metadata: dict[str, object] = {
        "training_kind": "synthetic_pretrain_plus_weak_msr_adaptation",
        "source_name": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "requests": len(requests),
        "window_size": window_size,
        "complete_windows": complete_windows,
        "real_windows_updated": selected,
        "real_proxy_class": "random",
        "real_label_rule": (
            "read_ratio>=0.5; contiguous_ratio<=0.2; "
            "dominant_stride_ratio<=0.2; random_jump_ratio>=0.75"
        ),
        "true_real_labels_available": False,
        "synthetic_held_out_accuracy_before": before,
        "synthetic_held_out_accuracy_after": after,
        "seed": seed,
        "classes": list(CLASSES),
    }
    save_model(save_path, model, metadata)
    return metadata
