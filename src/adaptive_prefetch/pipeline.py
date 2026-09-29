"""Training/evaluation pipeline: preprocessing, postprocessing, model selection.

The pieces here exist because the project's original evaluation had two
structural weaknesses:

1. A single train/test split on one seed, which reports 100% on a generator
   that is nearly deterministic. :func:`evaluate_classifier` uses disjoint
   seeds and optional harder regimes so the number means something.
2. A raw per-window argmax feeding a policy directly, which thrashes on noisy
   windows. :class:`PolicySmoother` adds the standard hysteresis/smoothing
   step and can only change results when explicitly enabled.
"""

from __future__ import annotations

import random
from statistics import mean, pstdev
from time import perf_counter

from .features import FEATURE_NAMES, extract
from .model import CLASSIFIERS, DEFAULT_CLASSIFIER, make_classifier
from .trace import CLASSES, Request, synthetic_dataset


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

class StandardScaler:
    """Zero-mean unit-variance feature scaling fitted on a training set.

    The window features are all already ratios in [0, 1] except ``gap_cv``,
    which can reach 10. Scaling puts them on comparable footing before any
    distance-based model (QDA) computes covariances. It is a no-op for Naive
    Bayes, which is scale-invariant per feature.
    """

    def __init__(self):
        self.mean: list[float] | None = None
        self.scale: list[float] | None = None

    def fit(self, rows: list[tuple]) -> "StandardScaler":
        if not rows:
            raise ValueError("cannot fit a scaler on no data")
        dimension = len(rows[0])
        self.mean = [mean(r[i] for r in rows) for i in range(dimension)]
        self.scale = []
        for i in range(dimension):
            sd = pstdev([r[i] for r in rows]) or 0.0
            self.scale.append(sd if sd > 1e-12 else 1.0)
        return self

    def transform(self, values: tuple[float, ...]) -> tuple[float, ...]:
        if self.mean is None or self.scale is None:
            raise ValueError("scaler is not fitted")
        return tuple((v - m) / s for v, m, s in zip(values, self.mean, self.scale))

    def transform_rows(self, rows: list[tuple]) -> list[tuple]:
        return [self.transform(row) for row in rows]


def clip_outliers(rows: list[tuple], low: float = -6.0,
                  high: float = 6.0) -> list[tuple]:
    """Clamp standardized rows; guards against a single wild window in
    real traces dominating a covariance estimate."""
    return [tuple(max(low, min(high, v)) for v in row) for row in rows]


# --------------------------------------------------------------------------
# Postprocessing
# --------------------------------------------------------------------------

class PolicySmoother:
    """Consecutive-run hysteresis over window predictions.

    A single misclassified window flips the prefetch policy for a whole
    window, which is pure wasted I/O. Requiring ``min_windows`` *consecutive*
    agreeing predictions before switching removes that churn.

    The run length is counted at the tail of the history, not as a total
    within it: counting totals means a strictly alternating stream still
    produces a 3:2 majority once the buffer trims, which is exactly the churn
    this class exists to remove.
    """

    def __init__(self, min_windows: int = 2, hysteresis: float = 0.0,
                 history: int = 5):
        if min_windows < 1:
            raise ValueError("min_windows must be >= 1")
        self.min_windows = min_windows
        self.hysteresis = hysteresis
        self.history: list[str] = []
        self.capacity = max(history, min_windows)
        self.current: str | None = None
        self.switches = 0

    def _tail_run(self) -> tuple[str, int]:
        """Label and length of the run of identical predictions at the tail."""
        label = self.history[-1]
        run = 1
        for previous in reversed(self.history[:-1]):
            if previous != label:
                break
            run += 1
        return label, run

    def update(self, label: str, confidence: float) -> str:
        self.history.append(label)
        if len(self.history) > self.capacity:
            del self.history[0]
        if self.current is None:
            self.current = label
            return label
        if label == self.current:
            return self.current
        run_label, run = self._tail_run()
        if run_label != label or run < self.min_windows:
            return self.current
        if self.hysteresis and confidence < self.hysteresis:
            return self.current
        self.current = run_label
        self.switches += 1
        return self.current


# --------------------------------------------------------------------------
# Datasets for evaluation
# --------------------------------------------------------------------------

def labelled_dataset(windows_per_class: int, seed: int, n: int = 32
                     ) -> list[tuple[list[Request], str]]:
    return synthetic_dataset(windows_per_class, seed, n)


def make_harder_generator(label: str, rng: random.Random, n: int = 32,
                          noise: float = 0.3, lba_base: int = 1000,
                          lba_span: int = 99000, size: int = 1,
                          stride_range: tuple[int, int] = (4, 16)) -> list[Request]:
    """A configurable generator for regimes harder than the shipped one.

    ``noise`` raises the random-jump rate for sequential/strided phases.
    ``stride_range`` is what actually creates *class overlap*: once a strided
    phase draws strides of 1-3 it is geometrically indistinguishable from a
    sequential phase, which is where a diagonal covariance model starts to
    fail and a full one does not.
    """
    if label not in CLASSES or n < 8:
        raise ValueError("label must be a known class and n >= 8")
    current = rng.randrange(lba_base, lba_base + lba_span)
    stride = rng.randint(*stride_range)
    time = 0.0
    result: list[Request] = []
    for i in range(n):
        block = max(1, rng.randint(1, 3) * size)
        if label == "sequential":
            if i:
                current += block
                if rng.random() < noise:
                    current += rng.randint(5, 20)
        elif label == "strided":
            if i:
                current += stride + (rng.randint(1, 5) if rng.random() < noise else 0)
        elif label == "random":
            current = rng.randrange(lba_base, lba_base + lba_span)
        else:
            if i % 8 in (0, 5, 6, 7):
                current = rng.randrange(lba_base, lba_base + lba_span)
        time += rng.uniform(0.01, 0.1) if label == "mixed" and i % 8 < 4 \
            else rng.uniform(0.2, 2.0)
        result.append(Request(round(time, 4), current, block,
                              "R" if rng.random() < 0.9 else "W"))
        if label in ("sequential", "mixed") and (label != "mixed" or i % 8 < 3):
            current += block
    return result


def regime(name: str, windows_per_class: int, seed: int, n: int = 32,
           **kwargs) -> list[tuple[list[Request], str]]:
    """Build a labelled dataset for a named evaluation regime."""
    rng = random.Random(seed)
    examples = [(make_harder_generator(label, rng, n, **kwargs), label)
                for label in CLASSES for _ in range(windows_per_class)]
    rng.shuffle(examples)
    return examples


def blended_regime(windows_per_class: int, seed: int, n: int = 32,
                   blend: float = 0.5) -> list[tuple[list[Request], str]]:
    """Windows that straddle a phase change, labelled by the dominant phase.

    Real workload streams change behaviour mid-window, so a window genuinely
    contains two patterns at once and no feature-based classifier can be
    fully right about which one it is. This is the irreducible-error regime:
    the residual here is a property of the task, not of the model, and it is
    where a per-window argmax is genuinely the wrong decision rule (see
    :class:`PolicySmoother`).
    """
    rng = random.Random(seed)
    others = [c for c in CLASSES]
    examples = []
    for label in CLASSES:
        for _ in range(windows_per_class):
            partner = rng.choice([c for c in others if c != label])
            head = make_harder_generator(label, rng, n, noise=0.05)
            tail = make_harder_generator(partner, rng, n, noise=0.05)
            cut = int(n * (1.0 - blend))
            merged = head[:cut] + tail[cut:]
            # Re-stamp timestamps so the window stays monotonic in time.
            step = 0.5
            for index, request in enumerate(merged):
                merged[index] = Request((index + 1) * step, request.lba,
                                        request.size_blocks, request.operation)
            examples.append((merged, label))
    rng.shuffle(examples)
    return examples


REGIMES = {
    # Shipped generator.
    "baseline":  dict(),
    # Heavier random jumps, strides still clearly distinguishable.
    "noisy":     dict(noise=0.30),
    # Small strides (1-3): a strided phase becomes geometrically close to a
    # sequential one, so the classes overlap. This is the regime that separates
    # a diagonal covariance model from a full one.
    "overlap":   dict(noise=0.20, stride_range=(1, 3)),
    "overlap2":  dict(noise=0.35, stride_range=(1, 2)),
    # Realistic device geometry: 1e8-1e11 LBA span, multi-block requests.
    "realistic": dict(noise=0.05, lba_base=10**8, lba_span=10**11, size=8),
}

#: Regimes built by a dedicated generator rather than by keyword overrides.
BLENDED_REGIMES = ("transition", "transition_heavy")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _fit_and_test(train: list[tuple], test: list[tuple], classifier: str,
                  scale: bool) -> tuple[float, float, float]:
    """Return (accuracy, predict_us_per_call, train_seconds)."""
    train_x = [extract(window) for window, _ in train]
    test_x = [extract(window) for window, _ in test]

    scaler = StandardScaler().fit(train_x) if scale else None
    if scaler is not None:
        train_x = scaler.transform_rows(train_x)
        test_x = scaler.transform_rows(test_x)

    model = make_classifier(classifier, len(FEATURE_NAMES))
    start = perf_counter()
    for values, (_, label) in zip(train_x, train):
        model.update(values, label)
    train_seconds = perf_counter() - start

    correct = 0
    calls = 0
    predict_seconds = 0.0
    for values, (_, label) in zip(test_x, test):
        t0 = perf_counter()
        predicted, _ = model.predict(values)
        predict_seconds += perf_counter() - t0
        calls += 1
        if predicted == label:
            correct += 1
    accuracy = correct / len(test) if test else 0.0
    per_call_us = 1e6 * predict_seconds / calls if calls else 0.0
    return accuracy, per_call_us, train_seconds


def evaluate_classifier(classifier: str = DEFAULT_CLASSIFIER, seeds: int = 12,
                        windows_per_class: int = 100, test_per_class: int = 40,
                        regime_name: str = "baseline", scale: bool = False,
                        ) -> dict:
    """Accuracy and cost for one classifier on one regime over disjoint seeds.

    Train and test are generated from *different* seeds, so no window can
    appear in both.
    """
    params = REGIMES.get(regime_name)
    if params is None and regime_name not in BLENDED_REGIMES:
        raise ValueError(f"unknown regime {regime_name!r}; "
                         f"choose from {', '.join(REGIMES)}")
    accuracies, per_call, train_times = [], [], []
    for i in range(seeds):
        if regime_name in BLENDED_REGIMES:
            blend = 0.5 if regime_name == "transition" else 0.75
            train = blended_regime(windows_per_class, 1000 + i, 32, blend)
            test = blended_regime(test_per_class, 900000 + i, 32, blend)
        elif regime_name == "baseline":
            train = labelled_dataset(windows_per_class, 1000 + i, 32)
            test = labelled_dataset(test_per_class, 900000 + i, 32)
        else:
            train = regime(regime_name, windows_per_class, 1000 + i, 32, **params)
            test = regime(regime_name, test_per_class, 900000 + i, 32, **params)
        accuracy, us, seconds = _fit_and_test(train, test, classifier, scale)
        accuracies.append(accuracy)
        per_call.append(us)
        train_times.append(seconds)
    return {
        "classifier": classifier,
        "regime": regime_name,
        "accuracy": mean(accuracies),
        "accuracy_sd": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "min": min(accuracies),
        "max": max(accuracies),
        "predict_us": mean(per_call),
        "train_ms": 1000 * mean(train_times),
    }


def select_classifier(seeds: int = 12, windows_per_class: int = 100,
                      test_per_class: int = 40) -> dict:
    """Compare every registered classifier across every regime.

    Returns a report string plus the structured results, so the CLI can print
    it and tests can assert on the numbers.
    """
    results = []
    lines = ["# Classifier selection", "",
             f"Protocol: {seeds} disjoint seed pairs per cell "
             f"(train seed 1000+i, test seed 900000+i), "
             f"{windows_per_class} train / {test_per_class} test windows "
             f"per class.", ""]
    header = f"| {'regime':<12} | " + " | ".join(
        f"{name:>18}" for name in CLASSIFIERS) + " |"
    lines.append(header)
    lines.append("| " + " | ".join(["---"] * (len(CLASSIFIERS) + 1)) + " |")
    for regime_name in list(REGIMES) + list(BLENDED_REGIMES):
        cells = []
        for name in CLASSIFIERS:
            stats = evaluate_classifier(name, seeds, windows_per_class,
                                        test_per_class, regime_name)
            results.append(stats)
            cells.append(f"{100 * stats['accuracy']:6.2f} +/- "
                         f"{100 * stats['accuracy_sd']:4.2f}")
        lines.append(f"| {regime_name:<12} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cost (median over regimes):")
    lines.append("")
    lines.append("| classifier | predict us/call | train ms |")
    lines.append("| --- | ---: | ---: |")
    for name in CLASSIFIERS:
        subset = [r for r in results if r["classifier"] == name]
        if subset:
            lines.append(f"| {name} | {mean(r['predict_us'] for r in subset):.2f} "
                         f"| {mean(r['train_ms'] for r in subset):.2f} |")
    return {"report": "\n".join(lines), "results": results}
