"""Labelled synthetic window vectors for the classifier zoo (stdlib only).

Every vector is the 12-feature contextual layout ``FEATURE_NAMES`` that the
``adaptive`` replay builds, produced through the same ``WindowContext`` path:

* ``baseline`` is ``benchmark.make_model``'s own generator and layout (one
  continuous ``trace.synthetic_window`` stream per class, one
  ``WindowContext`` per class, classes fitted one after another).
  :func:`baseline_vectors` feeding ``OnlineGaussianNB`` in order reproduces
  ``make_model(seed)`` exactly (pinned by ``tests/test_clfzoo.py``).
* the harder regimes of ``docs/CLASSIFIER_EVAL.md`` (``noisy``, ``overlap``,
  ``overlap2``, ``realistic``, ``transition``, ``transition_heavy``) draw the
  windows exactly as ``pipeline.regime`` / ``pipeline.blended_regime`` do,
  then lay each class out as a ``make_model``-style stream so the four
  contextual features carry stream values rather than cold-start zeros;
* ``transfer`` is ``main.transfer_probe``'s block-size covariate-shift test,
  built exactly as that function builds it (one context in class order,
  then shuffled with the same generator).

Train seeds are ``1000 + i`` and test seeds ``900000 + i`` (12 pairs), the
protocol of ``pipeline.evaluate_classifier``; no window appears in both.
"""

from __future__ import annotations

import random

from ..features import FEATURE_NAMES, WindowContext, dominant_stride
from ..pipeline import BLENDED_REGIMES, REGIMES, blended_regime, make_harder_generator, regime
from ..trace import CLASSES, synthetic_window

N_FEATURES = len(FEATURE_NAMES)
SEED_PAIRS = 12
TRAIN_SEEDS = tuple(1000 + i for i in range(SEED_PAIRS))
TEST_SEEDS = tuple(900000 + i for i in range(SEED_PAIRS))
#: Windows per class: 100 train / 40 test, as ``evaluate_classifier``.
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 40
#: ``main.transfer_probe`` sizes: 60 train / 30 test windows per class.
TRANSFER_TRAIN_PER_CLASS = 60
TRANSFER_TEST_PER_CLASS = 30
TRANSFER_SIZES = (1, 8, 32, 128)
#: Regimes in report order. ``baseline`` is make_model's generator.
REGIME_NAMES = ("baseline", "noisy", "overlap", "overlap2", "realistic",
                "transition", "transition_heavy")
ALL_SETTINGS = REGIME_NAMES + ("transfer",)


def baseline_vectors(seed: int, per_class: int = TRAIN_PER_CLASS
                     ) -> list[tuple[tuple[float, ...], str]]:
    """``make_model(seed, per_class)``'s vectors, in its fitting order."""
    out: list[tuple[tuple[float, ...], str]] = []
    for label in CLASSES:
        rng = random.Random(seed)
        context = WindowContext()
        flat: list = []
        time = 0.0
        for _ in range(per_class):
            window = synthetic_window(label, rng, 32, time)
            flat.extend(window)
            time = window[-1].timestamp_ms
        for start in range(0, len(flat), 32):
            window = flat[start:start + 32]
            if len(window) < 32:
                break
            out.append((context.observe(window, dominant_stride(window)), label))
    return out


def _streamed(examples) -> list[tuple[tuple[float, ...], str]]:
    """Group labelled windows by class (generation order kept) and build
    contextual vectors with one ``WindowContext`` per class stream."""
    by_class: dict[str, list] = {label: [] for label in CLASSES}
    for window, label in examples:
        by_class[label].append(window)
    out = []
    for label in CLASSES:
        context = WindowContext()
        for window in by_class[label]:
            out.append((context.observe(window, dominant_stride(window)), label))
    return out


def regime_vectors(name: str, seed: int, per_class: int
                   ) -> list[tuple[tuple[float, ...], str]]:
    """Contextual vectors of one CLASSIFIER_EVAL regime."""
    if name == "baseline":
        return baseline_vectors(seed, per_class)
    if name in BLENDED_REGIMES:
        blend = 0.5 if name == "transition" else 0.75
        return _streamed(blended_regime(per_class, seed, 32, blend))
    if name in REGIMES:
        return _streamed(regime(name, per_class, seed, 32, **REGIMES[name]))
    raise ValueError(f"unknown regime {name!r}; choose from {REGIME_NAMES}")


def transfer_vectors(size: int, seed: int, per_class: int
                     ) -> list[tuple[tuple[float, ...], str]]:
    """``main.transfer_probe``'s ``build(size, seed, per_class)``, verbatim."""
    rng = random.Random(seed)
    context = WindowContext()
    rows = []
    for label in CLASSES:
        for _ in range(per_class):
            window = make_harder_generator(label, rng, 32, noise=0.15,
                                           lba_base=10**7, lba_span=10**9,
                                           size=size)
            rows.append((context.observe(window, dominant_stride(window)), label))
    rng.shuffle(rows)
    return rows


def train_set(setting: str, seed: int):
    """Training vectors of a setting (``transfer`` trains at size 1)."""
    if setting == "transfer":
        return transfer_vectors(1, seed, TRANSFER_TRAIN_PER_CLASS)
    return regime_vectors(setting, seed, TRAIN_PER_CLASS)


def test_conditions(setting: str) -> tuple[str, ...]:
    """Named test conditions of a setting: ``1->8`` etc. for transfer."""
    if setting == "transfer":
        return tuple(f"1->{size}" for size in TRANSFER_SIZES)
    return ("in_regime",)


def test_set(setting: str, condition: str, seed: int):
    if setting == "transfer":
        size = int(condition.split("->")[1])
        return transfer_vectors(size, seed, TRANSFER_TEST_PER_CLASS)
    return regime_vectors(setting, seed, TEST_PER_CLASS)


def validation_seed(index: int) -> int:
    """Selection fold ``index`` trains on ``TRAIN_SEEDS[index]`` and validates
    on the next training seed (cyclically), so selection never sees a test
    seed and each fold's training size equals the evaluation protocol's."""
    return TRAIN_SEEDS[(index + 1) % SEED_PAIRS]
