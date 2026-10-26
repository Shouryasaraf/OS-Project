"""Validation-selected policies for the live run, read from E4's results.

E6 replays the controllers that the replay benchmark (E4) chose on
validation, **per capacity**, never a hand-picked parameter. A policy name
``validated_<family>`` (optionally ``@<scorer>``, ``hint:`` in front) is a
placeholder that :func:`resolve` replaces, at plan time, with the policy E4
selected for that scorer and family at the live cache capacity and the
chosen cost, for example ``validated_feedback`` -> ``feedback_0.01_eps0.002``
(the gate's break-even c *and* its exploration rate, both validated).

The source is ``results/web/comparisons.csv`` (schema ``web/comparisons.csv``,
rows ``split=test, selection=per_capacity``): each row names the validated
controller of one family (``adaptive_policy``) and the validated best fixed
depth (``best_fixed_policy``) for one (scorer, capacity, cost). A missing
file, or a missing (scorer, family, capacity, cost), is an explicit
:class:`SelectionError`; there is no built-in fallback. Concrete policy
names (``fixed4``, ``adaptive_0.01``) never need the file, which is what
the fixture tests and smoke runs use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PREFIX = "validated_"
#: ``validated_fixed`` is the best fixed depth; the others are controllers.
FAMILIES = ("fixed", "adaptive", "adaptive_ca", "gate", "feedback")
DEFAULT_SOURCE = Path("results/web/comparisons.csv")
#: The cost E6 takes its controllers at: the pivot's headline cost and the
#: one E4's decomposition compares controllers at (benchmark.DECOMPOSITION_COST).
DEFAULT_COST = 0.01
HINT = "hint:"


class SelectionError(RuntimeError):
    """A validated policy was asked for and E4's selection cannot supply it."""


def _split(label: str) -> tuple[str, str, str]:
    """``hint:validated_adaptive@trans_online`` -> (``hint:``, base, ``@trans_online``)."""
    hint = HINT if label.startswith(HINT) else ""
    base, at, scorer = label[len(hint):].partition("@")
    return hint, base, (at + scorer) if at else ""


def is_placeholder(label: str) -> bool:
    return _split(label)[1].startswith(PREFIX)


def needs_selection(labels) -> bool:
    return any(is_placeholder(label) for label in labels)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selection(path: str | Path, *, capacity: int, cost: float = DEFAULT_COST) -> dict:
    """``{(scorer, family): policy}`` validated at ``capacity`` and ``cost``."""
    from prefetch_common.schemas import read_rows
    from .plan import parse_policy
    path = Path(path)
    if not path.is_file():
        raise SelectionError(
            f"{path} is missing. Validated live policies are read from E4's per-capacity "
            "selection (`reproduce.py web` writes results/web/comparisons.csv); there is no "
            "built-in fallback. Run E4 first, pass --selection <comparisons.csv>, or name "
            "concrete policies instead of validated_*.")
    out: dict[tuple[str, str], str] = {}
    for row in read_rows("web/comparisons.csv", path):
        if (row["split"] != "test" or row["selection"] != "per_capacity"
                or int(row["capacity"]) != int(capacity) or abs(row["cost"] - cost) > 1e-12):
            continue
        for name in (row["adaptive_policy"], row["best_fixed_policy"]):
            key = (row["scorer"], parse_policy(name).policy.kind)
            if out.setdefault(key, name) != name:
                raise SelectionError(f"{path}: two validated {key[1]} policies for scorer "
                                     f"{key[0]} at capacity {capacity}, cost {cost:g}: "
                                     f"{out[key]} and {name}")
    return out


def resolve(labels, *, source: str | Path = DEFAULT_SOURCE, capacity: int,
            cost: float = DEFAULT_COST, default_scorer: str = "logistic") -> tuple[list[str], dict | None]:
    """Replace every ``validated_*`` label; returns (labels, record or None).

    The record (source, its SHA-256, capacity, cost, placeholder -> policy)
    goes into ``live_config.json``. Labels without a placeholder come back
    unchanged and the source is never opened.
    """
    labels = list(labels)
    if not needs_selection(labels):
        return labels, None
    source = Path(source)
    selection = load_selection(source, capacity=capacity, cost=cost)
    resolved, mapping = [], {}
    for label in labels:
        hint, base, scorer = _split(label)
        if not base.startswith(PREFIX):
            resolved.append(label)
            continue
        family = base[len(PREFIX):]
        if family not in FAMILIES:
            raise SelectionError(f"unknown family in {label!r}; expected validated_<family> "
                                 f"with family one of {FAMILIES}")
        scorer_name = scorer[1:] or default_scorer
        name = selection.get((scorer_name, family))
        if name is None:
            raise SelectionError(
                f"{source} has no validated {family} policy for scorer {scorer_name} at "
                f"capacity {capacity}, cost {cost:g} (E4 must replay that scorer; "
                f"available: {sorted(selection)})")
        mapping[label] = hint + name + scorer
        resolved.append(mapping[label])
    if len(set(resolved)) != len(resolved):
        raise SelectionError(f"validated policies collide with listed ones: {resolved}")
    return resolved, {"source": str(source), "sha256": _sha256(source),
                      "rows": "split=test, selection=per_capacity",
                      "capacity": int(capacity), "cost": float(cost), "resolved": mapping}
