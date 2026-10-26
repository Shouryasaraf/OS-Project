"""Regenerate ``web_fixture.csv``: a tiny SYNTHETIC clickstream for web tests.

It has the RetailRocket ``events.csv`` header and value conventions
(millisecond timestamps, ``view``/``addtocart``/``transaction`` events,
transaction IDs only on transactions) but contains no RetailRocket data and
must never be used as research data. ``webprefetch prepare`` rejects it on
purpose, because it checks the real file's SHA-256 and row count.

Stdlib only and seeded, so ``python data/samples/make_web_fixture.py`` always
writes the same bytes; ``tests/test_web_fixture.py`` checks that.
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

HEADER = ("timestamp", "visitorid", "event", "itemid", "transactionid")
TARGET = Path(__file__).resolve().with_name("web_fixture.csv")
SEED = 42
START_MS = 1_433_221_332_117  # same epoch range as the real log (June 2015)
DAY_MS = 24 * 60 * 60 * 1000
ITEMS = tuple(range(100, 124))  # 24 products
# Each product has two likely successors, so transitions repeat across
# sessions and a candidate index built from early sessions has something to
# retrieve later.
SUCCESSORS = {item: (ITEMS[(i + 1) % len(ITEMS)], ITEMS[(i + 5) % len(ITEMS)])
              for i, item in enumerate(ITEMS)}


def generate(seed: int = SEED, rows: int = 200) -> str:
    rng = random.Random(seed)
    events: list[tuple[int, int, str, int, str]] = []
    transaction = 1
    visitor = 0
    while len(events) < rows:
        visitor += 1
        clock = START_MS + rng.randrange(0, 20 * DAY_MS)
        # One or two sessions per visitor; the second starts after a gap
        # longer than the 30-minute session cut.
        for _ in range(rng.choice((1, 1, 2))):
            item = rng.choice(ITEMS)
            carted = None
            for _ in range(rng.randint(1, 8)):
                events.append((clock, visitor, "view", item, ""))
                if carted is None and rng.random() < 0.15:
                    clock += rng.randint(2_000, 30_000)
                    events.append((clock, visitor, "addtocart", item, ""))
                    carted = item
                clock += rng.randint(5_000, 240_000)
                if rng.random() < 0.2:
                    continue  # a reload of the same product page
                if rng.random() < 0.75:
                    item = rng.choice(SUCCESSORS[item])
                else:
                    item = rng.choice(ITEMS)
            if carted is not None and rng.random() < 0.5:
                events.append((clock, visitor, "transaction", carted, str(transaction)))
                transaction += 1
            clock += rng.randint(31, 600) * 60 * 1000
    events = events[:rows]
    # The real file is not globally time-ordered either; ordering by time
    # here keeps the fixture readable, and prepare() sorts regardless.
    events.sort(key=lambda row: (row[0], row[1]))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(events)
    return buffer.getvalue()


if __name__ == "__main__":
    TARGET.write_text(generate(), encoding="utf-8", newline="")
    print(f"wrote {TARGET}")
