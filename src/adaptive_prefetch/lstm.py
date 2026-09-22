"""Optional offline LSTM baseline for next-delta prediction.

This is a different prediction task from four-class workload classification.
Compare end-to-end cache outcomes and inference cost, not class accuracy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .trace import Request, transition_trace

HISTORY = 16
MAX_DELTA = 4096


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LSTM requires PyTorch: pip install -e .[lstm]") from exc
    return torch


def _network(vocab_size: int, hidden_size: int = 64):
    torch = _torch()

    class DeltaLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, hidden_size, batch_first=True)
            self.output = torch.nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            sequence, _ = self.lstm(x)
            return self.output(sequence[:, -1, :])

    return DeltaLSTM()


def _read_deltas(requests: list[Request]) -> list[int]:
    reads = [req.lba for req in requests if req.operation == "R"]
    return [b - a for a, b in zip(reads, reads[1:])]


def train_lstm(
    save_path: str | Path,
    traces: list[list[Request]] | None = None,
    epochs: int = 15,
    vocab_size: int = 256,
    seed: int = 42,
) -> dict[str, int]:
    torch = _torch()
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if traces is None:
        traces = [transition_trace(seed + i, windows_per_class=8)[0] for i in range(4)]

    all_deltas = [_read_deltas(trace) for trace in traces]

    # Index 0 is reserved for unknown/out-of-vocab deltas
    common_deltas = [
        delta
        for delta, _ in Counter(
            delta
            for stream in all_deltas
            for delta in stream
            if delta != 0 and -MAX_DELTA <= delta <= MAX_DELTA
        ).most_common(vocab_size)
    ]
    vocabulary = [0] + common_deltas
    targets = {delta: index for index, delta in enumerate(vocabulary)}

    examples = []
    for deltas in all_deltas:
        for i in range(HISTORY, len(deltas)):
            x = [
                max(-MAX_DELTA, min(MAX_DELTA, value)) / MAX_DELTA
                for value in deltas[i - HISTORY : i]
            ]
            examples.append((x, targets.get(deltas[i], 0)))

    if not examples:
        raise ValueError("need at least 18 read requests in training traces")

    torch.manual_seed(seed)
    network = _network(len(vocabulary))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()

    inputs = torch.tensor([x for x, _ in examples], dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([y for _, y in examples], dtype=torch.long)

    network.train()
    for _ in range(epochs):
        for start in range(0, len(examples), 128):
            optimizer.zero_grad()
            loss = loss_fn(
                network(inputs[start : start + 128]),
                labels[start : start + 128],
            )
            loss.backward()
            optimizer.step()

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "vocabulary": vocabulary,
            "hidden_size": 64,
            "history": HISTORY,
            "max_delta": MAX_DELTA,
        },
        destination,
    )

    return {
        "examples": len(examples),
        "vocabulary": len(vocabulary),
        "epochs": epochs,
    }


class LSTMPrefetcher:
    def __init__(self, network, vocabulary: list[int]):
        self.network = network
        self.vocabulary = vocabulary
        self.last_lba: int | None = None
        self.deltas: list[int] = []

    def next_candidates(self, request: Request) -> list[int]:
        if request.operation != "R":
            return []

        if self.last_lba is not None:
            self.deltas.append(request.lba - self.last_lba)
        self.last_lba = request.lba

        if len(self.deltas) < HISTORY:
            return []

        torch = _torch()
        values = [
            max(-MAX_DELTA, min(MAX_DELTA, d)) / MAX_DELTA
            for d in self.deltas[-HISTORY:]
        ]
        x = torch.tensor(values, dtype=torch.float32).reshape(1, HISTORY, 1)

        with torch.no_grad():
            scores = self.network(x).squeeze(0)
            # Inspect top 10 scores instead of top 2 to bypass unk/Class 0 predictions
            top_indices = torch.topk(
                scores, min(10, len(self.vocabulary))
            ).indices.tolist()

        end = request.lba + request.size_blocks
        candidates = []

        for index in top_indices:
            delta = self.vocabulary[index]
            # Valid positive delta targeting blocks ahead of current request
            if delta > 0 and (request.lba + delta) >= end:
                candidates.append(request.lba + delta)
                if len(candidates) == 2:  # Limit prefetch depth to 2 blocks
                    break

        return candidates


def load_predictor(path: str | Path) -> LSTMPrefetcher:
    torch = _torch()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LSTM artifact not found: {source}")

    artifact = torch.load(source, map_location="cpu", weights_only=True)
    if (
        artifact.get("history") != HISTORY
        or artifact.get("max_delta") != MAX_DELTA
    ):
        raise ValueError("unsupported LSTM artifact configuration")

    vocabulary = artifact["vocabulary"]
    network = _network(len(vocabulary), artifact["hidden_size"])
    network.load_state_dict(artifact["state_dict"])
    network.eval()
    return LSTMPrefetcher(network, vocabulary)