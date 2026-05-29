"""Train the candidate-as-row listwise ranker from cached records."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.array_search.labels import label_records
from src.array_search.ranker import CandidateRanker, save_ranker, train_ranker_epoch
from src.array_search.records import load_record_shards


def collect_shards(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("records_*.pkl")))
        else:
            paths.extend(sorted(path.parent.glob(path.name)))
    return sorted(dict.fromkeys(p.resolve() for p in paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--pairwise-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_record_shards(collect_shards(args.records))
    label_records(records, primary_horizon=args.horizon)
    model = CandidateRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(records)
        stats = train_ranker_epoch(
            model,
            records,
            optimizer,
            horizon=args.horizon,
            pairwise_weight=args.pairwise_weight,
            device=args.device,
        )
        print(
            f"epoch={epoch} loss={stats.loss:.4f} "
            f"list={stats.loss_listwise:.4f} pair={stats.loss_pairwise:.4f} "
            f"records={stats.records_used} mean_k={stats.mean_candidates_per_turn:.1f}"
        )
    save_ranker(args.out, model, extra={"epochs": args.epochs, "horizon": args.horizon})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
