#!/usr/bin/env python3
"""Measure product-level Recall@K and MRR from a CSV of warehouse photos."""
import argparse
import csv
import json
import mimetypes
import time
import urllib.request
from pathlib import Path

from index_manifest import encode_form


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path,
                        help="CSV columns: query_path,expected_product_id[,category]")
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    ranks: list[int | None] = []
    latencies: list[float] = []
    failures: list[dict[str, object]] = []
    with args.dataset.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            query_path = Path(row["query_path"])
            body, content_type = encode_form(
                {"top_k": str(args.top_k), "category": row.get("category", "")}, query_path)
            request = urllib.request.Request(
                f"{args.url.rstrip('/')}/v1/search", data=body, method="POST",
                headers={"Content-Type": content_type, "X-API-Key": args.api_key},
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            latencies.append((time.perf_counter() - started) * 1000)
            ids = [item["product_id"] for item in result["matches"]]
            expected = row["expected_product_id"]
            rank = ids.index(expected) + 1 if expected in ids else None
            ranks.append(rank)
            if rank is None or rank > 1:
                failures.append({"query_path": str(query_path), "expected": expected,
                                 "rank": rank, "returned": ids})

    count = len(ranks)
    if not count:
        raise SystemExit("评测集为空")
    report = {
        "queries": count,
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / count,
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks if rank is not None) / count,
        "latency_ms_mean": sum(latencies) / count,
        "latency_ms_p95": sorted(latencies)[min(count - 1, int(count * 0.95))],
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
