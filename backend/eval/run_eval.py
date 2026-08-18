"""Measure the retrieval pipeline instead of eyeballing it.

Relevance is expressed as a predicate over document metadata rather than a frozen
list of ids, so the golden set survives re-ingestion and dataset changes.

Reported metrics:
  hit_rate@k   fraction of questions with at least one relevant document in top-k
  precision@k  fraction of returned documents that are relevant
  MRR          1 / rank of the first relevant document, averaged

Recall is deliberately not reported: several questions have hundreds of relevant
records, so recall@6 would be near zero for reasons that say nothing about quality.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document  # noqa: E402

from backend.app.core.logging import get_logger  # noqa: E402
from backend.app.rag.retriever import retrieve  # noqa: E402

log = get_logger("eval")

GOLDEN_SET = Path(__file__).parent / "golden_set.json"
MODES = ["vector", "hybrid", "hybrid_rerank"]

REFUSAL_MARKERS = (
    "could not find", "don't have", "do not have", "no record", "not contain",
    "unable to", "no information", "cannot find", "not available",
)


def _scalar_match(actual, expected) -> bool:
    if isinstance(expected, dict):
        for op, bound in expected.items():
            if actual is None:
                return False
            if op == "gte" and not actual >= bound:
                return False
            if op == "lte" and not actual <= bound:
                return False
            if op == "gt" and not actual > bound:
                return False
            if op == "lt" and not actual < bound:
                return False
        return True
    if isinstance(actual, list):
        return expected in actual
    return actual == expected


def is_relevant(doc: Document, spec: dict) -> bool:
    meta = doc.metadata
    for key, expected in spec.items():
        if key.endswith("_contains_any"):
            field = meta.get(key.removesuffix("_contains_any")) or ""
            if not any(token.lower() in str(field).lower() for token in expected):
                return False
        elif key.endswith("_contains"):
            values = meta.get(key.removesuffix("_contains")) or []
            if expected not in values:
                return False
        elif key.endswith("_in"):
            if meta.get(key.removesuffix("_in")) not in expected:
                return False
        elif key == "year" and isinstance(meta.get("year"), list):
            years = meta["year"]
            if isinstance(expected, dict):
                if not any(_scalar_match(y, expected) for y in years):
                    return False
            elif expected not in years:
                return False
        elif not _scalar_match(meta.get(key), expected):
            return False
    return True


def evaluate_mode(cases: list[dict], mode: str, k: int) -> dict:
    per_case, per_category = [], defaultdict(list)

    for case in cases:
        started = time.perf_counter()
        result = retrieve(case["question"], mode=mode, k=k)
        elapsed = (time.perf_counter() - started) * 1000
        docs = result.documents

        if case.get("expect_refusal"):
            # Nothing in the corpus answers these, so success means the filters and
            # search did not manufacture confident-looking context.
            row = {
                "id": case["id"],
                "category": case["category"],
                "hit": None,
                "precision": None,
                "rr": None,
                "latency_ms": round(elapsed),
                "retrieved": len(docs),
            }
        else:
            flags = [is_relevant(d, case["relevant"]) for d in docs]
            first = next((i for i, f in enumerate(flags) if f), None)
            row = {
                "id": case["id"],
                "category": case["category"],
                "hit": bool(first is not None),
                "precision": sum(flags) / len(flags) if flags else 0.0,
                "rr": 1 / (first + 1) if first is not None else 0.0,
                "latency_ms": round(elapsed),
                "retrieved": len(docs),
                "filters": result.parsed.filters,
            }
            per_category[case["category"]].append(row)

        per_case.append(row)
        status = "-" if row["hit"] is None else ("ok " if row["hit"] else "MISS")
        log.info("  %-10s %s p@k=%s  %sms", case["id"], status,
                 "n/a" if row["precision"] is None else f"{row['precision']:.2f}", row["latency_ms"])

    scored = [r for r in per_case if r["hit"] is not None]
    return {
        "mode": mode,
        "k": k,
        "questions": len(scored),
        "hit_rate": statistics.mean(r["hit"] for r in scored),
        "precision": statistics.mean(r["precision"] for r in scored),
        "mrr": statistics.mean(r["rr"] for r in scored),
        "median_latency_ms": statistics.median(r["latency_ms"] for r in per_case),
        "by_category": {
            cat: {
                "n": len(rows),
                "hit_rate": statistics.mean(r["hit"] for r in rows),
                "precision": statistics.mean(r["precision"] for r in rows),
                "mrr": statistics.mean(r["rr"] for r in rows),
            }
            for cat, rows in sorted(per_category.items())
        },
        "cases": per_case,
    }


def markdown_table(summaries: list[dict]) -> str:
    lines = [
        "| Configuration | hit-rate@k | precision@k | MRR | median latency |",
        "| --- | --- | --- | --- | --- |",
    ]
    labels = {
        "vector": "Vector only (cosine)",
        "hybrid": "+ BM25 hybrid (RRF)",
        "hybrid_rerank": "+ cross-encoder rerank",
    }
    for s in summaries:
        lines.append(
            f"| {labels.get(s['mode'], s['mode'])} | {s['hit_rate']:.3f} | "
            f"{s['precision']:.3f} | {s['mrr']:.3f} | {s['median_latency_ms']:.0f} ms |"
        )
    return "\n".join(lines)


def category_table(summaries: list[dict]) -> str:
    categories = sorted({c for s in summaries for c in s["by_category"]})
    header = "| Question type | " + " | ".join(s["mode"] for s in summaries) + " |"
    lines = [header, "| --- |" + " --- |" * len(summaries)]
    for cat in categories:
        cells = [f"{s['by_category'].get(cat, {}).get('hit_rate', 0):.2f}" for s in summaries]
        n = next(s["by_category"][cat]["n"] for s in summaries if cat in s["by_category"])
        lines.append(f"| {cat} (n={n}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval configurations")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--all-configs", action="store_true")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    modes = MODES if args.all_configs else args.modes
    cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    log.info("%d questions, modes: %s, k=%d", len(cases), ", ".join(modes), args.k)

    summaries = []
    for mode in modes:
        log.info("")
        log.info("=== %s ===", mode)
        summaries.append(evaluate_mode(cases, mode, args.k))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    report = f"{markdown_table(summaries)}\n\nHit-rate by question type:\n\n{category_table(summaries)}\n"
    (args.out / "report.md").write_text(report, encoding="utf-8")

    print("\n" + report)
    log.info("wrote %s", args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
