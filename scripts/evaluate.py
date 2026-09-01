"""Run a versioned, local-only evaluation after ingesting the bundled data."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from a2r.graph import build_engine


def main() -> None:
    cases = json.loads((Path(__file__).parents[1] / "evals" / "query_set.json").read_text(encoding="utf-8"))
    engine = build_engine()
    engine.ensure_ingested()
    outcomes, latencies = [], []
    for case in cases:
        started = time.perf_counter()
        result = engine.query(case["query"])
        latencies.append((time.perf_counter() - started) * 1000)
        correct_domain = result["domain"] == case["domain"]
        correct_source = (result["pipeline_used"] == case["pipeline"]) if case["answerable"] else result["answer_source"] == "out_of_scope"
        outcomes.append({"query": case["query"], "domain_ok": correct_domain, "route_ok": correct_source, "latency_ms": round(latencies[-1], 1)})
    summary = {
        "cases": len(cases),
        "domain_accuracy": round(sum(item["domain_ok"] for item in outcomes) / len(outcomes), 3),
        "route_or_out_of_scope_accuracy": round(sum(item["route_ok"] for item in outcomes) / len(outcomes), 3),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "results": outcomes,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
