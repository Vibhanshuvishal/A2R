from a2r.graph import build_engine
from a2r.settings import load_config, project_path
from tests.conftest import FakeVectorStore
import json
import time


def main():
    config = load_config()
    data = json.loads(project_path("evals/query_set.json").read_text())
    chunks = {
        "billing": [{"text": "Refund requests are processed in thirty days.", "source": "billing.md", "chunk_index": 0, "score": 0.9}],
        "product": [{"text": "Rate limits allow 120 requests per minute.", "source": "product.md", "chunk_index": 0, "score": 0.9}],
        "hr": [{"text": "Core hours are 10am to 4pm.", "source": "hr.md", "chunk_index": 0, "score": 0.9}],
    }
    engine = build_engine(config, vector_store=FakeVectorStore(chunks))
    correct = 0
    total = len(data)
    latencies = []

    for item in data:
        start = time.perf_counter()
        result = engine.query(item["query"])
        latencies.append(time.perf_counter() - start)
        predicted = result["pipeline_used"] if result["answer_source"] == "rag_pipeline" else "out_of_scope"
        if predicted == item["expected"]:
            correct += 1

    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    print(f"Accuracy: {correct / total:.2%} ({correct}/{total})")
    print(f"Median latency: {median_latency * 1000:.1f}ms")


if __name__ == "__main__":
    main()
