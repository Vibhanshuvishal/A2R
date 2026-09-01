from a2r.pipelines.rag_pipeline import chunk_text
from a2r.settings import load_config, project_path
from a2r.storage.vector_store import VectorStoreManager


def main():
    config = load_config()
    store = VectorStoreManager(config)
    chunk_words = config["vector_store"]["chunk_words"]
    overlap_words = config["vector_store"]["chunk_overlap_words"]

    for pipeline in config["pipelines"]:
        store.reset_pipeline(pipeline["id"])
        data_dir = project_path(pipeline["data_dir"])
        for file in data_dir.glob("*.md"):
            chunks = chunk_text(file.read_text(encoding="utf-8"), chunk_words, overlap_words)
            store.add_chunks(pipeline["id"], file.name, chunks)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
