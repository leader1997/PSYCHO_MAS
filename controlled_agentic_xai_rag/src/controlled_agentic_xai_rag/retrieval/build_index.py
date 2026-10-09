"""Build FAISS index CLI."""

from controlled_agentic_xai_rag.retrieval.dense_llama3 import Llama3DenseRetriever


def main() -> None:
    Llama3DenseRetriever()
    print("FAISS index ready.", flush=True)


if __name__ == "__main__":
    main()
