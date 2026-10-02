
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def _preprocess(self, text: str) -> str:
        return " ".join(text.strip().split())

    def index(self, documents):
        self.documents = [self._preprocess(d) for d in documents if d.strip()]
        print(f"Encoding {len(self.documents)} documents ...")
        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        print(f"Embedding shape: {self.embeddings.shape}")

    def search(self, query: str, top_k: int = 3):
        if self.embeddings is None:
            raise RuntimeError("Call index() before search().")

        query = self._preprocess(query)
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = cosine_similarity(query_vec, self.embeddings)[0]
        ranked_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {"document": self.documents[i], "similarity": float(scores[i])}
            for i in ranked_idx
        ]


def load_documents(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    docs = load_documents("documents.txt")

    engine = SemanticSearch()
    engine.index(docs)

    print("\n" + "=" * 60)
    print("Semantic Search Ready")
    print(f"Indexed {len(docs)} documents.")
    print("Type a query and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye LOL.")
            break

        results = engine.search(query, top_k=3)
        print("-" * 60)
        for rank, r in enumerate(results, start=1):
            print(f"{rank}. [{r['similarity']:.4f}] {r['document']}")


if __name__ == "__main__":
    main()