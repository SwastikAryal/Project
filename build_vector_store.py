import json
import math
import os
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ManualVectorStoreBuilder:
    """Builds vector store with bilingual support. No ChromaDB - Pure Python + JSON."""

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.documents: List[Dict] = []

    def _compute_norm(self, vector: List[float]) -> float:
        return math.sqrt(sum(x * x for x in vector))

    def process_chunks(self, chunks_path: str):
        """Process chunks from code_chunks.json - include ALL chunks."""
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Generating embeddings for {len(chunks)} chunks...")
        print(f"  Model: paraphrase-multilingual-MiniLM-L12-v2")
        print(f"  Dimension: {self.dimension}")

        for i, chunk in enumerate(chunks):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(chunks)}")

            metadata = chunk.get("metadata", {})
            text = chunk.get("text", "").strip()

            if not text:
                continue

            try:
                embedding = self.model.encode(text).tolist()

                doc = {
                    "id": str(chunk.get("chunk_id", i)),
                    "text": text,
                    "section": metadata.get("section", "Unknown"),
                    "page": metadata.get("page", "Unknown"),
                    "doc_name": metadata.get("doc", "Unknown"),
                    "language": metadata.get("language", "unknown"),
                    "embedding": embedding,
                    "norm": self._compute_norm(embedding)
                }
                self.documents.append(doc)
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {e}")
                continue

        eng = sum(1 for d in self.documents if d["language"] == "english")
        nep = sum(1 for d in self.documents if d["language"] == "nepali")
        print(f"Done: {len(self.documents)} documents ({eng} English, {nep} Nepali)")

    def save(self, output_path: str = "manual_vectors.json"):
        """Save to JSON."""
        data = {
            "metadata": {
                "model": "paraphrase-multilingual-MiniLM-L12-v2",
                "dimension": self.dimension,
                "total_documents": len(self.documents),
                "english_docs": sum(1 for d in self.documents if d["language"] == "english"),
                "nepali_docs": sum(1 for d in self.documents if d["language"] == "nepali"),
                "storage_type": "manual_json",
                "algorithm": "cosine_similarity_manual"
            },
            "documents": self.documents
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Saved to {output_path} ({size_mb:.2f} MB)")

    def build(self, chunks_path: str = "code_chunks.json",
              output_path: str = "manual_vectors.json"):
        """Full build pipeline."""
        self.process_chunks(chunks_path)
        self.save(output_path)
        return self


if __name__ == "__main__":
    builder = ManualVectorStoreBuilder()
    builder.build("code_chunks.json", "manual_vectors.json")
    print("\nVector store built!")
