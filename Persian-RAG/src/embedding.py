
from sentence_transformers import SentenceTransformer

from src.config import (
    MODEL_NAME,
    MODEL_DEVICE,
    MAX_SEQ_LENGTH
)


class EmbeddingModel:

    def __init__(self):

        print(f"Loading embedding model: {MODEL_NAME}")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=MODEL_DEVICE
        )

        self.model.max_seq_length = MAX_SEQ_LENGTH

        self.dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        print("Embedding model loaded!")
        print("Vector dimension:", self.dimension)

    def embed_documents(self, texts):

        # E5 requires the passage prefix for documents
        passages = [
            f"passage: {text.strip()}"
            for text in texts
        ]

        embeddings = self.model.encode(
            passages,
            normalize_embeddings=True,
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings

    def embed_query(self, query):

        # E5 requires the query prefix for search queries
        query_text = f"query: {query.strip()}"

        embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embedding[0]
