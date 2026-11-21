
from src.data_loader import load_wikipedia_dataframe
from src.preprocessing import preprocess_documents
from src.documents import create_documents

from src.chunking import create_chunks

from src.embedding import EmbeddingModel
from src.vector_store import initialize_vector_store
from src.indexing import index_chunks
from src.retrieval import semantic_search


def main():

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = load_wikipedia_dataframe()

    # --------------------------------------------------
    # 2. Preprocessing
    # --------------------------------------------------

    df = preprocess_documents(df)

    # --------------------------------------------------
    # 3. Create documents
    # --------------------------------------------------

    documents = create_documents(df)

    print("\nNumber of documents:", len(documents))

    # --------------------------------------------------
    # 4. Inspect a document
    # --------------------------------------------------

    if documents:

        doc = documents[10] if len(documents) > 10 else documents[0]

        print("\nDOCUMENT CONTENT")
        print("=" * 80)
        print(doc.page_content[:1000])

        print("\nDOCUMENT METADATA")
        print("=" * 80)
        print(doc.metadata)

    # --------------------------------------------------
    # 5. Chunking
    # --------------------------------------------------

    chunk_records = create_chunks(documents)

    print("\nChunking completed!")

    print("Original documents:", len(documents))
    print("Total chunks:", len(chunk_records))

    if chunk_records:

        print("\nExample chunk:")
        print("=" * 80)
        print(chunk_records[0]["text"][:500])

        print("\nExample metadata:")
        print(chunk_records[0]["metadata"])

    # --------------------------------------------------
    # 6. Load embedding model
    # --------------------------------------------------

    embedding_model = EmbeddingModel()

    print("\nEmbedding dimension:", embedding_model.dimension)

    assert embedding_model.dimension == 384

    # --------------------------------------------------
    # 7. Initialize Qdrant
    # --------------------------------------------------

    client = initialize_vector_store()

    # --------------------------------------------------
    # 8. Index chunks
    # --------------------------------------------------

    index_chunks(
        chunk_records=chunk_records,
        embedding_model=embedding_model,
        client=client
    )

    # --------------------------------------------------
    # 9. Test semantic search
    # --------------------------------------------------

    query = "تاریخچه شکل گیری دانشگاه های ایران"

    results = semantic_search(
        query=query,
        embedding_model=embedding_model,
        client=client,
        top_k=5
    )

    # --------------------------------------------------
    # 10. Display results
    # --------------------------------------------------

    for i, result in enumerate(results):

        print("=" * 80)

        print(f"Result {i + 1}")
        print("Title:", result["title"])
        print("Score:", round(result["score"], 4))
        print("Document ID:", result["doc_id"])
        print("Chunk Index:", result["chunk_index"])

        print("\nContent:")
        print(result["text"][:700])


if __name__ == "__main__":
    main()
