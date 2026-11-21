from src.embedding import EmbeddingModel
from src.vector_store import initialize_vector_store
from src.retrieval import semantic_search
from src.generation import generate_rag_answer
import fa_console


def main():

    # 1. Load embedding model
    embedding_model = EmbeddingModel()

    # 2. Connect to Qdrant
    client = initialize_vector_store()

    # 3. Get user query
    query = "تاریخچه شکل گیری دانشگاه های ایران"

    # 4. Retrieve relevant documents
    results = semantic_search(
        query=query,
        embedding_model=embedding_model,
        client=client,
        top_k=5
    )

    # 5. Generate answer using LLM
    answer = generate_rag_answer(
        query=query,
        retrieved_documents=results
    )

    # 6. Display answer
    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print("GENERATED ANSWER")
    print("=" * 80)
    print(answer)

    # 7. Display retrieved sources
    print("\n" + "=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\nSource {i}")
        print("Title:", result["title"])
        print("Score:", round(result["score"], 4))
        print("Document ID:", result["doc_id"])
        print("Chunk Index:", result["chunk_index"])


if __name__ == "__main__":
    main()