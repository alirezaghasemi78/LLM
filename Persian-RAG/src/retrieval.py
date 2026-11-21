
from src.config import COLLECTION_NAME, TOP_K


def semantic_search(
    query,
    embedding_model,
    client,
    top_k=TOP_K
):

    # Convert query to embedding
    query_vector = embedding_model.embed_query(
        query
    )

    # Search in Qdrant
    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector.tolist(),

        limit=top_k,

        with_payload=True

    ).points

    retrieved_documents = []

    for result in results:

        payload = result.payload or {}

        item = {
            "score": result.score,
            "text": payload.get("text", ""),
            "title": payload.get("title", ""),
            "doc_id": payload.get("doc_id"),
            "chunk_index": payload.get("chunk_index")
        }

        retrieved_documents.append(item)

    return retrieved_documents
