
import uuid

from tqdm.auto import tqdm
from qdrant_client import models

from src.config import (
    COLLECTION_NAME,
    BATCH_SIZE
)


def index_chunks(
    chunk_records,
    embedding_model,
    client
):

    total_chunks = len(chunk_records)

    print("Total chunks:", total_chunks)

    for start in tqdm(
        range(0, total_chunks, BATCH_SIZE)
    ):

        batch = chunk_records[
            start:start + BATCH_SIZE
        ]

        # Extract chunk texts
        texts = [
            item["text"]
            for item in batch
        ]

        # Generate embeddings
        vectors = embedding_model.embed_documents(
            texts
        )

        points = []

        for i, item in enumerate(batch):

            metadata = item["metadata"]

            # Deterministic ID
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"wiki:{metadata['doc_id']}:{metadata['chunk_index']}"
                )
            )

            payload = {
                "text": item["text"],
                **metadata
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vectors[i].tolist(),
                    payload=payload
                )
            )

        # Upload batch to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

    print("\nIndexing completed!")

    total_points = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    ).count

    print("Total points in Qdrant:", total_points)
