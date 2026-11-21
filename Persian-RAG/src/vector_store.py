
from qdrant_client import QdrantClient, models

from src.config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    VECTOR_SIZE
)


def get_qdrant_client():

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    return client


def create_collection(client, vector_size=VECTOR_SIZE):

    if client.collection_exists(COLLECTION_NAME):

        print("Collection already exists!")

    else:

        client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

        print("Collection created!")

    return client.get_collection(COLLECTION_NAME)


def initialize_vector_store():

    client = get_qdrant_client()

    collection_info = create_collection(client)

    print("\nCollection information:")
    print(collection_info)

    return client
