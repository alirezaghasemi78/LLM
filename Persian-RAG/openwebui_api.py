
import time
import uuid
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.embedding import EmbeddingModel
from src.vector_store import initialize_vector_store
from src.retrieval import semantic_search
from src.generation import generate_rag_answer


app = FastAPI(title="Persian Wikipedia RAG")

MODEL_ID = "persian-wikipedia-rag"

# Initialize RAG components once
print("Loading embedding model...")
embedding_model = EmbeddingModel()

print("Connecting to Qdrant...")
client = initialize_vector_store()

print("RAG API is ready!")


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-rag"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(body: dict):

    messages = body.get("messages", [])

    user_messages = [
        message.get("content")
        for message in messages
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    ]

    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No valid user message found."
        )

    query = user_messages[-1]

    print(f"\nNew question: {query}")

    try:
        # 1. Retrieve relevant documents
        results = semantic_search(
            query=query,
            embedding_model=embedding_model,
            client=client,
            top_k=5
        )

        # 2. Generate answer
        answer = generate_rag_answer(
            query=query,
            retrieved_documents=results
        )

        # 3. Add retrieved sources
        if results:
            sources = "\n\n### منابع بازیابی‌شده\n"

            for i, result in enumerate(results, start=1):
                sources += (
                    f"\n{i}. **{result['title']}**"
                    f" — Score: {result['score']:.4f}"
                )

            answer += sources

    except Exception as error:
        print(f"RAG Error: {error}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the RAG request."
        )

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # 4. Streaming response
    if body.get("stream", False):

        def generate_stream():

            chunks = [
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None
                    }]
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": answer},
                        "finish_reason": None
                    }]
                },
                {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
            ]

            for chunk in chunks:
                yield (
                    "data: "
                    + json.dumps(chunk, ensure_ascii=False)
                    + "\n\n"
                )

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )

    # 5. Normal response
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop"
        }]
    }
