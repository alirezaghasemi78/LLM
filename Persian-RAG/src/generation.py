from src.llm import GeminiLLM


def build_prompt(query, retrieved_documents):

    context_parts = []

    for i, document in enumerate(retrieved_documents, start=1):

        context = f"""
[Source {i}]
Title: {document["title"]}
Document ID: {document["doc_id"]}
Chunk ID: {document["chunk_index"]}

Content:
{document["text"]}
"""

        context_parts.append(context)

    context = "\n\n".join(context_parts)

    prompt = f"""
تو یک دستیار پرسش و پاسخ فارسی هستی.

وظیفه تو پاسخ دادن به سؤال کاربر با استفاده از
اطلاعات موجود در بخش منابع است.

قوانین:
1. پاسخ را به زبان فارسی روان بنویس.
2. فقط از اطلاعات موجود در منابع استفاده کن.
3. اطلاعاتی را که در منابع وجود ندارند حدس نزن.
4. اگر منابع برای پاسخ کافی نیستند، صریحاً اعلام کن.
5. برای ادعاهای مهم، شماره منبع را مانند [Source 1] ذکر کن.
6. پاسخ را منظم و متناسب با سؤال بنویس.

Sources:
{context}

User Question:
{query}

Answer:
"""

    return prompt


def generate_rag_answer(query, retrieved_documents):

    if not retrieved_documents:
        return "اطلاعات مرتبطی در پایگاه دانش پیدا نشد."

    llm = GeminiLLM()

    prompt = build_prompt(
        query=query,
        retrieved_documents=retrieved_documents
    )

    answer = llm.generate(prompt)

    return answer