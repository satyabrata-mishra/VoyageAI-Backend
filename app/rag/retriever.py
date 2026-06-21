from app.rag.vector_store import get_vector_store


def get_retrieval_context(
    query: str,
    k: int = 3,
    vector_store=None,
) -> str:
    if vector_store is None:
        vector_store = get_vector_store()

    docs = vector_store.similarity_search(query, k=k)

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        city = doc.metadata.get("city")
        source = doc.metadata.get("source")

        context = f"""
Context {i}
City: {city}
Source: {source}
Content:
{doc.page_content}
"""
        context_parts.append(context.strip())

    return "\n\n".join(context_parts)