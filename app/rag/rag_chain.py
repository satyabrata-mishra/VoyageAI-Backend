from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

prompt = ChatPromptTemplate.from_template(
    """
    User Query:
    {user_query}

    Context:
    {context}

    Generate a travel recommendation.
    """
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

rag_chain = prompt | llm