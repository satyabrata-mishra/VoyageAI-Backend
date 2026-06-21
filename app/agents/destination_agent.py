from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config.settings import settings
from app.rag.vector_store import get_vector_store
from app.rag.retriever import get_retrieval_context
from app.utils.destination_utils import (
    get_available_destinations,
    is_destination_in_kb,
    format_available_destinations,
)
from app.utils.json_utils import parse_json_response


DESTINATION_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are the Destination Agent of VoyageAI.

Your task is to recommend the best destination based only on the available VoyageAI knowledge base.

Available destinations in the knowledge base:
{available_destinations}

User Travel Request:
{user_query}

Retrieved Travel Knowledge:
{context}

Return your answer strictly in valid JSON format.

JSON structure:
{{
  "agent_name": "Destination Agent",
  "status": "success/out_of_knowledge_base",
  "recommended_destination": "destination name or null",
  "reason": "short reason",
  "suitable_for": ["type 1", "type 2", "type 3"],
  "suggested_duration": "duration from retrieved knowledge or null",
  "best_time_to_visit": "best time from retrieved knowledge or null",
  "key_attractions": ["attraction 1", "attraction 2", "attraction 3"],
  "confidence": "high/medium/low"
}}

Rules:
- You are allowed to recommend ONLY destinations from the available destinations list.
- Do not use your general knowledge.
- Do not invent destinations outside the available destinations list.
- If the user explicitly asks for a destination that is not in the available destinations list, set status to "out_of_knowledge_base".
- If status is "out_of_knowledge_base", set recommended_destination to null.
- If the retrieved knowledge does not support the answer, set confidence to "low".
- Do not add markdown.
- Do not wrap JSON in ```json.
""")


class DestinationAgent:
    def __init__(self):
        self.vector_store = get_vector_store()

        self.available_destinations = get_available_destinations(
            self.vector_store
        )

        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found. Please check your .env file."
            )

        self.llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            temperature=0.2,
            groq_api_key=settings.GROQ_API_KEY,
        )

        self.chain = (
            DESTINATION_AGENT_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def run(self, user_query: str, k: int = 3) -> dict:
        context = get_retrieval_context(
            query=user_query,
            k=k,
            vector_store=self.vector_store,
        )

        response = self.chain.invoke(
            {
                "user_query": user_query,
                "context": context,
                "available_destinations": format_available_destinations(
                    self.available_destinations
                ),
            }
        )

        try:
            parsed_response = parse_json_response(response)
        except Exception:
            return {
                "agent_name": "Destination Agent",
                "status": "error",
                "recommended_destination": None,
                "reason": "Failed to parse agent response.",
                "suitable_for": [],
                "suggested_duration": None,
                "best_time_to_visit": None,
                "key_attractions": [],
                "confidence": "low",
                "raw_response": response,
            }

        recommended_destination = parsed_response.get(
            "recommended_destination"
        )

        if recommended_destination and not is_destination_in_kb(
            recommended_destination,
            self.available_destinations,
        ):
            return {
                "agent_name": "Destination Agent",
                "status": "out_of_knowledge_base",
                "recommended_destination": None,
                "reason": (
                    f"{recommended_destination} is not available in the "
                    "current VoyageAI knowledge base."
                ),
                "suitable_for": [],
                "suggested_duration": None,
                "best_time_to_visit": None,
                "key_attractions": [],
                "available_destinations": self.available_destinations,
                "confidence": "low",
            }

        return parsed_response


def run_destination_agent(user_query: str, k: int = 3) -> dict:
    agent = DestinationAgent()
    return agent.run(user_query=user_query, k=k)

if __name__ == "__main__":
    import json

    test_queries = [
        {
            "name": "Adventure beach trip",
            "query": "I want to do scuba diving, island hopping and enjoy clean beaches",
            "k": 3,
        },
        {
            "name": "Romantic luxury trip",
            "query": "I want a romantic luxury trip with lakes and palaces",
            "k": 3,
        },
        {
            "name": "Out of knowledge base destination",
            "query": "I want to visit Venice for canals and Italian food",
            "k": 3,
        },
    ]

    destination_agent = DestinationAgent()

    for test_case in test_queries:
        print("=" * 100)
        print("Test Case:", test_case["name"])
        print("User Query:", test_case["query"])
        print("-" * 100)

        result = destination_agent.run(
            user_query=test_case["query"],
            k=test_case["k"],
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()