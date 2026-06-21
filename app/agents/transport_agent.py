import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config.settings import settings
from app.rag.vector_store import get_vector_store
from app.utils.destination_utils import (
    get_available_destinations,
    is_destination_in_kb,
    normalize_destination_to_key,
)
from app.utils.json_utils import parse_json_response


TRANSPORT_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are the Transport Agent of VoyageAI.

Your task is to suggest transport options for the selected destination based on:
1. User travel request
2. Destination Agent output
3. Retrieved destination knowledge

User Travel Request:
{user_query}

Destination Agent Output:
{destination_result}

Retrieved Travel Knowledge:
{context}

Return your answer strictly in valid JSON format.

JSON structure:
{{
  "agent_name": "Transport Agent",
  "status": "success",
  "source_city": "source city if mentioned, otherwise unknown",
  "destination": "destination name",
  "travel_style_detected": "budget/comfort/luxury/fastest/unknown",
  "recommended_intercity_mode": "flight/train/bus/car/ferry/mixed/unknown",
  "intercity_options": [
    {{
      "mode": "flight/train/bus/car/ferry/mixed",
      "suitability": "high/medium/low",
      "best_for": "who this mode is best for",
      "cost_level": "low/medium/high/unknown",
      "time_level": "slow/medium/fast/unknown",
      "note": "short practical note"
    }}
  ],
  "local_transport_options": ["option 1", "option 2", "option 3"],
  "transport_tips": ["tip 1", "tip 2", "tip 3"],
  "limitations": "mention that this version does not use live schedules or prices",
  "confidence": "high/medium/low"
}}

Rules:
- Use the selected destination from the Destination Agent.
- Use retrieved travel knowledge when destination-specific transport tips are available.
- You may give general transport guidance, but do not invent exact train names, flight numbers, bus operators, departure times, or live prices.
- If source city is not clearly mentioned, set source_city to "unknown".
- If exact transport data is missing, be transparent in the limitations field.
- Do not add markdown.
- Do not wrap JSON in ```json.
""")


class TransportAgent:
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

        self.chain = TRANSPORT_AGENT_PROMPT | self.llm | StrOutputParser()

    def get_transport_context(
        self,
        user_query: str,
        destination_result: dict[str, Any],
        k: int = 4,
    ) -> str:
        destination = destination_result.get("recommended_destination")
        city_key = normalize_destination_to_key(destination)

        retrieval_query = f"""
Destination: {destination}
User request: {user_query}
Need transport suggestions, local travel tips, travel convenience, trip duration, destination access, and movement advice.
"""

        try:
            docs = self.vector_store.similarity_search(
                retrieval_query,
                k=k,
                filter={"city": city_key},
            )

            if not docs:
                docs = self.vector_store.similarity_search(
                    retrieval_query,
                    k=k,
                )

        except Exception as error:
            print("Filter search failed. Running normal similarity search.")
            print("Error:", error)

            docs = self.vector_store.similarity_search(
                retrieval_query,
                k=k,
            )

        context_parts = []

        for index, doc in enumerate(docs, start=1):
            city = doc.metadata.get("city")
            source = doc.metadata.get("source")

            context = f"""
Context {index}
City: {city}
Source: {source}
Content:
{doc.page_content}
"""
            context_parts.append(context.strip())

        return "\n\n".join(context_parts)

    def run(
        self,
        user_query: str,
        destination_result: dict[str, Any],
        k: int = 4,
    ) -> dict[str, Any]:
        destination_status = destination_result.get("status")
        destination = destination_result.get("recommended_destination")

        if destination_status != "success":
            return {
                "agent_name": "Transport Agent",
                "status": "skipped",
                "destination": None,
                "message": (
                    "Transport Agent skipped because Destination Agent did not "
                    "return a valid destination."
                ),
                "reason": destination_result.get("reason"),
                "confidence": "low",
            }

        if not destination:
            return {
                "agent_name": "Transport Agent",
                "status": "no_destination_found",
                "destination": None,
                "message": (
                    "Transport Agent cannot run because no destination was provided."
                ),
                "confidence": "low",
            }

        if not is_destination_in_kb(
            destination,
            self.available_destinations,
        ):
            return {
                "agent_name": "Transport Agent",
                "status": "out_of_knowledge_base",
                "destination": destination,
                "message": (
                    f"{destination} is not available in the current "
                    "VoyageAI knowledge base."
                ),
                "available_destinations": self.available_destinations,
                "confidence": "low",
            }

        context = self.get_transport_context(
            user_query=user_query,
            destination_result=destination_result,
            k=k,
        )

        response = self.chain.invoke(
            {
                "user_query": user_query,
                "destination_result": json.dumps(
                    destination_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "context": context,
            }
        )

        try:
            parsed_response = parse_json_response(response)
        except Exception:
            return {
                "agent_name": "Transport Agent",
                "status": "error",
                "destination": destination,
                "message": "Failed to parse Transport Agent response.",
                "confidence": "low",
                "raw_response": response,
            }

        parsed_response["status"] = parsed_response.get("status", "success")

        return parsed_response


_transport_agent_instance = None


def get_transport_agent() -> TransportAgent:
    global _transport_agent_instance

    if _transport_agent_instance is None:
        _transport_agent_instance = TransportAgent()

    return _transport_agent_instance


def run_transport_agent(
    user_query: str,
    destination_result: dict[str, Any],
    k: int = 4,
) -> dict[str, Any]:
    agent = get_transport_agent()

    return agent.run(
        user_query=user_query,
        destination_result=destination_result,
        k=k,
    )


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Low budget Bhubaneswar to Goa trip",
            "user_query": "I want to travel from Bhubaneswar to Goa on a low budget",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Goa",
                "reason": "Goa matches beaches, seafood, nightlife and relaxed coastal travel.",
                "suitable_for": [
                    "beach lovers",
                    "food lovers",
                    "nightlife travelers",
                ],
                "suggested_duration": "3 to 5 days",
                "best_time_to_visit": "November to February",
                "confidence": "high",
            },
        },
        {
            "name": "Andaman honeymoon water activities trip",
            "user_query": "I want a honeymoon trip to Andaman with clean beaches, island hopping and water activities",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Andaman and Nicobar Islands",
                "reason": "Andaman is suitable for clean beaches, scuba diving, snorkeling and island experiences.",
                "suitable_for": [
                    "honeymoon couples",
                    "beach lovers",
                    "water sports lovers",
                ],
                "suggested_duration": "5 to 7 days",
                "best_time_to_visit": "October to May",
                "confidence": "high",
            },
        },
        {
            "name": "Destination Agent failed",
            "user_query": "I want to go to Venice",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "out_of_knowledge_base",
                "recommended_destination": None,
                "reason": "Venice is not available in the current VoyageAI knowledge base.",
                "confidence": "low",
            },
        },
    ]

    transport_agent = TransportAgent()

    for test_case in test_cases:
        print("=" * 100)
        print("Test Case:", test_case["name"])
        print("User Query:", test_case["user_query"])
        print("-" * 100)

        result = transport_agent.run(
            user_query=test_case["user_query"],
            destination_result=test_case["destination_result"],
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()