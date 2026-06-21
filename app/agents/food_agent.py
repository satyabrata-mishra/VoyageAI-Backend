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


FOOD_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are the Food Agent of VoyageAI.

Your task is to suggest local food and cuisine experiences for the selected destination based on:
1. User travel request
2. Destination Agent output
3. Retrieved travel knowledge

User Travel Request:
{user_query}

Destination Agent Output:
{destination_result}

Retrieved Travel Knowledge:
{context}

Return your answer strictly in valid JSON format.

JSON structure:
{{
  "agent_name": "Food Agent",
  "status": "success",
  "destination": "destination name",
  "food_preference_detected": "seafood/vegetarian/street_food/local_cuisine/cafe_food/unknown",
  "recommended_foods": ["food 1", "food 2", "food 3", "food 4", "food 5"],
  "food_experience": "short description of the overall food experience",
  "best_for": ["food lover type 1", "food lover type 2"],
  "food_tips": ["tip 1", "tip 2"],
  "reason": "why these food recommendations match the user",
  "confidence": "high/medium/low"
}}

Rules:
- Use only the retrieved travel knowledge.
- Do not invent restaurant names.
- Do not invent food prices.
- Do not recommend dishes outside the selected destination context.
- If the user has no specific food preference, recommend the most important local foods from the context.
- Do not add markdown.
- Do not wrap JSON in ```json.
""")


class FoodAgent:
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

        self.chain = FOOD_AGENT_PROMPT | self.llm | StrOutputParser()

    def get_food_context(
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
Need food recommendations, local dishes, cuisine, must-try food, food experience, eating tips.
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
                "agent_name": "Food Agent",
                "status": "skipped",
                "destination": None,
                "message": (
                    "Food Agent skipped because Destination Agent did not "
                    "return a valid destination."
                ),
                "reason": destination_result.get("reason"),
                "confidence": "low",
            }

        if not destination:
            return {
                "agent_name": "Food Agent",
                "status": "no_destination_found",
                "destination": None,
                "message": (
                    "Food Agent cannot run because no destination was provided."
                ),
                "confidence": "low",
            }

        if not is_destination_in_kb(
            destination,
            self.available_destinations,
        ):
            return {
                "agent_name": "Food Agent",
                "status": "out_of_knowledge_base",
                "destination": destination,
                "message": (
                    f"{destination} is not available in the current "
                    "VoyageAI knowledge base."
                ),
                "available_destinations": self.available_destinations,
                "confidence": "low",
            }

        context = self.get_food_context(
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
                "agent_name": "Food Agent",
                "status": "error",
                "destination": destination,
                "message": "Failed to parse Food Agent response.",
                "confidence": "low",
                "raw_response": response,
            }

        parsed_response["status"] = parsed_response.get("status", "success")

        return parsed_response


def run_food_agent(
    user_query: str,
    destination_result: dict[str, Any],
    k: int = 4,
) -> dict[str, Any]:
    agent = FoodAgent()

    return agent.run(
        user_query=user_query,
        destination_result=destination_result,
        k=k,
    )

if __name__ == "__main__":
    test_cases = [
        {
            "user_query": "I want beaches, seafood, nightlife and a relaxed vacation",
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
            "user_query": "I want forts, palaces, royal culture and traditional food",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Jaipur",
                "reason": "Jaipur matches forts, palaces, culture and traditional food.",
                "suitable_for": [
                    "history lovers",
                    "culture lovers",
                    "families",
                ],
                "suggested_duration": "2 to 4 days",
                "best_time_to_visit": "October to March",
                "confidence": "high",
            },
        },
        {
            "user_query": "I want to go to Venice and try Italian food",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "out_of_knowledge_base",
                "recommended_destination": None,
                "reason": "Venice is not available in the current VoyageAI knowledge base.",
                "confidence": "low",
            },
        },
    ]

    food_agent = FoodAgent()

    for test_case in test_cases:
        print("=" * 100)
        print("User Query:", test_case["user_query"])
        print("-" * 100)

        result = food_agent.run(
            user_query=test_case["user_query"],
            destination_result=test_case["destination_result"],
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()