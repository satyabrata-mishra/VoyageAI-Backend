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


HOTEL_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are the Hotel Agent of VoyageAI.

Your task is to suggest the best stay type for the user based on:
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
  "agent_name": "Hotel Agent",
  "destination": "destination name",
  "budget_preference": "budget/comfort/luxury/unknown",
  "recommended_stay_type": "short recommendation",
  "best_areas": ["area 1", "area 2", "area 3"],
  "stay_options": {{
    "budget": "budget stay suggestion",
    "comfort": "comfort stay suggestion",
    "luxury": "luxury stay suggestion"
  }},
  "reason": "why this stay recommendation matches the user",
  "hotel_booking_tip": "practical booking tip",
  "confidence": "high/medium/low"
}}

Rules:
- Use only the retrieved travel knowledge.
- Do not invent exact hotel names.
- Do not invent exact prices.
- Suggest stay categories and areas only.
- If the user's budget is not mentioned, set budget_preference to "unknown".
- Do not add markdown.
- Do not wrap JSON in ```json.
""")


class HotelAgent:
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

        self.chain = HOTEL_AGENT_PROMPT | self.llm | StrOutputParser()

    def get_hotel_context(
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
Need hotel stay suggestions, budget stay, comfort stay, luxury stay, best areas, accommodation tips.
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

        if destination_status and destination_status != "success":
            return {
                "agent_name": "Hotel Agent",
                "status": "skipped",
                "destination": None,
                "message": (
                    "Hotel Agent skipped because Destination Agent did not "
                    "return a valid destination."
                ),
                "reason": destination_result.get("reason"),
                "confidence": "low",
            }

        if not destination:
            return {
                "agent_name": "Hotel Agent",
                "status": "no_destination_found",
                "destination": None,
                "message": (
                    "Hotel Agent cannot run because no destination was provided."
                ),
                "confidence": "low",
            }

        if not is_destination_in_kb(
            destination,
            self.available_destinations,
        ):
            return {
                "agent_name": "Hotel Agent",
                "status": "out_of_knowledge_base",
                "destination": destination,
                "message": (
                    f"{destination} is not available in the current "
                    "VoyageAI knowledge base."
                ),
                "available_destinations": self.available_destinations,
                "confidence": "low",
            }

        context = self.get_hotel_context(
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
                "agent_name": "Hotel Agent",
                "status": "error",
                "destination": destination,
                "message": "Failed to parse Hotel Agent response.",
                "confidence": "low",
                "raw_response": response,
            }

        parsed_response["status"] = parsed_response.get("status", "success")

        return parsed_response


def run_hotel_agent(
    user_query: str,
    destination_result: dict[str, Any],
    k: int = 4,
) -> dict[str, Any]:
    agent = HotelAgent()

    return agent.run(
        user_query=user_query,
        destination_result=destination_result,
        k=k,
    )


if __name__ == "__main__":
    import json

    test_cases = [
        {
            "name": "Luxury romantic Udaipur trip",
            "user_query": "I want a romantic luxury trip with lakes and palaces",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "udaipur",
                "reason": "Udaipur matches lakes, palaces and romantic luxury travel.",
                "suitable_for": ["romantic", "luxury", "couples"],
                "suggested_duration": "4-5 days",
                "best_time_to_visit": "October to March",
                "key_attractions": ["City Palace", "Lake Pichola", "Jag Mandir"],
                "confidence": "high",
            },
        },
        {
            "name": "Low budget Goa beach trip",
            "user_query": "I want beaches, seafood, nightlife and a relaxed vacation. My budget is low.",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Goa",
                "reason": "Goa matches beaches, seafood, nightlife and relaxed coastal travel.",
                "suitable_for": ["beach lovers", "food lovers", "nightlife travelers"],
                "suggested_duration": "3 to 5 days",
                "best_time_to_visit": "November to February",
                "key_attractions": ["Baga Beach", "Anjuna Beach", "Fort Aguada"],
                "confidence": "high",
            },
        },
        {
            "name": "Comfort Jaipur culture trip",
            "user_query": "I want forts, palaces, royal culture and traditional food. I prefer a comfortable stay.",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Jaipur",
                "reason": "Jaipur matches forts, palaces, royal culture and traditional food.",
                "suitable_for": ["history lovers", "culture lovers", "families"],
                "suggested_duration": "2 to 4 days",
                "best_time_to_visit": "October to March",
                "key_attractions": ["Amber Fort", "City Palace", "Hawa Mahal"],
                "confidence": "high",
            },
        },
        {
            "name": "Budget Rishikesh backpacking trip",
            "user_query": "I want yoga, rafting, cafes and a budget backpacking trip",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Rishikesh",
                "reason": "Rishikesh matches yoga, rafting, cafes and backpacking travel.",
                "suitable_for": ["backpackers", "adventure lovers", "spiritual travelers"],
                "suggested_duration": "3 to 4 days",
                "best_time_to_visit": "September to November and March to May",
                "key_attractions": ["Laxman Jhula", "Ganga Ghat", "River Rafting"],
                "confidence": "high",
            },
        },
        {
            "name": "Destination Agent failed",
            "user_query": "I want to visit Venice and stay near canals",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "out_of_knowledge_base",
                "recommended_destination": None,
                "reason": "Venice is not available in the current VoyageAI knowledge base.",
                "suitable_for": [],
                "suggested_duration": None,
                "best_time_to_visit": None,
                "key_attractions": [],
                "confidence": "low",
            },
        },
        {
            "name": "No destination returned",
            "user_query": "I want a peaceful trip but I am not sure where to go",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": None,
                "reason": "No suitable destination was found.",
                "suitable_for": [],
                "suggested_duration": None,
                "best_time_to_visit": None,
                "key_attractions": [],
                "confidence": "low",
            },
        },
        {
            "name": "Destination not in KB",
            "user_query": "I want luxury hotels in Switzerland",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Switzerland",
                "reason": "Switzerland matches luxury mountain travel.",
                "suitable_for": ["luxury travelers", "nature lovers"],
                "suggested_duration": "5 to 7 days",
                "best_time_to_visit": "April to October",
                "key_attractions": ["Swiss Alps", "Lucerne", "Interlaken"],
                "confidence": "medium",
            },
        },
    ]

    hotel_agent = HotelAgent()

    for test_case in test_cases:
        print("=" * 100)
        print("Test Case:", test_case["name"])
        print("User Query:", test_case["user_query"])
        print("-" * 100)

        result = hotel_agent.run(
            user_query=test_case["user_query"],
            destination_result=test_case["destination_result"],
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()