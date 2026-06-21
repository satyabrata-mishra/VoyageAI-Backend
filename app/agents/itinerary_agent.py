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


ITINERARY_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are the Itinerary Agent of VoyageAI.

Your task is to create the final travel itinerary by combining:
1. User travel request
2. Destination Agent output
3. Hotel Agent output
4. Food Agent output
5. Transport Agent output
6. Weather Agent output
7. Budget Agent output
8. Retrieved destination knowledge

User Travel Request:
{user_query}

Destination Agent Output:
{destination_result}

Hotel Agent Output:
{hotel_result}

Food Agent Output:
{food_result}

Transport Agent Output:
{transport_result}

Weather Agent Output:
{weather_result}

Budget Agent Output:
{budget_result}

Retrieved Destination Knowledge:
{context}

Return your answer strictly in valid JSON format.

JSON structure:
{{
  "agent_name": "Itinerary Agent",
  "status": "success/needs_budget_revision/skipped",
  "destination": "destination name",
  "trip_duration_days": number,
  "budget_status": "within_budget/over_budget/budget_not_provided/unknown",
  "summary": "short trip summary",
  "day_wise_itinerary": [
    {{
      "day": 1,
      "theme": "day theme",
      "morning": "morning plan",
      "afternoon": "afternoon plan",
      "evening": "evening plan",
      "food_suggestion": "food idea for the day",
      "transport_note": "local/intercity transport note"
    }}
  ],
  "stay_summary": "hotel or stay recommendation summary",
  "food_summary": "food recommendation summary",
  "transport_summary": "transport recommendation summary",
  "weather_summary": "seasonal weather advice summary",
  "budget_summary": {{
    "user_budget": number or null,
    "estimated_cost": number or null,
    "remaining_budget": number or null,
    "budget_status": "within_budget/over_budget/budget_not_provided/unknown"
  }},
  "budget_revision_notes": ["note 1", "note 2"],
  "travel_tips": ["tip 1", "tip 2", "tip 3"],
  "limitations": "short limitation statement",
  "confidence": "high/medium/low"
}}

Rules:
- Use only the retrieved destination knowledge and agent outputs.
- Do not invent exact hotel names.
- Do not invent live prices.
- Do not invent live weather.
- Do not invent train names, flight numbers, or bus operators.
- If Budget Agent says over_budget, set status to "needs_budget_revision".
- If Budget Agent says over_budget, include budget_revision_notes.
- If Budget Agent says within_budget, set status to "success".
- If budget is not provided, set budget_status to "budget_not_provided" and still generate itinerary.
- Keep the itinerary practical and beginner-friendly.
- Do not add markdown.
- Do not wrap JSON in ```json.
""")


def validate_required_agent_result(
    agent_result: dict[str, Any] | None,
    agent_name: str,
) -> tuple[bool, str | None]:
    if agent_result is None:
        return False, f"{agent_name} result is missing."

    status = agent_result.get("status")

    if status in [
        "skipped",
        "out_of_knowledge_base",
        "no_destination_found",
        "error",
    ]:
        return False, f"{agent_name} did not return a valid result."

    return True, None


class ItineraryAgent:
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
            temperature=0.3,
            groq_api_key=settings.GROQ_API_KEY,
        )

        self.chain = ITINERARY_AGENT_PROMPT | self.llm | StrOutputParser()

    def get_itinerary_context(
        self,
        user_query: str,
        destination_result: dict[str, Any],
        k: int = 5,
    ) -> str:
        destination = destination_result.get("recommended_destination")
        city_key = normalize_destination_to_key(destination)

        retrieval_query = f"""
Destination: {destination}
User request: {user_query}
Need attractions, food, stay suggestions, best time, travel tips, ideal duration, day-wise itinerary planning.
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
        hotel_result: dict[str, Any] | None = None,
        food_result: dict[str, Any] | None = None,
        transport_result: dict[str, Any] | None = None,
        weather_result: dict[str, Any] | None = None,
        budget_result: dict[str, Any] | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        destination_status = destination_result.get("status")
        destination = destination_result.get("recommended_destination")

        if destination_status != "success":
            return {
                "agent_name": "Itinerary Agent",
                "status": "skipped",
                "destination": None,
                "message": (
                    "Itinerary Agent skipped because Destination Agent did not "
                    "return a valid destination."
                ),
                "reason": destination_result.get("reason"),
                "confidence": "low",
            }

        if not destination:
            return {
                "agent_name": "Itinerary Agent",
                "status": "no_destination_found",
                "destination": None,
                "message": (
                    "Itinerary Agent cannot run because no destination was provided."
                ),
                "confidence": "low",
            }

        if not is_destination_in_kb(
            destination,
            self.available_destinations,
        ):
            return {
                "agent_name": "Itinerary Agent",
                "status": "out_of_knowledge_base",
                "destination": destination,
                "message": (
                    f"{destination} is not available in the current "
                    "VoyageAI knowledge base."
                ),
                "available_destinations": self.available_destinations,
                "confidence": "low",
            }

        context = self.get_itinerary_context(
            user_query=user_query,
            destination_result=destination_result,
            k=k,
        )

        safe_hotel_result = hotel_result or {}
        safe_food_result = food_result or {}
        safe_transport_result = transport_result or {}
        safe_weather_result = weather_result or {}
        safe_budget_result = budget_result or {}

        response = self.chain.invoke(
            {
                "user_query": user_query,
                "destination_result": json.dumps(
                    destination_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "hotel_result": json.dumps(
                    safe_hotel_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "food_result": json.dumps(
                    safe_food_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "transport_result": json.dumps(
                    safe_transport_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "weather_result": json.dumps(
                    safe_weather_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "budget_result": json.dumps(
                    safe_budget_result,
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
                "agent_name": "Itinerary Agent",
                "status": "error",
                "destination": destination,
                "message": "Failed to parse Itinerary Agent response.",
                "confidence": "low",
                "raw_response": response,
            }

        budget_status = safe_budget_result.get("budget_status", "unknown")

        if budget_status == "over_budget":
            parsed_response["status"] = "needs_budget_revision"
        elif budget_status in ["within_budget", "budget_not_provided"]:
            parsed_response["status"] = parsed_response.get("status", "success")
        else:
            parsed_response["status"] = parsed_response.get("status", "success")

        parsed_response["budget_status"] = budget_status

        return parsed_response


_itinerary_agent_instance = None


def get_itinerary_agent() -> ItineraryAgent:
    global _itinerary_agent_instance

    if _itinerary_agent_instance is None:
        _itinerary_agent_instance = ItineraryAgent()

    return _itinerary_agent_instance


def run_itinerary_agent(
    user_query: str,
    destination_result: dict[str, Any],
    hotel_result: dict[str, Any] | None = None,
    food_result: dict[str, Any] | None = None,
    transport_result: dict[str, Any] | None = None,
    weather_result: dict[str, Any] | None = None,
    budget_result: dict[str, Any] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    agent = get_itinerary_agent()

    return agent.run(
        user_query=user_query,
        destination_result=destination_result,
        hotel_result=hotel_result,
        food_result=food_result,
        transport_result=transport_result,
        weather_result=weather_result,
        budget_result=budget_result,
        k=k,
    )


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Goa within budget itinerary",
            "user_query": (
                "I want a 5 day Goa trip from Bhubaneswar under 20000 "
                "with beaches, seafood and nightlife. Keep it low budget."
            ),
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Goa",
                "reason": (
                    "Goa matches beaches, seafood, nightlife and relaxed "
                    "coastal travel."
                ),
                "suitable_for": [
                    "beach lovers",
                    "food lovers",
                    "nightlife travelers",
                ],
                "suggested_duration": "3 to 5 days",
                "best_time_to_visit": "November to February",
                "confidence": "high",
            },
            "hotel_result": {
                "agent_name": "Hotel Agent",
                "status": "success",
                "destination": "Goa",
                "budget_preference": "budget",
                "recommended_stay_type": (
                    "Hostel or budget stay near Anjuna, Vagator, or Baga"
                ),
                "best_areas": ["Anjuna", "Vagator", "Baga"],
                "stay_options": {
                    "budget": "Hostels near Anjuna, Vagator, or Baga",
                    "comfort": "Boutique hotels near Candolim or Panjim",
                    "luxury": "Beach resorts in South Goa",
                },
                "confidence": "high",
            },
            "food_result": {
                "agent_name": "Food Agent",
                "status": "success",
                "destination": "Goa",
                "food_preference_detected": "seafood",
                "recommended_foods": [
                    "Goan fish curry",
                    "Prawn balchao",
                    "Pork vindaloo",
                    "Xacuti",
                    "Bebinca",
                    "Seafood thali",
                ],
                "confidence": "high",
            },
            "transport_result": {
                "agent_name": "Transport Agent",
                "status": "success",
                "source_city": "Bhubaneswar",
                "destination": "Goa",
                "travel_style_detected": "budget",
                "recommended_intercity_mode": "train",
                "local_transport_options": [
                    "scooter rental",
                    "local taxi",
                    "walking near beach areas",
                ],
                "confidence": "medium",
            },
            "weather_result": {
                "agent_name": "Weather Agent",
                "status": "success",
                "destination": "Goa",
                "best_time_to_visit": "November to February",
                "season_suitability": "high",
                "weather_summary": (
                    "January is suitable for beaches, seafood and nightlife."
                ),
                "confidence": "high",
            },
            "budget_result": {
                "agent_name": "Budget Agent",
                "status": "success",
                "destination": "Goa",
                "trip_duration_days": 5,
                "estimated_nights": 4,
                "user_budget": 20000,
                "currency": "INR",
                "budget_level_detected": "budget",
                "transport_mode_used": "train",
                "estimated_cost_breakdown": {
                    "transport": 3000,
                    "hotel": 3960,
                    "food": 3080,
                    "activities": 2640,
                    "local_transport": 1760,
                    "buffer": 1444,
                },
                "total_estimated_cost": 15884,
                "remaining_budget": 4116,
                "budget_status": "within_budget",
                "confidence": "medium",
            },
        },
        {
            "name": "Goa over budget itinerary",
            "user_query": (
                "I want a 5 day Goa trip from Bhubaneswar under 10000 "
                "with beaches, seafood and nightlife."
            ),
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Goa",
                "reason": "Goa matches beaches and nightlife.",
                "suitable_for": ["beach lovers", "nightlife travelers"],
                "suggested_duration": "3 to 5 days",
                "best_time_to_visit": "November to February",
                "confidence": "high",
            },
            "hotel_result": {
                "agent_name": "Hotel Agent",
                "status": "success",
                "destination": "Goa",
                "budget_preference": "budget",
                "recommended_stay_type": "Hostel or budget guesthouse",
                "best_areas": ["Anjuna", "Vagator", "Baga"],
                "confidence": "high",
            },
            "food_result": {
                "agent_name": "Food Agent",
                "status": "success",
                "destination": "Goa",
                "food_preference_detected": "seafood",
                "recommended_foods": [
                    "Goan fish curry",
                    "Prawn balchao",
                    "Bebinca",
                ],
                "confidence": "high",
            },
            "transport_result": {
                "agent_name": "Transport Agent",
                "status": "success",
                "source_city": "Bhubaneswar",
                "destination": "Goa",
                "travel_style_detected": "budget",
                "recommended_intercity_mode": "train",
                "local_transport_options": [
                    "scooter rental",
                    "local bus",
                    "walking",
                ],
                "confidence": "medium",
            },
            "weather_result": {
                "agent_name": "Weather Agent",
                "status": "success",
                "destination": "Goa",
                "best_time_to_visit": "November to February",
                "weather_summary": "Winter season is suitable for beaches.",
                "confidence": "high",
            },
            "budget_result": {
                "agent_name": "Budget Agent",
                "status": "success",
                "destination": "Goa",
                "trip_duration_days": 5,
                "user_budget": 10000,
                "total_estimated_cost": 15884,
                "remaining_budget": -5884,
                "budget_status": "over_budget",
                "cost_saving_suggestions": [
                    "Reduce paid activities and include free sightseeing.",
                    "Use local buses and shared transport.",
                    "Choose hostels or dorm beds.",
                ],
                "confidence": "medium",
            },
        },
        {
            "name": "Destination Agent failed",
            "user_query": "I want a Venice itinerary with canals and Italian food",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "out_of_knowledge_base",
                "recommended_destination": None,
                "reason": (
                    "Venice is not available in the current VoyageAI knowledge base."
                ),
                "confidence": "low",
            },
            "hotel_result": None,
            "food_result": None,
            "transport_result": None,
            "weather_result": None,
            "budget_result": None,
        },
    ]

    itinerary_agent = ItineraryAgent()

    for test_case in test_cases:
        print("=" * 100)
        print("Test Case:", test_case["name"])
        print("User Query:", test_case["user_query"])
        print("-" * 100)

        result = itinerary_agent.run(
            user_query=test_case["user_query"],
            destination_result=test_case["destination_result"],
            hotel_result=test_case.get("hotel_result"),
            food_result=test_case.get("food_result"),
            transport_result=test_case.get("transport_result"),
            weather_result=test_case.get("weather_result"),
            budget_result=test_case.get("budget_result"),
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()