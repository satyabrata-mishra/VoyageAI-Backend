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


WEATHER_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are the Weather Agent of VoyageAI.

Your task is to give seasonal weather guidance for the selected destination based on:
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
  "agent_name": "Weather Agent",
  "status": "success",
  "destination": "destination name",
  "weather_advice_type": "seasonal_guidance_not_live_forecast",
  "requested_time_period": "month/season if mentioned by user, otherwise unknown",
  "best_time_to_visit": "best time from retrieved knowledge",
  "season_suitability": "high/medium/low/unknown",
  "weather_summary": "short seasonal weather summary",
  "activity_suitability": {{
    "sightseeing": "high/medium/low/unknown",
    "outdoor_activities": "high/medium/low/unknown",
    "water_activities": "high/medium/low/unknown",
    "snow_activities": "high/medium/low/unknown"
  }},
  "packing_tips": ["tip 1", "tip 2", "tip 3"],
  "weather_risks": ["risk 1", "risk 2"],
  "travel_advice": "practical advice based on the season",
  "limitations": "This version uses seasonal knowledge from the RAG database and does not use live weather forecast APIs.",
  "confidence": "high/medium/low"
}}

Rules:
- Use only the retrieved travel knowledge.
- Do not invent live temperature, rainfall, wind speed, AQI, or exact weather forecast.
- Do not claim to know current weather.
- If the user asks for live/current weather, clearly mention the limitation.
- If the user mentions a month or season, compare it with the best time to visit from the retrieved knowledge.
- Do not recommend weather details outside the selected destination context.
- Do not add markdown.
- Do not wrap JSON in ```json.
""")


class WeatherAgent:
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

        self.chain = WEATHER_AGENT_PROMPT | self.llm | StrOutputParser()

    def get_weather_context(
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
Need best time to visit, weather season, monsoon, summer, winter, snow, packing tips, travel risks, activity suitability.
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
                "agent_name": "Weather Agent",
                "status": "skipped",
                "destination": None,
                "message": (
                    "Weather Agent skipped because Destination Agent did not "
                    "return a valid destination."
                ),
                "reason": destination_result.get("reason"),
                "confidence": "low",
            }

        if not destination:
            return {
                "agent_name": "Weather Agent",
                "status": "no_destination_found",
                "destination": None,
                "message": (
                    "Weather Agent cannot run because no destination was provided."
                ),
                "confidence": "low",
            }

        if not is_destination_in_kb(
            destination,
            self.available_destinations,
        ):
            return {
                "agent_name": "Weather Agent",
                "status": "out_of_knowledge_base",
                "destination": destination,
                "message": (
                    f"{destination} is not available in the current "
                    "VoyageAI knowledge base."
                ),
                "available_destinations": self.available_destinations,
                "confidence": "low",
            }

        context = self.get_weather_context(
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
                "agent_name": "Weather Agent",
                "status": "error",
                "destination": destination,
                "message": "Failed to parse Weather Agent response.",
                "confidence": "low",
                "raw_response": response,
            }

        parsed_response["status"] = parsed_response.get("status", "success")

        return parsed_response


_weather_agent_instance = None


def get_weather_agent() -> WeatherAgent:
    global _weather_agent_instance

    if _weather_agent_instance is None:
        _weather_agent_instance = WeatherAgent()

    return _weather_agent_instance


def run_weather_agent(
    user_query: str,
    destination_result: dict[str, Any],
    k: int = 4,
) -> dict[str, Any]:
    agent = get_weather_agent()

    return agent.run(
        user_query=user_query,
        destination_result=destination_result,
        k=k,
    )


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Manali December snow trip",
            "user_query": "I want to visit Manali in December for snow and adventure",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Manali",
                "reason": "Manali matches snow, mountains, adventure sports, cafes and scenic valleys.",
                "suitable_for": [
                    "adventure seekers",
                    "snow lovers",
                    "nature lovers",
                ],
                "suggested_duration": "4 to 6 days",
                "best_time_to_visit": (
                    "March to June for pleasant weather and December to February for snow"
                ),
                "confidence": "high",
            },
        },
        {
            "name": "Current weather limitation test",
            "user_query": "What is the current weather in Goa right now?",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Goa",
                "reason": "Goa matches beach and coastal travel.",
                "suitable_for": [
                    "beach lovers",
                    "relaxed vacation seekers",
                ],
                "suggested_duration": "3 to 5 days",
                "best_time_to_visit": "November to February",
                "confidence": "high",
            },
        },
        {
            "name": "Destination Agent failed",
            "user_query": "What is the weather in Venice right now?",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "out_of_knowledge_base",
                "recommended_destination": None,
                "reason": "Venice is not available in the current VoyageAI knowledge base.",
                "confidence": "low",
            },
        },
    ]

    weather_agent = WeatherAgent()

    for test_case in test_cases:
        print("=" * 100)
        print("Test Case:", test_case["name"])
        print("User Query:", test_case["user_query"])
        print("-" * 100)

        result = weather_agent.run(
            user_query=test_case["user_query"],
            destination_result=test_case["destination_result"],
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()