import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools.weather_tool import get_current_weather

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)


def calculate_weather_suitability(weather_data):
    """
    Rule-based weather scoring.
    """

    condition = weather_data.get("condition", "").lower()
    temperature = weather_data.get("temperature", 0)

    suitability_score = 0

    # Condition scoring

    if condition in ["clear"]:
        suitability_score += 4

    elif condition in ["clouds"]:
        suitability_score += 3

    elif condition in ["mist", "haze"]:
        suitability_score += 2

    elif condition in ["rain", "drizzle"]:
        suitability_score += 1

    elif condition in ["thunderstorm"]:
        suitability_score += 0

    # Temperature scoring

    if 22 <= temperature <= 32:
        suitability_score += 4

    elif 18 <= temperature < 22:
        suitability_score += 3

    elif 32 < temperature <= 36:
        suitability_score += 2

    else:
        suitability_score += 1

    # Final classification

    if suitability_score >= 7:
        return "high"

    elif suitability_score >= 5:
        return "medium"

    else:
        return "low"


def generate_weather_recommendation(
    user_query,
    destination,
    weather_data,
    suitability
):
    """
    Uses Groq only for recommendation generation.
    """

    prompt = f"""
You are an expert travel weather advisor.

User Query:
{user_query}

Destination:
{destination}

Current Weather:

Temperature: {weather_data['temperature']}°C
Feels Like: {weather_data['feels_like']}°C
Humidity: {weather_data['humidity']}%
Condition: {weather_data['condition']}
Description: {weather_data['description']}

Weather Suitability:
{suitability}

Write a short travel recommendation in 2-3 sentences.

Mention only about the below aspects of the weather and its impact on travel:
- whether outdoor activities are suitable
- whether sightseeing is suitable
- any precautions travelers should take

Don't mention anything else from the user query. Keep the response concise and to the point.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def run_weather_agent(
    user_query,
    destination_result
):
    """
    Weather Agent V2
    Uses live weather data from OpenWeather API.
    """

    try:

        destination = destination_result.get(
            "recommended_destination"
        )

        if not destination:
            return {
                "agent_name": "Weather Agent",
                "status": "failed",
                "error": "No destination provided."
            }

        # --------------------------------------------------
        # Get live weather
        # --------------------------------------------------

        weather_data = get_current_weather(destination)

        if not weather_data.get("success"):

            return {
                "agent_name": "Weather Agent",
                "status": "failed",
                "destination": destination,
                "error": weather_data.get("error")
            }

        # --------------------------------------------------
        # Rule based suitability
        # --------------------------------------------------

        suitability = calculate_weather_suitability(
            weather_data
        )

        # --------------------------------------------------
        # LLM recommendation
        # --------------------------------------------------

        recommendation = generate_weather_recommendation(
            user_query=user_query,
            destination=destination,
            weather_data=weather_data,
            suitability=suitability
        )

        # --------------------------------------------------
        # Final output
        # --------------------------------------------------

        return {
            "agent_name": "Weather Agent",
            "status": "success",

            "destination": destination,

            "live_weather": {
                "temperature": weather_data["temperature"],
                "feels_like": weather_data["feels_like"],
                "humidity": weather_data["humidity"],
                "condition": weather_data["condition"],
                "description": weather_data["description"]
            },

            "weather_suitability": suitability,

            "recommendation": recommendation,

            "confidence": "high"
        }

    except Exception as e:

        return {
            "agent_name": "Weather Agent",
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":

    destination_result = {
        "recommended_destination": "Goa"
    }

    result = run_weather_agent(
        user_query="I want a beach vacation with nightlife.",
        destination_result=destination_result
    )

    print(json.dumps(result, indent=2))