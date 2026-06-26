import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools.itinerary_tool import (
    prepare_itinerary_context
)

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)


# =====================================================
# Generate Final Itinerary
# =====================================================

def generate_itinerary(context):

    prompt = f"""
You are an expert AI travel planner.

Below is a structured travel planning context.

{json.dumps(context, indent=2)}

Generate a practical day-by-day itinerary.

Rules:

- Use ONLY the provided attractions.
- Do NOT invent new attractions.
- Recommend restaurants from the provided list whenever possible.
- Mention the recommended hotel naturally.
- Mention weather advice whenever relevant.
- Mention transport suggestions whenever useful.
- Spread attractions evenly across the trip.
- Keep each day realistic.
- Give each day a short theme.

Return ONLY valid JSON.

JSON Schema:

{{
    "trip_summary":"...",

    "hotel_recommendation":"...",

    "daily_itinerary":[

        {{
            "day":1,

            "theme":"...",

            "activities":[
                "...",
                "...",
                "..."
            ]
        }}

    ],

    "travel_tips":[
        "...",
        "..."
    ]
}}
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    if "```json" in content:
        content = content.replace("```json", "")
        content = content.replace("```", "").strip()

    return json.loads(content)


# =====================================================
# Main Agent
# =====================================================

def run_itinerary_agent(

    user_query,

    destination_result,

    attraction_result,

    hotel_result,

    food_result,

    weather_result,

    transport_result

):

    try:

        context = prepare_itinerary_context(

            user_query=user_query,

            destination_result=destination_result,

            attraction_result=attraction_result,

            hotel_result=hotel_result,

            food_result=food_result,

            weather_result=weather_result,

            transport_result=transport_result

        )

        itinerary = generate_itinerary(
            context
        )

        return {

            "agent_name": "Itinerary Agent",

            "status": "success",

            "destination": context["destination"],

            "trip_duration_days":
                context["trip_duration_days"],

            **itinerary,

            "confidence": "high"

        }

    except Exception as e:

        return {

            "agent_name": "Itinerary Agent",

            "status": "failed",

            "error": str(e)

        }


# =====================================================
# Local Testing
# =====================================================

if __name__ == "__main__":

    result = run_itinerary_agent(

        user_query="I want a 4 day Goa trip.",

        destination_result={
            "recommended_destination": "Goa",
            "country": "India"
        },

        attraction_result={
            "top_attractions":[
                {"name":"Baga Beach"},
                {"name":"Fort Aguada"},
                {"name":"Chapora Fort"},
                {"name":"Anjuna Beach"}
            ]
        },

        hotel_result={
            "recommended_hotels":[
                {
                    "name":"Sea View Resort",
                    "rating":4.5
                }
            ]
        },

        food_result={
            "recommended_restaurants":[
                {
                    "name":"Martin's Corner"
                },
                {
                    "name":"Ritz Classic"
                }
            ]
        },

        weather_result={
            "forecast":"Sunny",
            "travel_advice":"Carry sunscreen."
        },

        transport_result={
            "recommended_transport":{
                "mode":"Scooter Rental"
            }
        }

    )

    print(json.dumps(result, indent=4))