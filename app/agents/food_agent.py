import os
import json
import re
import traceback

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools.food_tool import (
    search_restaurants,
    enrich_restaurants
)

from app.tools.image_tool import enrich_restaurant_images

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)


# =====================================================
# Safe JSON Parser
# =====================================================

def safe_json_parse(content: str):

    content = content.strip()

    if "```json" in content:
        content = (
            content.replace("```json", "")
            .replace("```", "")
            .strip()
        )

    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError("No JSON found.")

    return json.loads(match.group())


# =====================================================
# Extract Food Preference
# =====================================================

def extract_food_preference(user_query):

    prompt = f"""
You are a food preference extraction system.

User Query:
{user_query}

Identify the primary food preference.

Possible values:

- seafood
- vegetarian
- vegan
- street food
- local cuisine
- fine dining
- cafe
- desserts
- budget
- no preference

Return ONLY one value.
"""

    response = llm.invoke(prompt)

    preference = response.content.strip().lower()

    return preference


# =====================================================
# Rank Restaurants
# =====================================================

def rank_restaurants(
    user_query,
    destination,
    restaurants,
    preference
):

    prompt = f"""
You are an expert travel food planner.

Destination:
{destination}

Food Preference:
{preference}

Nearby Restaurants:

{json.dumps(restaurants[:15], indent=2)}

Recommend the best restaurants.

Respond ONLY in JSON.

Schema:

{{
    "recommended_restaurants":[
        {{
            "name":"",
            "reason":""
        }}
    ],
    "must_try_foods":[]
}}
"""

    response = llm.invoke(prompt)

    return safe_json_parse(response.content)


# =====================================================
# Main Food Agent
# =====================================================

def run_food_agent(
    user_query,
    destination_result
):

    try:

        destination = destination_result.get(
            "recommended_destination"
        )

        coordinates = destination_result.get(
            "coordinates",
            {}
        )

        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")

        if latitude is None or longitude is None:

            return {
                "agent_name": "Food Agent",
                "status": "failed",
                "error": "Destination coordinates not found."
            }

        food_preference = extract_food_preference(
            user_query
        )

        restaurants = search_restaurants(
            latitude=latitude,
            longitude=longitude,
            radius=5000,
            limit=30
        )

        restaurants = enrich_restaurants(
            restaurants,
            max_results=15
        )

        ranking = rank_restaurants(
            user_query=user_query,
            destination=destination,
            restaurants=restaurants,
            preference=food_preference
        )

        for restaurant in ranking.get("recommended_restaurants", []):
            restaurant["image"] = enrich_restaurant_images(restaurant["name"])
        
        return {

            "agent_name": "Food Agent",

            "status": "success",

            "destination": destination,

            "food_preference_detected":
                food_preference,

            "recommended_restaurants":
                ranking.get(
                    "recommended_restaurants",
                    []
                ),

            "must_try_foods":
                ranking.get(
                    "must_try_foods",
                    []
                ),

            "restaurants_considered":
                len(restaurants),

            "confidence": "high"
        }

    except Exception as e:

        traceback.print_exc()

        return {

            "agent_name": "Food Agent",

            "status": "failed",

            "error": str(e)
        }


# =====================================================
# Local Testing
# =====================================================

if __name__ == "__main__":

    destination_result = {

        "recommended_destination": "Goa",

        "coordinates": {
            "latitude": 15.2993,
            "longitude": 74.1240
        }
    }

    result = run_food_agent(

        user_query=(
            "I want a Goa trip with seafood, beaches "
            "and nightlife."
        ),

        destination_result=destination_result
    )

    print(json.dumps(result, indent=2))