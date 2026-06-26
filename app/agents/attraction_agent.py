import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools.attraction_tool import (
    search_attractions,
    enrich_attractions
)

from app.tools.image_tool import enrich_attraction_images

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def rank_attractions(
    user_query,
    destination,
    attractions
):
    """
    Use LLM to rank attractions based on user interests.
    """

    prompt = f"""
        You are an expert travel planner.

        User Query:
        {user_query}

        Destination:
        {destination}

        Available Attractions:
        {json.dumps(attractions[:25], indent=2)}

        Select:

        1. Top 5 attractions
        2. Reason for each attraction
        3. 5 alternative attractions

        Return ONLY valid JSON.

        {{
        "top_attractions": [
            {{
            "name": "...",
            "reason": "..."
            }}
        ],

        "alternative_attractions": [
            "...",
            "...",
            "..."
        ]
        }}
    """

    response = llm.invoke(prompt)

    content = response.content.strip()

    if "```json" in content:
        content = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    return json.loads(content)

def find_attraction_by_name(
    attraction_name,
    attractions
):

    for attraction in attractions:

        if (attraction.get("name","").lower()==attraction_name.lower()):
            return attraction

    return None

def run_attraction_agent(
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

        latitude = coordinates.get(
            "latitude"
        )

        longitude = coordinates.get(
            "longitude"
        )

        if latitude is None or longitude is None:

            return {
                "agent_name": "Attraction Agent",
                "status": "failed",
                "error": "Destination coordinates missing."
            }

        # ------------------------------------
        # Fetch Attractions
        # ------------------------------------

        attractions = search_attractions(
            latitude=latitude,
            longitude=longitude,
            radius=50000,
            limit=50
        )

        if len(attractions) == 0:

            return {
                "agent_name": "Attraction Agent",
                "status": "failed",
                "error": "No attractions found."
            }

        attractions = enrich_attractions(
            attractions
        )
        for attraction in attractions:
            if(attraction.get("name")):
                attraction["image"] = enrich_attraction_images(attraction["name"])

        # ------------------------------------
        # Rank Attractions
        # ------------------------------------

        ranking = rank_attractions(
            user_query=user_query,
            destination=destination,
            attractions=attractions
        )

        # ------------------------------------
        # Top Attractions
        # ------------------------------------

        top_attractions = []

        for recommendation in ranking.get(
            "top_attractions",
            []
        ):

            attraction_name = recommendation.get(
                "name",
                ""
            )

            attraction = find_attraction_by_name(
                attraction_name,
                attractions
            )

            if attraction:

                attraction_copy = attraction.copy()

                attraction_copy[
                    "selection_reason"
                ] = recommendation.get(
                    "reason",
                    ""
                )

                top_attractions.append(
                    attraction_copy
                )

        # Fallback

        if len(top_attractions) == 0:

            for attraction in attractions[:5]:

                attraction_copy = attraction.copy()

                attraction_copy[
                    "selection_reason"
                ] = (
                    "Popular attraction in destination."
                )

                top_attractions.append(
                    attraction_copy
                )

        # ------------------------------------
        # Alternative Attractions
        # ------------------------------------

        alternative_attractions = []

        for attraction_name in ranking.get(
            "alternative_attractions",
            []
        ):

            attraction = find_attraction_by_name(
                attraction_name,
                attractions
            )

            if attraction:

                alternative_attractions.append(
                    attraction
                )

        if len(alternative_attractions) == 0:

            alternative_attractions = (
                attractions[5:10]
            )

        # ------------------------------------
        # Final Response
        # ------------------------------------

        return {

            "agent_name":
                "Attraction Agent",

            "status":
                "success",

            "destination":
                destination,

            "top_attractions":
                top_attractions,

            "alternative_attractions":
                alternative_attractions,

            "attractions_considered":
                len(attractions),

            "confidence":
                "high"
        }

    except Exception as e:

        return {

            "agent_name":
                "Attraction Agent",

            "status":
                "failed",

            "error":
                str(e)
        }

if __name__ == "__main__":
    result = run_attraction_agent(
        user_query="""
            I want a Goa trip with beaches,
            nightlife and scenic places.
        """,
        destination_result={
            "agent_name": "Destination Agent",
            "status": "success",
            "recommended_destination": "Goa",
            "coordinates": {
                "latitude": 15.2993,
                "longitude": 74.1240
            }
        }
    )
    print(json.dumps(result,indent=2)
)