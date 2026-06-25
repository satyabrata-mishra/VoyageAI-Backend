import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools.hotel_tool import search_hotels

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

def safe_str(value):
    """
    Convert None to empty string and strip whitespace.
    """
    return str(value or "").strip()


def safe_lower(value):
    """
    Safe lowercase conversion.
    """
    return safe_str(value).lower()

# =====================================================
# Budget Detection
# =====================================================

def detect_budget_level(user_query):

    query = user_query.lower()

    if any(word in query for word in [
        "budget",
        "cheap",
        "low budget",
        "affordable",
        "hostel"
    ]):
        return "budget"

    if any(word in query for word in [
        "luxury",
        "premium",
        "5 star",
        "resort"
    ]):
        return "luxury"

    return "comfort"


# =====================================================
# Traveler Type Detection
# =====================================================

def detect_traveler_type(user_query):

    query = user_query.lower()

    if any(word in query for word in [
        "family",
        "kids",
        "children"
    ]):
        return "family"

    if any(word in query for word in [
        "honeymoon",
        "couple",
        "romantic"
    ]):
        return "couple"

    if any(word in query for word in [
        "business",
        "office",
        "work"
    ]):
        return "business"

    return "solo"


# =====================================================
# Travel Preferences
# =====================================================

def detect_preferences(user_query):

    query = user_query.lower()

    preferences = []

    mapping = {
        "beach": ["beach", "sea", "coast"],
        "nightlife": ["nightlife", "party", "club"],
        "food": ["food", "restaurant", "seafood"],
        "history": ["history", "heritage", "culture"],
        "nature": ["nature", "mountain", "waterfall"],
        "shopping": ["shopping", "market"]
    }

    for category, keywords in mapping.items():

        if any(
            keyword in query
            for keyword in keywords
        ):
            preferences.append(category)

    return preferences


# =====================================================
# Hotel Ranking
# =====================================================

def rank_hotels(
    user_query,
    destination,
    hotels,
    budget_level,
    traveler_type,
    preferences
):

    prompt = f"""
You are an expert travel hotel advisor.

User Query:
{user_query}

Destination:
{destination}

Budget Level:
{budget_level}

Traveler Type:
{traveler_type}

Traveler Preferences:
{preferences}

Available Hotels:
{json.dumps(hotels[:25], indent=2)}

Select:

1. Top 3 recommended hotels.
2. Reason for each hotel.
3. Recommended area.
4. 3 alternative hotels.

Return ONLY valid JSON.

{{
  "recommended_area": "...",

  "top_recommended_hotels": [
    {{
      "hotel_name": "...",
      "reason": "..."
    }},
    {{
      "hotel_name": "...",
      "reason": "..."
    }},
    {{
      "hotel_name": "...",
      "reason": "..."
    }}
  ],

  "alternative_hotels": [
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

    try:
        return json.loads(content)

    except Exception:

        print("Failed JSON:")
        print(content)

        return {
            "recommended_area": destination,
            "top_recommended_hotels": [],
            "alternative_hotels": []
        }

def find_hotel_by_name(
    hotel_name,
    hotels
):

    hotel_name = safe_lower(hotel_name)

    if not hotel_name:
        return None

    for hotel in hotels:

        if (safe_lower(hotel.get("name"))==hotel_name):
            return hotel

    return None

# =====================================================
# Main Agent
# =====================================================

def run_hotel_agent(
    user_query,
    destination_result
):

    try:

        destination = destination_result.get(
            "recommended_destination"
        )

        if not destination:

            return {
                "agent_name": "Hotel Agent",
                "status": "failed",
                "error": "Destination missing."
            }

        coordinates = destination_result.get(
            "coordinates",
            {}
        )

        latitude = coordinates.get("latitude")
        longitude = coordinates.get("longitude")

        if latitude is None or longitude is None:

            return {
                "agent_name": "Hotel Agent",
                "status": "failed",
                "error": "Destination coordinates missing."
            }

        # -----------------------------------------
        # Fetch Hotels
        # -----------------------------------------

        hotels = search_hotels(
            latitude=latitude,
            longitude=longitude,
            radius=50000,
            limit=50
        )

        if not hotels:

            return {
                "agent_name": "Hotel Agent",
                "status": "failed",
                "error": "No hotels found."
            }

        # -----------------------------------------
        # User Profiling
        # -----------------------------------------

        budget_level = detect_budget_level(
            user_query
        )

        traveler_type = detect_traveler_type(
            user_query
        )

        preferences = detect_preferences(
            user_query
        )

        # -----------------------------------------
        # LLM Ranking
        # -----------------------------------------

        ranking = rank_hotels(
            user_query=user_query,
            destination=destination,
            hotels=hotels,
            budget_level=budget_level,
            traveler_type=traveler_type,
            preferences=preferences
        )

        print("\n===== HOTEL RANKING OUTPUT =====")
        print(
            json.dumps(
                ranking,
                indent=2,
                ensure_ascii=False
            )
        )

        # -----------------------------------------
        # Build Top Recommendations
        # -----------------------------------------

        top_recommended_hotels = []

        for recommendation in ranking.get(
            "top_recommended_hotels",
            []
        ):

            hotel_name = safe_str(
                recommendation.get(
                    "hotel_name"
                )
            )

            if not hotel_name:
                continue

            matched_hotel = None

            for hotel in hotels:

                hotel_api_name = safe_lower(
                    hotel.get("name")
                )

                llm_hotel_name = safe_lower(
                    hotel_name
                )

                if hotel_api_name == llm_hotel_name:

                    matched_hotel = hotel.copy()

                    matched_hotel[
                        "selection_reason"
                    ] = safe_str(
                        recommendation.get(
                            "reason"
                        )
                    )

                    break

            if matched_hotel:

                top_recommended_hotels.append(
                    matched_hotel
                )

        # -----------------------------------------
        # Fallback
        # -----------------------------------------

        if len(
            top_recommended_hotels
        ) == 0:

            for hotel in hotels[:3]:

                hotel_copy = hotel.copy()

                hotel_copy[
                    "selection_reason"
                ] = (
                    "Recommended based on "
                    "overall relevance."
                )

                top_recommended_hotels.append(
                    hotel_copy
                )

        # -----------------------------------------
        # Alternative Hotels
        # -----------------------------------------

        alternative_hotels = []

        for hotel_name in ranking.get(
            "alternative_hotels",
            []
        ):

            hotel_name = safe_str(
                hotel_name
            )

            if not hotel_name:
                continue

            matched_hotel = None

            for hotel in hotels:

                if (
                    safe_lower(
                        hotel.get("name")
                    )
                    ==
                    safe_lower(
                        hotel_name
                    )
                ):

                    matched_hotel = hotel.copy()
                    break

            if matched_hotel:

                alternative_hotels.append(
                    matched_hotel
                )

        # -----------------------------------------
        # Alternative Fallback
        # -----------------------------------------

        if len(
            alternative_hotels
        ) == 0:

            alternative_hotels = hotels[3:8]

        # -----------------------------------------
        # Final Response
        # -----------------------------------------

        return {

            "agent_name":
                "Hotel Agent",

            "status":
                "success",

            "destination":
                destination,

            "budget_preference":
                budget_level,

            "traveler_type":
                traveler_type,

            "preferences":
                preferences,

            "recommended_area":
                ranking.get(
                    "recommended_area",
                    destination
                ),

            "top_recommended_hotels":
                top_recommended_hotels,

            "alternative_hotels":
                alternative_hotels,

            "hotels_considered":
                len(hotels),

            "confidence":
                "high"
        }

    except Exception as e:

        import traceback

        print(
            "\n===== HOTEL AGENT ERROR ====="
        )

        traceback.print_exc()

        return {

            "agent_name":
                "Hotel Agent",

            "status":
                "failed",

            "error":
                str(e)
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

    result = run_hotel_agent(
        user_query="""
        I want a low budget Goa trip
        with beaches, seafood
        and nightlife.
        """,
        destination_result=destination_result
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )