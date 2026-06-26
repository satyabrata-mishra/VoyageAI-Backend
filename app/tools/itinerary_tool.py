import math
import re


# ==========================================================
# Extract Trip Duration
# ==========================================================

def extract_trip_duration(user_query: str) -> int:
    """
    Extract trip duration from user query.

    Examples:
        "4 day Goa trip"
        "7 days in Japan"
        "Weekend in Goa"

    Default = 3 days
    """

    query = user_query.lower()

    patterns = [
        r"(\d+)\s*day",
        r"(\d+)\s*days",
        r"(\d+)\s*night",
        r"(\d+)\s*nights"
    ]

    for pattern in patterns:

        match = re.search(pattern, query)

        if match:
            return max(1, int(match.group(1)))

    if "weekend" in query:
        return 2

    return 3


# ==========================================================
# Create Empty Daily Plan
# ==========================================================

def create_daily_plan(days: int):

    return [

        {
            "day": day,
            "activities": []
        }

        for day in range(1, days + 1)

    ]


# ==========================================================
# Distribute Attractions Evenly
# ==========================================================

def distribute_attractions(
        attractions,
        days
):
    """
    Evenly distribute attractions across all trip days.
    """

    if not attractions:

        return {
            day: []
            for day in range(1, days + 1)
        }

    grouped = {
        day: []
        for day in range(1, days + 1)
    }

    for index, attraction in enumerate(attractions):

        day = (index % days) + 1

        grouped[day].append(attraction)

    return grouped


# ==========================================================
# Build Weather Summary
# ==========================================================

def prepare_weather_summary(weather_result):

    if not weather_result:

        return {}

    summary = {}

    if "forecast" in weather_result:
        summary["forecast"] = weather_result["forecast"]

    if "travel_advice" in weather_result:
        summary["travel_advice"] = weather_result["travel_advice"]

    if "best_time_to_visit" in weather_result:
        summary["best_time_to_visit"] = weather_result["best_time_to_visit"]

    return summary


# ==========================================================
# Build Hotel Summary
# ==========================================================

def prepare_hotel_summary(hotel_result):

    if not hotel_result:
        return {}

    hotels = hotel_result.get(
        "recommended_hotels",
        []
    )

    if hotels:

        return hotels[0]

    return {}


# ==========================================================
# Build Restaurant Summary
# ==========================================================

def prepare_restaurant_summary(food_result):

    if not food_result:
        return []

    restaurants = food_result.get(
        "recommended_restaurants",
        []
    )

    return restaurants


# ==========================================================
# Build Transport Summary
# ==========================================================

def prepare_transport_summary(
        transport_result
):

    if not transport_result:
        return {}

    return transport_result


# ==========================================================
# Build Activity Plan
# ==========================================================

def build_daily_plan(
        grouped_attractions,
        days
):

    daily_plan = create_daily_plan(days)

    for day_plan in daily_plan:

        day = day_plan["day"]

        attractions = grouped_attractions.get(
            day,
            []
        )

        for attraction in attractions:

            day_plan["activities"].append(

                {
                    "type": "attraction",
                    "details": attraction
                }

            )

    return daily_plan


# ==========================================================
# Prepare Context For Itinerary Agent
# ==========================================================

def prepare_itinerary_context(

        user_query,

        destination_result,

        attraction_result,

        hotel_result,

        food_result,

        weather_result,

        transport_result

):

    days = extract_trip_duration(
        user_query
    )

    attractions = attraction_result.get(
        "top_attractions",
        []
    )

    grouped = distribute_attractions(

        attractions,

        days

    )

    daily_plan = build_daily_plan(

        grouped,

        days

    )

    context = {

        "trip_duration_days": days,

        "destination": destination_result.get(
            "recommended_destination"
        ),

        "country": destination_result.get(
            "country"
        ),

        "hotel": prepare_hotel_summary(
            hotel_result
        ),

        "restaurants": prepare_restaurant_summary(
            food_result
        ),

        "weather": prepare_weather_summary(
            weather_result
        ),

        "transport": prepare_transport_summary(
            transport_result
        ),

        "daily_plan": daily_plan

    }

    return context


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    destination_result = {

        "recommended_destination": "Goa",

        "country": "India"

    }

    attraction_result = {

        "top_attractions": [

            {"name": "Baga Beach"},

            {"name": "Fort Aguada"},

            {"name": "Calangute Beach"},

            {"name": "Chapora Fort"},

            {"name": "Dudhsagar Falls"},

            {"name": "Anjuna Beach"}

        ]

    }

    hotel_result = {

        "recommended_hotels": [

            {

                "name": "Sea View Resort",

                "rating": 4.5

            }

        ]

    }

    food_result = {

        "recommended_restaurants": [

            {

                "name": "Martin's Corner"

            },

            {

                "name": "Ritz Classic"

            }

        ]

    }

    weather_result = {

        "forecast": "Sunny",

        "travel_advice": "Carry sunscreen."

    }

    transport_result = {

        "recommended_transport": {

            "mode": "Scooter Rental"

        }

    }

    context = prepare_itinerary_context(

        user_query="I want a 4 day Goa trip.",

        destination_result=destination_result,

        attraction_result=attraction_result,

        hotel_result=hotel_result,

        food_result=food_result,

        weather_result=weather_result,

        transport_result=transport_result

    )

    from pprint import pprint

    pprint(context)