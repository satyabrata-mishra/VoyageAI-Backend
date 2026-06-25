import os
import requests
import json
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEOAPIFY_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_source_city(user_query):
    """
    Extract departure city from user query.
    """

    try:

        prompt = f"""
Extract the source/departure city from the travel query.

Query:
{user_query}

Return ONLY JSON.

Example:

{{
    "source_city": "Bhubaneswar"
}}

If no city is found:

{{
    "source_city": null
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

        result = json.loads(content)

        return result.get(
            "source_city"
        )

    except Exception:

        return None

def get_transport_context(
    user_query,
    destination
):
    """
    Complete transport context builder.
    """

    source_city = extract_source_city(
        user_query
    )

    if not source_city:

        return {
            "status": "failed",
            "error":
                "Could not identify source city."
        }

    route_info = get_route_information(
        source_city,
        destination
    )

    if not route_info:

        return {
            "status": "failed",
            "error":
                "Route information unavailable."
        }

    return {

        "status": "success",

        "source_city":
            source_city,

        **route_info
    }

# =====================================================
# Get Coordinates From City Name
# =====================================================

def get_city_coordinates(city_name):
    """
    Convert city name into coordinates.
    """

    try:

        url = (
            "https://api.geoapify.com/v1/geocode/search"
        )

        params = {
            "text": city_name,
            "limit": 1,
            "apiKey": API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        features = data.get(
            "features",
            []
        )

        if len(features) == 0:
            return None

        properties = features[0].get(
            "properties",
            {}
        )

        return {

            "city":
                city_name,

            "latitude":
                properties.get("lat"),

            "longitude":
                properties.get("lon"),

            "formatted":
                properties.get("formatted")
        }

    except Exception as e:

        print(
            f"Coordinate Lookup Error: {e}"
        )

        return None


# =====================================================
# Route Information
# =====================================================

def get_route_information(
    source_city,
    destination_city
):
    """
    Calculate route distance and travel time.
    """

    try:

        source = get_city_coordinates(
            source_city
        )

        destination = get_city_coordinates(
            destination_city
        )

        if not source or not destination:

            return None

        waypoints = (
            f"{source['latitude']},"
            f"{source['longitude']}|"
            f"{destination['latitude']},"
            f"{destination['longitude']}"
        )

        url = (
            "https://api.geoapify.com/v1/routing"
        )

        params = {

            "waypoints":
                waypoints,

            "mode":
                "drive",

            "apiKey":
                API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        features = data.get(
            "features",
            []
        )

        if len(features) == 0:
            return None

        properties = features[0].get(
            "properties",
            {}
        )

        distance_meters = properties.get(
            "distance",
            0
        )

        time_seconds = properties.get(
            "time",
            0
        )

        distance_km = round(
            distance_meters / 1000,
            2
        )

        estimated_hours = round(
            time_seconds / 3600,
            1
        )

        recommended_modes = []

        if distance_km < 300:

            recommended_modes = [
                "road",
                "train"
            ]

        elif distance_km < 800:

            recommended_modes = [
                "train",
                "road",
                "flight"
            ]

        else:

            recommended_modes = [
                "flight",
                "train",
                "road"
            ]

        return {

            "source_city":
                source_city,

            "destination_city":
                destination_city,

            "source_coordinates": {

                "latitude":
                    source["latitude"],

                "longitude":
                    source["longitude"]
            },

            "destination_coordinates": {

                "latitude":
                    destination["latitude"],

                "longitude":
                    destination["longitude"]
            },

            "distance_km":
                distance_km,

            "estimated_drive_hours":
                estimated_hours,

            "recommended_modes":
                recommended_modes
        }

    except Exception as e:

        print(
            f"Routing Error: {e}"
        )

        return None


# =====================================================
# Local Testing
# =====================================================

if __name__ == "__main__":

    result = get_transport_context(
    user_query="""
    I want a 5 day Goa trip
    from Bhubaneswar
    under 20000.
    """,
    destination="Goa"
    )

    print(result)