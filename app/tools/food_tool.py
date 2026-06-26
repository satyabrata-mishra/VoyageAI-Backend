import os
import requests

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEOAPIFY_API_KEY")

BASE_URL = "https://api.geoapify.com/v2/places"


# =====================================================
# Search Restaurants
# =====================================================

def search_restaurants(
    latitude: float,
    longitude: float,
    radius: int = 5000,
    limit: int = 30
):
    """
    Search restaurants around a destination.

    Returns:
        List[dict]
    """

    try:

        url = BASE_URL

        params = {
            "categories": ",".join([
                "catering.restaurant",
                "catering.fast_food",
                "catering.cafe"
            ]),
            "filter": f"circle:{longitude},{latitude},{radius}",
            "bias": f"proximity:{longitude},{latitude}",
            "limit": limit,
            "apiKey": API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        restaurants = []

        for feature in data.get("features", []):

            props = feature.get("properties", {})

            name = props.get("name")

            if not name:
                continue

            restaurants.append({

                "place_id":
                    props.get("place_id"),

                "name":
                    name,

                "address":
                    props.get("formatted"),

                "categories":
                    props.get("categories", []),

                "latitude":
                    props.get("lat"),

                "longitude":
                    props.get("lon")
            })

        return restaurants

    except Exception as e:

        print(f"Restaurant Search Error: {e}")

        return []


# =====================================================
# Restaurant Details
# =====================================================

def get_restaurant_details(place_id):
    """
    Fetch additional restaurant details.

    Returns:
        dict | None
    """

    try:

        url = BASE_URL

        params = {
            "categories": ",".join([
                "catering.restaurant",
                "catering.fast_food",
                "catering.cafe"
            ]),
            "filter": f"place:{place_id}",
            "limit": 1,
            "apiKey": API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        features = response.json().get(
            "features",
            []
        )

        if not features:
            return None

        props = features[0]["properties"]

        return {

            "place_id":
                props.get("place_id"),

            "name":
                props.get("name"),

            "address":
                props.get("formatted"),

            "website":
                props.get("website"),

            "phone":
                props.get("contact", {}).get("phone"),

            "opening_hours":
                props.get("opening_hours"),

            "categories":
                props.get("categories", []),

            "latitude":
                props.get("lat"),

            "longitude":
                props.get("lon")
        }

    except Exception:

        return None


# =====================================================
# Enrich Restaurants
# =====================================================

def enrich_restaurants(
    restaurants,
    max_results=15
):
    """
    Enrich restaurants with detailed information.
    """

    enriched = []

    seen = set()

    for restaurant in restaurants:

        place_id = restaurant.get("place_id")

        if not place_id:
            continue

        if place_id in seen:
            continue

        seen.add(place_id)

        details = get_restaurant_details(
            place_id
        )

        if details:
            enriched.append(details)
        else:
            enriched.append(restaurant)

        if len(enriched) >= max_results:
            break

    return enriched


# =====================================================
# Local Testing
# =====================================================

if __name__ == "__main__":

    # Goa Coordinates
    lat = 15.2993
    lon = 74.1240

    restaurants = search_restaurants(
        latitude=lat,
        longitude=lon,
        radius=50000,
        limit=30
    )

    restaurants = enrich_restaurants(
        restaurants
    )

    print("\nTop Restaurants\n")

    for restaurant in restaurants:
        
        print("-" * 60)
        print("Name :", restaurant["name"])
        print("Address :", restaurant["address"])
        print("Categories :", restaurant["categories"])
        print()