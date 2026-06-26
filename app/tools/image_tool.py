import os
import requests

from dotenv import load_dotenv

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

PEXELS_URL = "https://api.pexels.com/v1/search"

HEADERS = {
    "Authorization": PEXELS_API_KEY
}


# ==========================================================
# Search Pexels
# ==========================================================

def search_pexels(
    query: str,
    per_page: int = 1
):
    """
    Search Pexels for a location/place.

    Returns:
    {
        "success": True,
        "image_url": "...",
        "photographer": "...",
        "source": "Pexels"
    }

    or

    {
        "success": False
    }
    """

    try:

        response = requests.get(

            PEXELS_URL,

            headers=HEADERS,

            params={
                "query": query,
                "per_page": per_page
            },

            timeout=15

        )

        response.raise_for_status()

        data = response.json()

        photos = data.get("photos", [])

        if not photos:

            return {
                "success": False
            }

        photo = photos[0]

        return {

            "success": True,

            "image_url":
                photo["src"]["large"],

            "thumbnail":
                photo["src"]["medium"],

            "photographer":
                photo["photographer"],

            "photographer_url":
                photo["photographer_url"],

            "pexels_page":
                photo["url"],

            "source":
                "Pexels"

        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)

        }


# ==========================================================
# Public Function
# ==========================================================

def get_place_image(place_name: str):
    """
    Fetch image for any place.

    Examples:

    Goa

    Fort Aguada

    Grand Hyatt Goa

    Martin's Corner

    Baga Beach
    """

    result = search_pexels(place_name)

    if result["success"]:
        return result

    return {

        "success": False,

        "image_url": None,

        "thumbnail": None,

        "photographer": None,

        "photographer_url": None,

        "pexels_page": None,

        "source": None

    }
    
def enrich_destination_images(destination_name):
    return get_place_image(f"{destination_name} travel destination")


def enrich_hotel_images(hotel_name):
    return get_place_image(f"{hotel_name} hotel")


def enrich_restaurant_images(restaurant_name):
    return get_place_image(f"{restaurant_name} restaurant")


def enrich_attraction_images(attraction_name):
    return get_place_image(f"{attraction_name} attraction")


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    tests = [

        "Goa",

        "Fort Aguada",

        "Grand Hyatt Goa",

        "Martin's Corner Goa",

        "Baga Beach"

    ]

    for place in tests:

        print("\n=========================")
        print(place)

        result = get_place_image(place)

        print(result)