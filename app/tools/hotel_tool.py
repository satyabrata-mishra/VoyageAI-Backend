import os
import requests

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEOAPIFY_API_KEY")


def search_hotels(
    latitude,
    longitude,
    radius=10000,
    limit=20
):
    """
    Search hotels near coordinates.
    """

    try:

        url = (
            "https://api.geoapify.com/v2/places"
        )

        params = {
            "categories": (
                "accommodation.hotel,"
                "accommodation.guest_house,"
                "accommodation.hostel"
            ),
            "filter":
                f"circle:{longitude},{latitude},{radius}",
            "limit": limit,
            "apiKey": API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=15
        )
        response.raise_for_status()

        data = response.json()

        hotels = []

        for item in data.get(
            "features",
            []
        ):

            properties = item.get(
                "properties",
                {}
            )

            hotels.append({

                "name":
                    properties.get(
                        "name"
                    ),

                "address":
                    properties.get(
                        "formatted"
                    ),

                "city":
                    properties.get(
                        "city"
                    ),

                "country":
                    properties.get(
                        "country"
                    ),

                "latitude":
                    properties.get(
                        "lat"
                    ),

                "longitude":
                    properties.get(
                        "lon"
                    ),

                "categories":
                    properties.get(
                        "categories",
                        []
                    )
            })

        return hotels

    except Exception as e:

        print(
            f"Geoapify Hotel Search Error: {e}"
        )

        return []
    
if __name__ == "__main__":
    # Example usage
    hotels = search_hotels(15.2993, 74.1240, radius=50000, limit=10)
    for hotel in hotels:
        print(hotel)