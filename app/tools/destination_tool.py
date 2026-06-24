import os
import requests

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENTRIPMAP_API_KEY")
BASE_URL = "https://api.opentripmap.com/0.1/en/places"

def search_destination(city_name: str):
    """
    Search destination and return coordinates.
    """

    try:
        url = (
            f"{BASE_URL}/geoname"
            f"?name={city_name}"
            f"&apikey={API_KEY}"
        )
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"API returned {response.status_code}"
            }

        data = response.json()

        return {
            "success": True,
            "name": data.get("name"),
            "country": data.get("country"),
            "latitude": data.get("lat"),
            "longitude": data.get("lon"),
            "population": data.get("population")
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
        
def get_top_attractions(
    latitude,
    longitude,
    radius=10000,
    limit=10
):
    """
    Get attractions near coordinates.
    """

    try:

        url = (
            f"{BASE_URL}/radius"
            f"?radius={radius}"
            f"&lon={longitude}"
            f"&lat={latitude}"
            f"&rate=3"
            f"&limit={limit}"
            f"&format=json"
            f"&apikey={API_KEY}"
        )

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()

        attractions = []

        for item in data:

            name = item.get("name")

            if not name:
                continue

            attractions.append({
                "name": name,
                "kind": item.get("kinds"),
                "distance": round(
                    item.get("dist", 0),
                    2
                )
            })

        return attractions

    except Exception:
        return []
    
if __name__ == "__main__":
    city_name = "Goa"
    result = search_destination(city_name)
    print(result)

    if result["success"]:
        latitude = result["latitude"]
        longitude = result["longitude"]
        attractions = get_top_attractions(latitude, longitude)
        print(attractions)