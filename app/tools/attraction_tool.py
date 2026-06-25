import os
import requests

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENTRIPMAP_API_KEY")


# =====================================================
# Filters
# =====================================================

BAD_KEYWORDS = {

    "cross",
    "cemetery",
    "grave",
    "gate",
    "seminary",
    "excavated",
    "memorial",
    "monument",
    "statue",
    "roadside",
    "chapel",
    "shrine"
}


# =====================================================
# Attraction Classification
# =====================================================

def classify_attraction(kinds):

    kinds = kinds.lower()

    if "beaches" in kinds:
        return "Beach"

    elif "waterfalls" in kinds:
        return "Waterfall"

    elif "historic" in kinds:
        return "Heritage"

    elif "architecture" in kinds:
        return "Architecture"

    elif "religion" in kinds:
        return "Religious"

    elif "natural" in kinds:
        return "Nature"

    elif "museums" in kinds:
        return "Museum"

    return "Tourist Attraction"


# =====================================================
# Search Attractions
# =====================================================

def search_attractions(
    latitude,
    longitude,
    radius=50000,
    limit=100
):

    try:

        url = (
            "https://api.opentripmap.com/0.1/en/places/radius"
        )

        params = {

            "radius": radius,

            "lon": longitude,

            "lat": latitude,

            "rate": 2,

            "format": "json",

            "limit": limit,

            "apikey": API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        attractions = response.json()

        # Sort by popularity score
        attractions.sort(
            key=lambda x: x.get("rate", 0),
            reverse=True
        )

        return attractions

    except Exception as e:

        print(
            f"Attraction Search Error: {e}"
        )

        return []
    
def get_attraction_details(xid):

    try:

        url = (
            f"https://api.opentripmap.com/0.1/en/places/xid/{xid}"
        )

        params = {
            "apikey": API_KEY
        }

        response = requests.get(
            url,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        return response.json()

    except Exception:

        return None

def is_valid_attraction(
    name,
    kinds
):

    if not name:
        return False

    name_lower = name.lower()

    for keyword in BAD_KEYWORDS:

        if keyword in name_lower:
            return False

    if not kinds:
        return False

    return True

def enrich_attractions(
    attractions,
    max_results=25
):

    enriched = []

    for attraction in attractions:

        xid = attraction.get("xid")

        if not xid:
            continue

        details = get_attraction_details(
            xid
        )

        if not details:
            continue

        name = details.get(
            "name"
        )

        kinds = details.get(
            "kinds",
            ""
        )

        if not is_valid_attraction(
            name,
            kinds
        ):
            continue

        description = (
            details.get(
                "wikipedia_extracts",
                {}
            ).get(
                "text",
                ""
            )
        )

        point = details.get(
            "point",
            {}
        )

        enriched.append({

            "name":
                name,

            "type":
                classify_attraction(
                    kinds
                ),

            "kinds":
                kinds,

            "description":
                description[:500],

            "latitude":
                point.get("lat"),

            "longitude":
                point.get("lon"),

            "rate":
                attraction.get(
                    "rate",
                    0
                ),

            "xid":
                xid
        })

        if len(enriched) >= max_results:
            break

    # Final popularity sort

    enriched.sort(
        key=lambda x: x["rate"],
        reverse=True
    )

    return enriched

if __name__ == "__main__":

    raw_attractions = search_attractions(
        latitude=15.2993,
        longitude=74.1240,
        radius=50000,
        limit=100
    )

    attractions = enrich_attractions(
        raw_attractions
    )

    print("\nTop Attractions\n")

    for attraction in attractions[:25]:

        print(
            attraction["name"],
            "|",
            attraction["type"],
            "| Rate:",
            attraction["rate"]
        )