def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return text.lower().strip()


DESTINATION_KEY_MAP = {
    "goa": "goa",
    "jaipur": "jaipur",
    "manali": "manali",
    "kerala": "kerala",
    "andaman": "andaman",
    "andaman and nicobar islands": "andaman",
    "udaipur": "udaipur",
    "rishikesh": "rishikesh",
    "varanasi": "varanasi",
    "ladakh": "ladakh",
    "kashmir": "kashmir",
    "darjeeling": "darjeeling",
    "pondicherry": "pondicherry",
    "puducherry": "pondicherry",
    "agra": "agra",
    "delhi": "delhi",
    "mumbai": "mumbai",
}


def normalize_destination_to_key(destination_name: str | None) -> str | None:
    if not destination_name:
        return None

    destination_name = destination_name.lower().strip()

    for name, key in DESTINATION_KEY_MAP.items():
        if name in destination_name:
            return key

    return destination_name.replace(" ", "_")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return text.lower().strip()


def get_available_destinations(vector_store) -> list[str]:
    """
    Reads all destination names from ChromaDB metadata.
    Assumes metadata contains a `city` field.
    """
    collection_data = vector_store._collection.get(include=["metadatas"])

    destinations = set()

    for metadata in collection_data.get("metadatas", []):
        if not metadata:
            continue

        city = metadata.get("city")

        if city:
            destinations.add(city.lower().strip())

    return sorted(list(destinations))


def is_destination_in_kb(
    destination_name: str | None,
    available_destinations: list[str],
) -> bool:
    destination_name = normalize_text(destination_name)

    if not destination_name:
        return False

    for destination in available_destinations:
        destination = normalize_text(destination)

        if destination == destination_name:
            return True

        if destination in destination_name:
            return True

        if destination_name in destination:
            return True

    return False


def format_available_destinations(available_destinations: list[str]) -> str:
    return ", ".join(available_destinations)