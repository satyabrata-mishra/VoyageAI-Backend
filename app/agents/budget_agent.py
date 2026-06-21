import re
import json
import math
from typing import Any


HOTEL_COST_PER_NIGHT = {
    "budget": 1200,
    "comfort": 3000,
    "luxury": 8000,
    "unknown": 2500,
}

FOOD_COST_PER_DAY = {
    "budget": 700,
    "comfort": 1500,
    "luxury": 3000,
    "unknown": 1200,
}

LOCAL_TRANSPORT_PER_DAY = {
    "budget": 400,
    "comfort": 1000,
    "luxury": 2500,
    "unknown": 700,
}

ACTIVITY_COST_PER_DAY = {
    "budget": 600,
    "comfort": 1500,
    "luxury": 3500,
    "unknown": 1000,
}

INTERCITY_TRANSPORT_COST = {
    "train": {
        "budget": 3000,
        "comfort": 6000,
        "luxury": 10000,
        "unknown": 6000,
    },
    "bus": {
        "budget": 2500,
        "comfort": 5000,
        "luxury": 8000,
        "unknown": 5000,
    },
    "flight": {
        "budget": 7000,
        "comfort": 12000,
        "luxury": 25000,
        "unknown": 12000,
    },
    "car": {
        "budget": 8000,
        "comfort": 14000,
        "luxury": 25000,
        "unknown": 14000,
    },
    "ferry": {
        "budget": 1500,
        "comfort": 3000,
        "luxury": 7000,
        "unknown": 3000,
    },
    "mixed": {
        "budget": 6000,
        "comfort": 10000,
        "luxury": 18000,
        "unknown": 10000,
    },
    "unknown": {
        "budget": 5000,
        "comfort": 9000,
        "luxury": 15000,
        "unknown": 9000,
    },
}

DESTINATION_COST_MULTIPLIER = {
    "goa": 1.10,
    "jaipur": 1.00,
    "manali": 1.10,
    "kerala": 1.15,
    "andaman": 1.40,
    "andaman and nicobar islands": 1.40,
    "udaipur": 1.20,
    "rishikesh": 0.90,
    "varanasi": 0.85,
    "ladakh": 1.45,
    "kashmir": 1.30,
    "darjeeling": 1.10,
    "pondicherry": 1.05,
    "agra": 0.95,
    "delhi": 1.10,
    "mumbai": 1.30,
    "unknown": 1.00,
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    return text.lower().strip()


def extract_user_budget(user_query: str) -> int | None:
    query = normalize_text(user_query)
    query = query.replace(",", "")

    patterns = [
        r"(?:under|within|below|less than|budget is|budget of|budget around|around)\s*₹?\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?",
        r"₹\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?",
        r"rs\.?\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?",
        r"inr\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lakhs)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)

        if match:
            amount = float(match.group(1))
            unit = match.group(2)

            if unit in ["k", "thousand"]:
                amount *= 1000
            elif unit in ["lakh", "lakhs"]:
                amount *= 100000

            return int(amount)

    return None


def extract_duration_from_text(text: str) -> int | None:
    text = normalize_text(text)

    day_match = re.search(r"(\d+)\s*(day|days)", text)
    if day_match:
        return int(day_match.group(1))

    night_match = re.search(r"(\d+)\s*(night|nights)", text)
    if night_match:
        nights = int(night_match.group(1))
        return nights + 1

    return None


def extract_duration_from_destination_result(
    destination_result: dict[str, Any],
) -> int | None:
    suggested_duration = destination_result.get("suggested_duration")

    if not suggested_duration:
        return None

    text = normalize_text(suggested_duration)
    numbers = re.findall(r"\d+", text)

    if not numbers:
        return None

    numbers = [int(num) for num in numbers]

    if len(numbers) == 1:
        return numbers[0]

    return math.ceil(sum(numbers) / len(numbers))


def extract_trip_duration(
    user_query: str,
    destination_result: dict[str, Any],
) -> int:
    duration = extract_duration_from_text(user_query)

    if duration:
        return duration

    duration = extract_duration_from_destination_result(destination_result)

    if duration:
        return duration

    return 4


def detect_travel_style(
    user_query: str,
    hotel_result: dict[str, Any] | None = None,
    transport_result: dict[str, Any] | None = None,
) -> str:
    query = normalize_text(user_query)

    budget_keywords = [
        "budget",
        "low budget",
        "cheap",
        "affordable",
        "economical",
        "backpacking",
        "hostel",
    ]

    luxury_keywords = [
        "luxury",
        "premium",
        "5 star",
        "five star",
        "resort",
        "private",
        "honeymoon luxury",
    ]

    comfort_keywords = [
        "comfort",
        "comfortable",
        "mid range",
        "mid-range",
        "boutique",
        "family",
    ]

    if any(keyword in query for keyword in budget_keywords):
        return "budget"

    if any(keyword in query for keyword in luxury_keywords):
        return "luxury"

    if any(keyword in query for keyword in comfort_keywords):
        return "comfort"

    if hotel_result:
        hotel_budget = normalize_text(hotel_result.get("budget_preference"))

        if hotel_budget in ["budget", "comfort", "luxury"]:
            return hotel_budget

    if transport_result:
        transport_style = normalize_text(
            transport_result.get("travel_style_detected")
        )

        if transport_style in ["budget", "comfort", "luxury"]:
            return transport_style

    return "unknown"


def detect_transport_mode(
    transport_result: dict[str, Any] | None = None,
) -> str:
    if not transport_result:
        return "unknown"

    mode = normalize_text(transport_result.get("recommended_intercity_mode"))

    valid_modes = ["flight", "train", "bus", "car", "ferry", "mixed"]

    if mode in valid_modes:
        return mode

    if "flight" in mode:
        return "flight"

    if "train" in mode:
        return "train"

    if "bus" in mode:
        return "bus"

    if "car" in mode or "cab" in mode or "taxi" in mode:
        return "car"

    if "ferry" in mode:
        return "ferry"

    if "mixed" in mode:
        return "mixed"

    return "unknown"


def normalize_destination(destination: str | None) -> str:
    destination = normalize_text(destination)

    if "andaman" in destination:
        return "andaman and nicobar islands"

    if "puducherry" in destination:
        return "pondicherry"

    return destination if destination else "unknown"


def calculate_budget(
    destination: str,
    trip_duration_days: int,
    travel_style: str,
    transport_mode: str,
) -> dict[str, int]:
    destination_key = normalize_destination(destination)

    multiplier = DESTINATION_COST_MULTIPLIER.get(destination_key, 1.00)

    nights = max(trip_duration_days - 1, 1)

    hotel_base = HOTEL_COST_PER_NIGHT.get(
        travel_style,
        HOTEL_COST_PER_NIGHT["unknown"],
    )

    food_base = FOOD_COST_PER_DAY.get(
        travel_style,
        FOOD_COST_PER_DAY["unknown"],
    )

    local_transport_base = LOCAL_TRANSPORT_PER_DAY.get(
        travel_style,
        LOCAL_TRANSPORT_PER_DAY["unknown"],
    )

    activity_base = ACTIVITY_COST_PER_DAY.get(
        travel_style,
        ACTIVITY_COST_PER_DAY["unknown"],
    )

    transport_cost_map = INTERCITY_TRANSPORT_COST.get(
        transport_mode,
        INTERCITY_TRANSPORT_COST["unknown"],
    )

    intercity_transport = transport_cost_map.get(
        travel_style,
        transport_cost_map["unknown"],
    )

    hotel_cost = int(hotel_base * nights * multiplier)
    food_cost = int(food_base * trip_duration_days * multiplier)
    activities_cost = int(activity_base * trip_duration_days * multiplier)
    local_transport_cost = int(
        local_transport_base * trip_duration_days * multiplier
    )

    subtotal = (
        intercity_transport
        + hotel_cost
        + food_cost
        + activities_cost
        + local_transport_cost
    )

    buffer = int(subtotal * 0.10)
    total = subtotal + buffer

    return {
        "transport": intercity_transport,
        "hotel": hotel_cost,
        "food": food_cost,
        "activities": activities_cost,
        "local_transport": local_transport_cost,
        "buffer": buffer,
        "total": total,
    }


def get_budget_status(
    total_estimated_cost: int,
    user_budget: int | None = None,
) -> tuple[str, int | None]:
    if user_budget is None:
        return "budget_not_provided", None

    remaining_budget = user_budget - total_estimated_cost

    if total_estimated_cost <= user_budget:
        return "within_budget", remaining_budget

    return "over_budget", remaining_budget


def generate_cost_saving_suggestions(
    budget_status: str,
    travel_style: str,
    transport_mode: str,
    hotel_result: dict[str, Any] | None = None,
) -> list[str]:
    suggestions = []

    if budget_status == "within_budget":
        suggestions.extend(
            [
                "Keep a small emergency buffer for local transport or activity changes.",
                "Book stays and transport early during peak season.",
                "Track daily food and activity spending to stay within budget.",
            ]
        )

        return suggestions

    if budget_status == "budget_not_provided":
        suggestions.extend(
            [
                "Provide a target budget to check if the trip is affordable.",
                "Choose budget stays if you want a lower overall estimate.",
                "Select train or bus instead of flight where practical.",
            ]
        )

        return suggestions

    if transport_mode == "flight":
        suggestions.append(
            "Consider train or bus instead of flight to reduce intercity transport cost."
        )
    elif transport_mode == "car":
        suggestions.append(
            "Consider shared transport, train, or bus instead of private car to reduce cost."
        )
    else:
        suggestions.append(
            "Compare transport options and choose the most economical available mode."
        )

    if travel_style in ["comfort", "luxury", "unknown"]:
        suggestions.append(
            "Switch to budget hotels, hostels, guesthouses, or homestays."
        )

    suggestions.extend(
        [
            "Reduce paid activities and include more free sightseeing.",
            "Use local transport instead of private cabs where safe and practical.",
            "Avoid peak season dates if budget is tight.",
            "Keep luxury meals limited and mix them with local food options.",
        ]
    )

    return suggestions


class BudgetAgent:
    def run(
        self,
        user_query: str,
        destination_result: dict[str, Any],
        hotel_result: dict[str, Any] | None = None,
        food_result: dict[str, Any] | None = None,
        transport_result: dict[str, Any] | None = None,
        weather_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        destination_status = destination_result.get("status")
        destination = destination_result.get("recommended_destination")

        if destination_status != "success":
            return {
                "agent_name": "Budget Agent",
                "status": "skipped",
                "destination": None,
                "message": (
                    "Budget Agent skipped because Destination Agent did not "
                    "return a valid destination."
                ),
                "reason": destination_result.get("reason"),
                "confidence": "low",
            }

        if not destination:
            return {
                "agent_name": "Budget Agent",
                "status": "no_destination_found",
                "destination": None,
                "message": (
                    "Budget Agent cannot run because no destination was provided."
                ),
                "confidence": "low",
            }

        user_budget = extract_user_budget(user_query)
        trip_duration_days = extract_trip_duration(
            user_query=user_query,
            destination_result=destination_result,
        )

        travel_style = detect_travel_style(
            user_query=user_query,
            hotel_result=hotel_result,
            transport_result=transport_result,
        )

        transport_mode = detect_transport_mode(transport_result)

        cost_result = calculate_budget(
            destination=destination,
            trip_duration_days=trip_duration_days,
            travel_style=travel_style,
            transport_mode=transport_mode,
        )

        total_estimated_cost = cost_result["total"]

        budget_status, remaining_budget = get_budget_status(
            total_estimated_cost=total_estimated_cost,
            user_budget=user_budget,
        )

        suggestions = generate_cost_saving_suggestions(
            budget_status=budget_status,
            travel_style=travel_style,
            transport_mode=transport_mode,
            hotel_result=hotel_result,
        )

        warnings = []

        if hotel_result and hotel_result.get("status") != "success":
            warnings.append(
                "Hotel Agent did not return a successful result, so hotel cost is estimated using general rules."
            )

        if transport_result and transport_result.get("status") != "success":
            warnings.append(
                "Transport Agent did not return a successful result, so transport cost is estimated using general rules."
            )

        if weather_result and weather_result.get("status") != "success":
            warnings.append(
                "Weather Agent did not return a successful result. Weather risk is not included in cost."
            )

        return {
            "agent_name": "Budget Agent",
            "status": "success",
            "destination": destination,
            "trip_duration_days": trip_duration_days,
            "estimated_nights": max(trip_duration_days - 1, 1),
            "user_budget": user_budget,
            "currency": "INR",
            "budget_level_detected": travel_style,
            "transport_mode_used": transport_mode,
            "estimated_cost_breakdown": {
                "transport": cost_result["transport"],
                "hotel": cost_result["hotel"],
                "food": cost_result["food"],
                "activities": cost_result["activities"],
                "local_transport": cost_result["local_transport"],
                "buffer": cost_result["buffer"],
            },
            "total_estimated_cost": total_estimated_cost,
            "remaining_budget": remaining_budget,
            "budget_status": budget_status,
            "cost_saving_suggestions": suggestions,
            "warnings": warnings,
            "limitations": (
                "This is a rule-based estimate. It does not use live hotel, "
                "train, flight, bus, or activity prices."
            ),
            "confidence": "medium",
        }


_budget_agent_instance = None


def get_budget_agent() -> BudgetAgent:
    global _budget_agent_instance

    if _budget_agent_instance is None:
        _budget_agent_instance = BudgetAgent()

    return _budget_agent_instance


def run_budget_agent(
    user_query: str,
    destination_result: dict[str, Any],
    hotel_result: dict[str, Any] | None = None,
    food_result: dict[str, Any] | None = None,
    transport_result: dict[str, Any] | None = None,
    weather_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agent = get_budget_agent()

    return agent.run(
        user_query=user_query,
        destination_result=destination_result,
        hotel_result=hotel_result,
        food_result=food_result,
        transport_result=transport_result,
        weather_result=weather_result,
    )


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Low budget Goa trip under 20000",
            "user_query": (
                "I want a 4 day Goa trip from Bhubaneswar under 20000 "
                "with beaches, seafood and nightlife. Keep it low budget."
            ),
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Goa",
                "reason": "Goa matches beaches, seafood, nightlife and relaxed coastal travel.",
                "suitable_for": [
                    "beach lovers",
                    "food lovers",
                    "nightlife travelers",
                ],
                "suggested_duration": "3 to 5 days",
                "best_time_to_visit": "November to February",
                "confidence": "high",
            },
            "hotel_result": {
                "agent_name": "Hotel Agent",
                "status": "success",
                "destination": "Goa",
                "budget_preference": "budget",
                "recommended_stay_type": (
                    "Hostel or budget stay near Anjuna, Vagator, or Baga"
                ),
                "best_areas": ["Anjuna", "Vagator", "Baga"],
                "confidence": "high",
            },
            "transport_result": {
                "agent_name": "Transport Agent",
                "status": "success",
                "source_city": "Bhubaneswar",
                "destination": "Goa",
                "travel_style_detected": "budget",
                "recommended_intercity_mode": "train",
                "confidence": "medium",
            },
            "food_result": {
                "agent_name": "Food Agent",
                "status": "success",
                "destination": "Goa",
                "food_preference_detected": "seafood",
                "recommended_foods": [
                    "Goan fish curry",
                    "Prawn balchao",
                    "Bebinca",
                ],
                "confidence": "high",
            },
            "weather_result": {
                "agent_name": "Weather Agent",
                "status": "success",
                "destination": "Goa",
                "season_suitability": "high",
                "confidence": "high",
            },
        },
        {
            "name": "Manali trip without user budget",
            "user_query": "I want a 4 day trip to Manali for snow and adventure.",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "success",
                "recommended_destination": "Manali",
                "reason": "Manali matches snow, mountains and adventure.",
                "suggested_duration": "4 to 6 days",
                "confidence": "high",
            },
            "hotel_result": {
                "agent_name": "Hotel Agent",
                "status": "success",
                "destination": "Manali",
                "budget_preference": "unknown",
                "recommended_stay_type": "Riverside cottages or hostels in Old Manali",
                "confidence": "medium",
            },
            "transport_result": {
                "agent_name": "Transport Agent",
                "status": "success",
                "destination": "Manali",
                "travel_style_detected": "unknown",
                "recommended_intercity_mode": "mixed",
                "confidence": "medium",
            },
            "food_result": None,
            "weather_result": None,
        },
        {
            "name": "Destination Agent failed",
            "user_query": "I want a 5 day Venice trip under 50000",
            "destination_result": {
                "agent_name": "Destination Agent",
                "status": "out_of_knowledge_base",
                "recommended_destination": None,
                "reason": "Venice is not available in the current VoyageAI knowledge base.",
                "confidence": "low",
            },
            "hotel_result": None,
            "transport_result": None,
            "food_result": None,
            "weather_result": None,
        },
    ]

    budget_agent = BudgetAgent()

    for test_case in test_cases:
        print("=" * 100)
        print("Test Case:", test_case["name"])
        print("User Query:", test_case["user_query"])
        print("-" * 100)

        result = budget_agent.run(
            user_query=test_case["user_query"],
            destination_result=test_case["destination_result"],
            hotel_result=test_case.get("hotel_result"),
            food_result=test_case.get("food_result"),
            transport_result=test_case.get("transport_result"),
            weather_result=test_case.get("weather_result"),
        )

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()