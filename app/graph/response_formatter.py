def format_graph_response(final_state: dict):
    destination_result = final_state.get("destination_result", {})
    attraction_result = final_state.get("attraction_result", {})
    hotel_result = final_state.get("hotel_result", {})
    food_result = final_state.get("food_result", {})
    transport_result = final_state.get("transport_result", {})
    weather_result = final_state.get("weather_result", {})
    budget_result = final_state.get("budget_result", {})
    itinerary_result = final_state.get("itinerary_result", {})

    errors = final_state.get("errors", [])

    success = (
        destination_result.get("status") == "success"
        and itinerary_result.get("status") in ["success", "needs_budget_revision"]
    )

    return {
        "success": success,
        "message": "VoyageAI travel plan generated successfully." if success else "VoyageAI could not generate a complete travel plan.",
        "user_query": final_state.get("user_query"),
        "destination": destination_result.get("recommended_destination"),
        "budget_status": budget_result.get("budget_status"),
        "progress_percentage": final_state.get("progress_percentage", 0),
        "agent_status": final_state.get("agent_status", {}),
        "revision_count": final_state.get("revision_count", 0),
        "revision_notes": final_state.get("revision_notes", []),
        "agent_outputs": {
            "destination": destination_result,
            "attraction": attraction_result,
            "hotel": hotel_result,
            "food": food_result,
            "transport": transport_result,
            "weather": weather_result,
            "budget": budget_result,
            "itinerary": itinerary_result
        },
        "errors": errors
    }