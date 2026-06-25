import json

from app.graph.workflow import travel_graph
from app.graph.nodes import get_default_agent_status

from app.graph.response_formatter import format_graph_response


def run_test():
    initial_state = {
        "user_query": "I want a 4 day Goa trip from Bhubaneswar under 20000 with beaches, seafood and nightlife. Keep it low budget.",
        "revision_count": 0,
        "revision_notes": [],
        "agent_status": get_default_agent_status(),
        "current_agent": None,
        "progress_percentage": 0,
        "errors": []
    }

    final_state = travel_graph.invoke(initial_state)
    formatted_response = format_graph_response(final_state)

    print("\n========== FORMATTED API RESPONSE ==========\n")
    print(json.dumps(formatted_response, indent=2))

    # print("\n========== FINAL VOYAGEAI STATE ==========\n")

    # print("Agent Status:")
    # print(json.dumps(final_state.get("agent_status"), indent=2))

    # print("\nProgress Percentage:")
    # print(final_state.get("progress_percentage"))

    # print("\nCurrent Agent:")
    # print(final_state.get("current_agent"))

    # print("\nDestination Result:")
    # print(json.dumps(final_state.get("destination_result"), indent=2))
    
    # print("\nAttraction Result:")
    # print(json.dumps(final_state.get("attraction_result"), indent=2))

    # print("\nHotel Result:")
    # print(json.dumps(final_state.get("hotel_result"), indent=2))

    # print("\nFood Result:")
    # print(json.dumps(final_state.get("food_result"), indent=2))

    # print("\nTransport Result:")
    # print(json.dumps(final_state.get("transport_result"), indent=2))

    # print("\nWeather Result:")
    # print(json.dumps(final_state.get("weather_result"), indent=2))

    # print("\nBudget Result:")
    # print(json.dumps(final_state.get("budget_result"), indent=2))

    # print("\nItinerary Result:")
    # print(json.dumps(final_state.get("itinerary_result"), indent=2))

    # print("\nRevision Count:")
    # print(final_state.get("revision_count"))

    # print("\nRevision Notes:")
    # print(json.dumps(final_state.get("revision_notes"), indent=2))

    # print("\nErrors:")
    # print(json.dumps(final_state.get("errors"), indent=2))


if __name__ == "__main__":
    run_test()