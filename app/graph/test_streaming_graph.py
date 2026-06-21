import json

from app.graph.workflow import travel_graph
from app.graph.nodes import get_default_agent_status
from app.graph.response_formatter import format_graph_response


def print_stream_event(event):
    print("\n========== STREAM EVENT ==========")

    if not isinstance(event, dict):
        print(event)
        return

    for node_name, node_update in event.items():
        print(f"\nNode executed: {node_name}")

        if not isinstance(node_update, dict):
            print(node_update)
            continue

        agent_status = node_update.get("agent_status")
        current_agent = node_update.get("current_agent")
        progress_percentage = node_update.get("progress_percentage")

        if current_agent:
            print("Current Agent:", current_agent)

        if progress_percentage is not None:
            print("Progress:", progress_percentage, "%")

        if agent_status:
            print("Agent Status:")
            print(json.dumps(agent_status, indent=2))

        # Print compact agent outputs if available
        for key in [
            "destination_result",
            "hotel_result",
            "food_result",
            "transport_result",
            "weather_result",
            "budget_result",
            "itinerary_result"
        ]:
            if key in node_update:
                result = node_update[key]

                print(f"\n{key}:")
                print(json.dumps(result, indent=2))


def run_streaming_test():
    initial_state = {
        "user_query": "I want a 4 day Goa trip from Bhubaneswar under 20000 with beaches, seafood and nightlife. Keep it low budget.",
        "revision_count": 0,
        "revision_notes": [],
        "agent_status": get_default_agent_status(),
        "current_agent": None,
        "progress_percentage": 0,
        "errors": []
    }

    print("\nStarting VoyageAI streaming graph test...\n")

    final_state = None

    for event in travel_graph.stream(
        initial_state,
        stream_mode="updates"
    ):
        print_stream_event(event)

    print("\nStreaming completed.")

    # Run normal invoke once to get final full state for formatted output.
    # Later in FastAPI, we can collect stream updates and final output separately.
    final_state = travel_graph.invoke(initial_state)

    formatted_response = format_graph_response(final_state)

    print("\n========== FINAL FORMATTED RESPONSE ==========\n")
    print(json.dumps(formatted_response, indent=2))


if __name__ == "__main__":
    run_streaming_test()