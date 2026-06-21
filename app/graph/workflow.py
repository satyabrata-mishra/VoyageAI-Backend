from langgraph.graph import StateGraph, START, END

from app.graph.state import VoyageAIState
from app.graph.nodes import (
    destination_node,
    hotel_node,
    food_node,
    transport_node,
    weather_node,
    budget_node,
    revise_plan_node,
    itinerary_node
)
from app.graph.router import (
    route_after_destination,
    route_after_budget
)


def build_travel_graph():
    graph = StateGraph(VoyageAIState)

    graph.add_node("destination", destination_node)
    graph.add_node("hotel", hotel_node)
    graph.add_node("food", food_node)
    graph.add_node("transport", transport_node)
    graph.add_node("weather", weather_node)
    graph.add_node("budget", budget_node)
    graph.add_node("revise_plan", revise_plan_node)
    graph.add_node("itinerary", itinerary_node)

    graph.add_edge(START, "destination")

    graph.add_conditional_edges(
        "destination",
        route_after_destination,
        {
            "hotel": "hotel",
            "end": END
        }
    )

    graph.add_edge("hotel", "food")
    graph.add_edge("food", "transport")
    graph.add_edge("transport", "weather")
    graph.add_edge("weather", "budget")

    graph.add_conditional_edges(
        "budget",
        route_after_budget,
        {
            "revise_plan": "revise_plan",
            "itinerary": "itinerary"
        }
    )

    graph.add_edge("revise_plan", "hotel")
    graph.add_edge("itinerary", END)

    return graph.compile()


travel_graph = build_travel_graph()