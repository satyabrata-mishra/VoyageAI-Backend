from typing import Literal


def route_after_destination(state) -> Literal["hotel", "end"]:
    destination_result = state.get("destination_result", {})

    if destination_result.get("status") != "success":
        return "end"

    return "hotel"


def route_after_budget(state) -> Literal["revise_plan", "itinerary"]:
    budget_result = state.get("budget_result", {})
    budget_status = budget_result.get("budget_status")
    revision_count = state.get("revision_count", 0)

    if budget_status == "over_budget" and revision_count < 1:
        return "revise_plan"

    return "itinerary"