from app.agents.destination_agent import run_destination_agent
from app.agents.hotel_agent import run_hotel_agent
from app.agents.food_agent import run_food_agent
from app.agents.transport_agent import run_transport_agent
from app.agents.weather_agent import run_weather_agent
from app.agents.budget_agent import run_budget_agent
from app.agents.itinerary_agent import run_itinerary_agent


AGENT_ORDER = [
    "Destination Agent",
    "Hotel Agent",
    "Food Agent",
    "Transport Agent",
    "Weather Agent",
    "Budget Agent",
    "Itinerary Agent"
]


def get_default_agent_status():
    return {
        "Destination Agent": "waiting",
        "Hotel Agent": "waiting",
        "Food Agent": "waiting",
        "Transport Agent": "waiting",
        "Weather Agent": "waiting",
        "Budget Agent": "waiting",
        "Itinerary Agent": "waiting"
    }


def update_agent_status(state, agent_name: str, status: str):
    agent_status = state.get("agent_status") or get_default_agent_status()
    agent_status = dict(agent_status)

    agent_status[agent_name] = status

    completed_count = sum(
        1 for value in agent_status.values()
        if value in ["completed", "skipped", "error"]
    )

    progress_percentage = int((completed_count / len(agent_status)) * 100)

    return {
        "agent_status": agent_status,
        "current_agent": agent_name if status == "running" else None,
        "progress_percentage": progress_percentage
    }


def mark_agent_completed(state, agent_name: str):
    return update_agent_status(state, agent_name, "completed")


def mark_agent_error(state, agent_name: str):
    return update_agent_status(state, agent_name, "error")


def mark_agent_skipped(state, agent_name: str):
    return update_agent_status(state, agent_name, "skipped")


def destination_node(state):
    agent_name = "Destination Agent"

    try:
        result = run_destination_agent(state["user_query"])

        status_update = mark_agent_completed(state, agent_name)

        return {
            "destination_result": result,
            "errors": state.get("errors", []),
            **status_update
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"{agent_name} failed: {str(e)}")

        status_update = mark_agent_error(state, agent_name)

        return {
            "destination_result": {
                "agent_name": agent_name,
                "status": "error",
                "recommended_destination": None,
                "reason": str(e),
                "confidence": "low"
            },
            "errors": errors,
            **status_update
        }


def hotel_node(state):
    agent_name = "Hotel Agent"

    try:
        result = run_hotel_agent(
            user_query=state["user_query"],
            destination_result=state.get("destination_result", {})
        )

        if result.get("status") == "skipped":
            status_update = mark_agent_skipped(state, agent_name)
        else:
            status_update = mark_agent_completed(state, agent_name)

        return {
            "hotel_result": result,
            "errors": state.get("errors", []),
            **status_update
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"{agent_name} failed: {str(e)}")

        status_update = mark_agent_error(state, agent_name)

        return {
            "hotel_result": {
                "agent_name": agent_name,
                "status": "error",
                "reason": str(e),
                "confidence": "low"
            },
            "errors": errors,
            **status_update
        }


def food_node(state):
    agent_name = "Food Agent"

    try:
        result = run_food_agent(
            user_query=state["user_query"],
            destination_result=state.get("destination_result", {})
        )

        if result.get("status") == "skipped":
            status_update = mark_agent_skipped(state, agent_name)
        else:
            status_update = mark_agent_completed(state, agent_name)

        return {
            "food_result": result,
            "errors": state.get("errors", []),
            **status_update
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"{agent_name} failed: {str(e)}")

        status_update = mark_agent_error(state, agent_name)

        return {
            "food_result": {
                "agent_name": agent_name,
                "status": "error",
                "reason": str(e),
                "confidence": "low"
            },
            "errors": errors,
            **status_update
        }


def transport_node(state):
    agent_name = "Transport Agent"

    try:
        result = run_transport_agent(
            user_query=state["user_query"],
            destination_result=state.get("destination_result", {})
        )

        if result.get("status") == "skipped":
            status_update = mark_agent_skipped(state, agent_name)
        else:
            status_update = mark_agent_completed(state, agent_name)

        return {
            "transport_result": result,
            "errors": state.get("errors", []),
            **status_update
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"{agent_name} failed: {str(e)}")

        status_update = mark_agent_error(state, agent_name)

        return {
            "transport_result": {
                "agent_name": agent_name,
                "status": "error",
                "reason": str(e),
                "confidence": "low"
            },
            "errors": errors,
            **status_update
        }


def weather_node(state):
    agent_name = "Weather Agent"

    try:
        result = run_weather_agent(
            user_query=state["user_query"],
            destination_result=state.get("destination_result", {})
        )

        if result.get("status") == "skipped":
            status_update = mark_agent_skipped(state, agent_name)
        else:
            status_update = mark_agent_completed(state, agent_name)

        return {
            "weather_result": result,
            "errors": state.get("errors", []),
            **status_update
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"{agent_name} failed: {str(e)}")

        status_update = mark_agent_error(state, agent_name)

        return {
            "weather_result": {
                "agent_name": agent_name,
                "status": "error",
                "reason": str(e),
                "confidence": "low"
            },
            "errors": errors,
            **status_update
        }


def budget_node(state):
    agent_name = "Budget Agent"

    try:
        result = run_budget_agent(
            user_query=state["user_query"],
            destination_result=state.get("destination_result", {}),
            hotel_result=state.get("hotel_result", {}),
            food_result=state.get("food_result", {}),
            transport_result=state.get("transport_result", {}),
            weather_result=state.get("weather_result", {})
        )

        if result.get("status") == "skipped":
            status_update = mark_agent_skipped(state, agent_name)
        else:
            status_update = mark_agent_completed(state, agent_name)

        return {
            "budget_result": result,
            "budget_status": result.get("budget_status"),
            "errors": state.get("errors", []),
            **status_update
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"{agent_name} failed: {str(e)}")

        status_update = mark_agent_error(state, agent_name)

        return {
            "budget_result": {
                "agent_name": agent_name,
                "status": "error",
                "reason": str(e),
                "budget_status": "unknown",
                "confidence": "low"
            },
            "budget_status": "unknown",
            "errors": errors,
            **status_update
        }


def revise_plan_node(state):
    revision_count = state.get("revision_count", 0) + 1

    revision_notes = state.get("revision_notes", [])
    revision_notes.append(
        "Budget was over limit. Revised plan toward budget stays, economical transport, local food, and fewer paid activities."
    )

    revised_query = (
        state["user_query"]
        + " Revise this plan to be more budget-friendly. "
        + "Use budget hotels or hostels, train or bus transport, local food, free sightseeing, and fewer paid activities. "
        + "Avoid luxury stays, flights, private cabs, and expensive activities."
    )

    agent_status = state.get("agent_status") or get_default_agent_status()
    agent_status = dict(agent_status)

    agent_status["Hotel Agent"] = "waiting"
    agent_status["Food Agent"] = "waiting"
    agent_status["Transport Agent"] = "waiting"
    agent_status["Weather Agent"] = "waiting"
    agent_status["Budget Agent"] = "waiting"
    agent_status["Itinerary Agent"] = "waiting"

    return {
        "user_query": revised_query,
        "revision_count": revision_count,
        "revision_notes": revision_notes,
        "agent_status": agent_status,
        "current_agent": "Revision Node"
    }


def itinerary_node(state):
    agent_name = "Itinerary Agent"

    try:
        result = run_itinerary_agent(
            user_query=state["user_query"],
            destination_result=state.get("destination_result", {}),
            hotel_result=state.get("hotel_result", {}),
            food_result=state.get("food_result", {}),
            transport_result=state.get("transport_result", {}),
            weather_result=state.get("weather_result", {}),
            budget_result=state.get("budget_result", {})
        )

        if result.get("status") == "skipped":
            status_update = mark_agent_skipped(state, agent_name)
        else:
            status_update = mark_agent_completed(state, agent_name)

        return {
            "itinerary_result": result,
            "errors": state.get("errors", []),
            **status_update
        }

    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"{agent_name} failed: {str(e)}")

        status_update = mark_agent_error(state, agent_name)

        return {
            "itinerary_result": {
                "agent_name": agent_name,
                "status": "error",
                "reason": str(e),
                "confidence": "low"
            },
            "errors": errors,
            **status_update
        }