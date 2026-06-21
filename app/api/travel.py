from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.schemas.travel_schema import TravelPlanRequest, TravelPlanResponse
from app.graph.workflow import travel_graph
from app.graph.nodes import get_default_agent_status
from app.graph.response_formatter import format_graph_response

router = APIRouter(
    prefix="/api/travel",
    tags=["Travel"]
)


def format_sse(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


@router.post("/plan", response_model=TravelPlanResponse)
def create_travel_plan(request: TravelPlanRequest):
    try:
        initial_state = {
            "user_query": request.user_query,
            "revision_count": 0,
            "revision_notes": [],
            "agent_status": get_default_agent_status(),
            "current_agent": None,
            "progress_percentage": 0,
            "errors": []
        }

        final_state = travel_graph.invoke(initial_state)

        formatted_response = format_graph_response(final_state)

        return formatted_response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate travel plan: {str(e)}"
        )


@router.post("/plan/stream")
def stream_travel_plan(request: TravelPlanRequest):
    def event_generator():
        initial_state = {
            "user_query": request.user_query,
            "revision_count": 0,
            "revision_notes": [],
            "agent_status": get_default_agent_status(),
            "current_agent": None,
            "progress_percentage": 0,
            "errors": []
        }

        current_state = dict(initial_state)

        # Send initial event
        yield format_sse(
            "start",
            {
                "success": True,
                "message": "VoyageAI streaming started.",
                "user_query": request.user_query,
                "progress_percentage": 0,
                "agent_status": current_state["agent_status"]
            }
        )

        try:
            for event in travel_graph.stream(initial_state, stream_mode="updates"):
                if not isinstance(event, dict):
                    continue

                for node_name, node_update in event.items():
                    if isinstance(node_update, dict):
                        # Merge top-level updates into current state
                        current_state.update(node_update)

                    payload = {
                        "node": node_name,
                        "progress_percentage": current_state.get("progress_percentage", 0),
                        "current_agent": current_state.get("current_agent"),
                        "agent_status": current_state.get("agent_status", {}),
                        "revision_count": current_state.get("revision_count", 0),
                        "revision_notes": current_state.get("revision_notes", []),
                        "errors": current_state.get("errors", [])
                    }

                    # Attach relevant agent result if available
                    for result_key in [
                        "destination_result",
                        "hotel_result",
                        "food_result",
                        "transport_result",
                        "weather_result",
                        "budget_result",
                        "itinerary_result"
                    ]:
                        if isinstance(node_update, dict) and result_key in node_update:
                            payload[result_key] = node_update[result_key]

                    yield format_sse("agent_update", payload)

            # Final formatted response
            final_response = format_graph_response(current_state)

            yield format_sse("final", final_response)

        except Exception as e:
            error_payload = {
                "success": False,
                "message": "Streaming failed while generating the travel plan.",
                "error": str(e)
            }
            yield format_sse("error", error_payload)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )