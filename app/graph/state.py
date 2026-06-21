from typing import TypedDict, Optional, Dict, Any, List


class VoyageAIState(TypedDict, total=False):
    user_query: str

    destination_result: Optional[Dict[str, Any]]
    hotel_result: Optional[Dict[str, Any]]
    food_result: Optional[Dict[str, Any]]
    transport_result: Optional[Dict[str, Any]]
    weather_result: Optional[Dict[str, Any]]
    budget_result: Optional[Dict[str, Any]]
    itinerary_result: Optional[Dict[str, Any]]

    budget_status: Optional[str]
    revision_count: int
    revision_notes: List[str]

    agent_status: Dict[str, str]
    current_agent: Optional[str]
    progress_percentage: int

    errors: List[str]