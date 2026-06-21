from typing import Optional
from pydantic import BaseModel, Field


class TravelPlanRequest(BaseModel):
    user_query: str = Field(
        ...,
        min_length=10,
        description="Natural language travel request from the user."
    )


class TravelPlanResponse(BaseModel):
    success: bool
    message: str
    user_query: Optional[str] = None
    destination: Optional[str] = None
    budget_status: Optional[str] = None
    progress_percentage: int = 0
    agent_status: dict = {}
    revision_count: int = 0
    revision_notes: list = []
    agent_outputs: dict = {}
    final_itinerary: dict = {}
    errors: list = []