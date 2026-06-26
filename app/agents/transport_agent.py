import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config.settings import settings
from app.utils.json_utils import parse_json_response

load_dotenv()


# =====================================================
# LLM Prompt
# =====================================================

TRANSPORT_AGENT_PROMPT = ChatPromptTemplate.from_template("""
You are the Transport Agent of VoyageAI.

Your task is to suggest transport options for the selected destination.

User Travel Request:
{user_query}

Destination Agent Output:
{destination_result}

Rules:
- Use general knowledge only
- Do not use live APIs or schedules

Return ONLY valid JSON.

JSON structure:

{{
  "agent_name": "Transport Agent",
  "status": "success",
  "source_city": "unknown",
  "destination": "destination name",
  "travel_style_detected": "budget/comfort/luxury/fastest/unknown",
  "recommended_intercity_mode": "flight/train/bus/car/ferry/mixed/unknown",
  "intercity_options": [
    {{
      "mode": "flight/train/bus/car/ferry/mixed",
      "suitability": "high/medium/low",
      "best_for": "who this mode is best for",
      "cost_level": "low/medium/high/unknown",
      "time_level": "slow/medium/fast/unknown",
      "note": "short explanation"
    }}
  ],
  "local_transport_options": [
    "cab services",
    "public transport",
    "rental bikes"
  ],
  "transport_tips": [
    "tip 1",
    "tip 2",
    "tip 3"
  ],
  "limitations": "No real-time data used",
  "confidence": "high"
}}
""")


# =====================================================
# Transport Agent Class
# =====================================================

class TransportAgent:
    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment")

        self.llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            temperature=0.2,
            groq_api_key=settings.GROQ_API_KEY,
        )

        self.chain = TRANSPORT_AGENT_PROMPT | self.llm | StrOutputParser()

    # -------------------------------------------------
    # Main Execution
    # -------------------------------------------------

    def run(
        self,
        user_query: str,
        destination_result: dict[str, Any],
    ) -> dict[str, Any]:

        try:
            destination_status = destination_result.get("status")
            destination = destination_result.get("recommended_destination")

            # -------------------------------------------------
            # Guard: Destination failure
            # -------------------------------------------------
            if destination_status != "success":
                return {
                    "agent_name": "Transport Agent",
                    "status": "skipped",
                    "destination": None,
                    "message": "Destination Agent did not return a valid destination.",
                    "confidence": "low",
                }

            if not destination:
                return {
                    "agent_name": "Transport Agent",
                    "status": "no_destination_found",
                    "destination": None,
                    "message": "No destination provided.",
                    "confidence": "low",
                }

            # -------------------------------------------------
            # LLM Invocation (NO RAG)
            # -------------------------------------------------
            response = self.chain.invoke(
                {
                    "user_query": user_query,
                    "destination_result": json.dumps(
                        destination_result,
                        indent=2,
                        ensure_ascii=False,
                    ),
                }
            )

            # -------------------------------------------------
            # Parse JSON safely
            # -------------------------------------------------
            parsed_response = parse_json_response(response)

            parsed_response["status"] = parsed_response.get("status", "success")

            return parsed_response

        except Exception as e:
            return {
                "agent_name": "Transport Agent",
                "status": "failed",
                "error": str(e),
            }


# =====================================================
# Singleton Pattern
# =====================================================

_transport_agent_instance = None


def get_transport_agent() -> TransportAgent:
    global _transport_agent_instance

    if _transport_agent_instance is None:
        _transport_agent_instance = TransportAgent()

    return _transport_agent_instance


def run_transport_agent(
    user_query: str,
    destination_result: dict[str, Any],
) -> dict[str, Any]:
    agent = get_transport_agent()
    return agent.run(user_query, destination_result)


# =====================================================
# Local Testing
# =====================================================

if __name__ == "__main__":

    test_input = {
        "agent_name": "Destination Agent",
        "status": "success",
        "recommended_destination": "Goa",
        "reason": "Beaches, nightlife, and food culture",
        "confidence": "high",
    }

    result = run_transport_agent(
        user_query="I want a low budget Goa trip from Bhubaneswar",
        destination_result=test_input,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))