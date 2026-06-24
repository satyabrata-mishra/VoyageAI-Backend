import os
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.tools.destination_tool import (
    search_destination,
    get_top_attractions
)

# Existing RAG utilities
from app.rag.retriever import get_retrieval_context
from app.rag.rag_chain import rag_chain

load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)


# =====================================================
# Extract destination if explicitly mentioned
# =====================================================

def extract_destination_from_query(user_query: str):
    """
    Returns:
    Destination name
    OR
    NONE
    """

    prompt = f"""
You are a destination extraction system.

User Query:
{user_query}

Rules:
- If user explicitly mentions a city, state, country or destination,
  return only that destination name.
- If no destination is mentioned,
  return NONE.

Examples:

I want to visit Venice
Venice

Plan a Goa trip
Goa

I want beaches and seafood
NONE

Return only the answer.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# =====================================================
# Enrich destination using OpenTripMap
# =====================================================

def enrich_destination(destination_name):

    destination_data = search_destination(
        destination_name
    )

    if not destination_data["success"]:
        return None

    attractions = get_top_attractions(
        destination_data["latitude"],
        destination_data["longitude"]
    )

    return {
        "destination_data": destination_data,
        "attractions": attractions
    }


# =====================================================
# Generate destination analysis
# =====================================================

def generate_destination_analysis(
    user_query,
    destination_data,
    attractions
):

    prompt = f"""
You are an expert travel planner.

User Query:
{user_query}

Destination Information:
{json.dumps(destination_data, indent=2)}

Top Attractions:
{json.dumps(attractions[:10], indent=2)}

Generate travel insights.

Return valid JSON with:

{{
    "reason": "...",
    "destination_type": [...],
    "suitable_for": [...],
    "suggested_duration": "...",
    "best_time_to_visit": "..."
}}
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    if "```json" in content:
        content = content.replace("```json", "")
        content = content.replace("```", "").strip()

    return json.loads(content)


# =====================================================
# Main Agent
# =====================================================

def run_destination_agent(user_query):

    try:

        # -------------------------------------------------
        # STEP 1
        # Check explicit destination
        # -------------------------------------------------

        explicit_destination = extract_destination_from_query(
            user_query
        )

        # -------------------------------------------------
        # CASE A
        # User explicitly provided destination
        # -------------------------------------------------

        if explicit_destination.upper() != "NONE":

            enriched = enrich_destination(
                explicit_destination
            )

            if not enriched:

                return {
                    "agent_name": "Destination Agent",
                    "status": "failed",
                    "error": f"Could not find destination: {explicit_destination}"
                }

            destination_data = enriched["destination_data"]
            attractions = enriched["attractions"]

            analysis = generate_destination_analysis(
                user_query=user_query,
                destination_data=destination_data,
                attractions=attractions
            )

            return {
                "agent_name": "Destination Agent",
                "status": "success",

                "recommended_destination":
                    destination_data["name"],

                "country":
                    destination_data["country"],

                "coordinates": {
                    "latitude":
                        destination_data["latitude"],
                    "longitude":
                        destination_data["longitude"]
                },

                "key_attractions": [
                    attraction["name"]
                    for attraction in attractions[:10]
                ],

                **analysis,

                "confidence": "high"
            }

        # -------------------------------------------------
        # CASE B
        # No destination mentioned
        # Use existing RAG workflow
        # -------------------------------------------------

        context = get_retrieval_context(
            query=user_query,
            k=3
        )

        rag_response = rag_chain.invoke(
            {
                "user_query": user_query,
                "context": context
            }
        )

        # -------------------------------------------------
        # Extract destination from RAG output
        # -------------------------------------------------

        extraction_prompt = f"""
Extract ONLY the recommended destination.

Response:

{rag_response}

Return only destination name.
"""

        destination_name = (
            llm.invoke(extraction_prompt)
            .content
            .strip()
        )

        enriched = enrich_destination(
            destination_name
        )

        if not enriched:

            return {
                "agent_name": "Destination Agent",
                "status": "failed",
                "error": f"Could not enrich destination: {destination_name}"
            }

        destination_data = enriched["destination_data"]
        attractions = enriched["attractions"]

        analysis = generate_destination_analysis(
            user_query=user_query,
            destination_data=destination_data,
            attractions=attractions
        )

        return {
            "agent_name": "Destination Agent",
            "status": "success",

            "recommended_destination":
                destination_data["name"],

            "country":
                destination_data["country"],

            "coordinates": {
                "latitude":
                    destination_data["latitude"],
                "longitude":
                    destination_data["longitude"]
            },

            "key_attractions": [
                attraction["name"]
                for attraction in attractions[:10]
            ],

            **analysis,

            "confidence": "high"
        }

    except Exception as e:

        return {
            "agent_name": "Destination Agent",
            "status": "failed",
            "error": str(e)
        }


# =====================================================
# Local Testing
# =====================================================

if __name__ == "__main__":

    print("\n===== TEST 1 =====\n")

    result = run_destination_agent(
        "I want to visit Venice"
    )

    print(json.dumps(result, indent=2))

    print("\n===== TEST 2 =====\n")

    result = run_destination_agent(
        "I want beaches, seafood and nightlife"
    )

    print(json.dumps(result, indent=2))