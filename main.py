from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.health import router as health_router
from app.api.travel import router as travel_router


app = FastAPI(
    title="VoyageAI Backend",
    description="Agentic AI travel planner using RAG, LangGraph, and FastAPI.",
    version="1.0.0"
)

_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_extra_origins = os.getenv("CORS_ORIGINS", "")
_cors_origins = _default_origins + [o.strip() for o in _extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(travel_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to VoyageAI Backend",
        "docs": "/docs",
        "health": "/api/health"
    }