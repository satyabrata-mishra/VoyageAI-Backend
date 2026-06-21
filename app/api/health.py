from fastapi import APIRouter

router = APIRouter(
    prefix="/api",
    tags=["Health"]
)


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Hello from VoyageAI Backend. Everything is running smoothly!"
    }