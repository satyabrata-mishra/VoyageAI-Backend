from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.path_utils import get_project_root

import os

class Settings(BaseSettings):
    APP_NAME: str = "VoyageAI"
    APP_VERSION: str = "0.1.0"
    
    GROQ_MODEL_NAME: str = "llama-3.1-8b-instant"
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_COLLECTION_NAME: str = "voyageai_travel_knowledge"

    model_config = SettingsConfigDict(
        env_file=str(get_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()