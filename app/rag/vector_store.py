from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.config.settings import settings
from app.utils.path_utils import get_chroma_db_path


@lru_cache(maxsize=1)
def get_embedding_model():
    print("Loading embedding model...")

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store():
    print("Loading ChromaDB vector store...")

    chroma_dir = get_chroma_db_path(must_exist=True)

    return Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=get_embedding_model(),
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )