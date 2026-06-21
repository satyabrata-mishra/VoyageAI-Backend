from pathlib import Path
import sys
from dotenv import load_dotenv


def get_current_dir() -> Path:
    return Path.cwd().resolve()


def get_project_root() -> Path:
    """
    Finds VoyageAI-Backend root from any folder:
    - root
    - notebooks
    - notebooks/agents
    - app/agents
    - app/rag
    """
    current_dir = get_current_dir()

    for path in [current_dir, *current_dir.parents]:
        if (path / "main.py").exists() and (path / "app").exists():
            return path

    raise FileNotFoundError(
        "Project root not found. Make sure you are inside VoyageAI-Backend."
    )


def add_project_root_to_sys_path() -> Path:
    """
    Useful for notebooks so imports like `from app.agents...` work properly.
    """
    project_root = get_project_root()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


def load_environment() -> None:
    project_root = get_project_root()
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def get_chroma_db_path(must_exist: bool = True) -> Path:
    project_root = get_project_root()
    chroma_dir = project_root / "chroma_db"

    if must_exist and not chroma_dir.exists():
        raise FileNotFoundError(
            f"Chroma DB folder not found at {chroma_dir}. "
            "Run 01_rag_ingestion.ipynb first."
        )

    return chroma_dir


def get_data_dir(must_exist: bool = False) -> Path:
    data_dir = get_project_root() / "data"

    if must_exist and not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found at {data_dir}")

    return data_dir


def get_travel_docs_dir(must_exist: bool = False) -> Path:
    travel_docs_dir = get_data_dir(must_exist=True) / "travel_docs"

    if must_exist and not travel_docs_dir.exists():
        raise FileNotFoundError(
            f"Travel docs folder not found at {travel_docs_dir}"
        )

    return travel_docs_dir