"""FastAPI dependencies for authentication and database access."""

from typing import Annotated
from fastapi import Depends, Header, HTTPException, Request, status

from app.config import get_settings, Settings
from app.services.supabase_client import get_supabase_client
from app.core.diarization import DiarizationPipeline
from app.core.embeddings import EmbeddingExtractor


def get_settings_dep() -> Settings:
    """Dependency to get application settings."""
    return get_settings()


async def get_supabase(settings: Annotated[Settings, Depends(get_settings_dep)]):
    """Dependency to get Supabase client."""
    return get_supabase_client(settings.supabase_url, settings.supabase_service_key)


async def get_api_key(
    x_api_key: Annotated[str, Header()],
    supabase=Depends(get_supabase),
) -> dict:
    """Validate API key and return organization info."""
    result = supabase.table("organizations").select("*").eq("api_key", x_api_key).execute()
    
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return result.data[0]


async def get_user_context(
    x_user_id: Annotated[str, Header()],
    org: Annotated[dict, Depends(get_api_key)],
    supabase=Depends(get_supabase),
) -> dict:
    """Get or create user context from external user ID."""
    # Check if user exists
    result = supabase.table("users").select("*").eq(
        "org_id", org["id"]
    ).eq(
        "external_user_id", x_user_id
    ).execute()
    
    if result.data:
        return {"user": result.data[0], "org": org}
    
    # Create new user
    new_user = supabase.table("users").insert({
        "org_id": org["id"],
        "external_user_id": x_user_id,
    }).execute()
    
    return {"user": new_user.data[0], "org": org}


def get_diarization_pipeline(request: Request) -> DiarizationPipeline:
    """Get the diarization pipeline (lazy loaded)."""
    from app.main import get_ml_models
    diarization, _ = get_ml_models()
    return diarization


def get_embedding_extractor(request: Request) -> EmbeddingExtractor:
    """Get the embedding extractor (lazy loaded)."""
    from app.main import get_ml_models
    _, embeddings = get_ml_models()
    return embeddings
