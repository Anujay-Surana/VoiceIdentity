"""Application configuration using Pydantic settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Supabase
    supabase_url: str
    supabase_key: str
    supabase_service_key: str
    
    # HuggingFace (for PyAnnote models)
    hf_token: str
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Voice Matching
    # ECAPA-TDNN typical same-speaker similarity: 0.4-0.85 depending on conditions
    # Real-world streaming audio often has lower scores due to noise, short segments
    voice_match_threshold: float = 0.40  # Low threshold - your scores are ~0.48
    voice_merge_threshold: float = 0.55  # Threshold for auto-merging speakers  
    embedding_dimension: int = 192
    
    # Audio Processing
    sample_rate: int = 16000
    min_segment_duration: float = 0.5  # Minimum segment to process
    min_match_duration: float = 0.8  # Do DB matching for segments >= 0.8s (lowered for better UX)
    
    # Transcription
    enable_transcription: bool = True
    whisper_model_size: str = "base"  # tiny, base, small, medium, large-v2
    transcription_language: str = "en"  # Language code or empty for auto-detect
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars not in schema


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    # Log the actual threshold values being used
    import logging
    logging.getLogger(__name__).info(
        f"[CONFIG] Loaded settings: voice_match_threshold={settings.voice_match_threshold}, "
        f"voice_merge_threshold={settings.voice_merge_threshold}"
    )
    return settings


def clear_settings_cache():
    """Clear the settings cache to reload from environment."""
    get_settings.cache_clear()
