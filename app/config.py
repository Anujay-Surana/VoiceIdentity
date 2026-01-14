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
    use_adaptive_threshold: bool = False  # Adjust threshold based on audio quality
    use_centroid_matching: bool = False   # Match against speaker centroids (experimental)
    
    # Audio Processing
    sample_rate: int = 16000
    min_segment_duration: float = 0.5  # Minimum segment to process
    min_match_duration: float = 0.8  # Do DB matching for segments >= 0.8s (lowered for better UX)
    
    # Audio Preprocessing
    enable_preprocessing: bool = False   # Enable noise reduction, normalization (experimental)
    enable_vad: bool = False             # Enable voice activity detection (experimental)
    noise_reduce_strength: float = 0.75  # Noise reduction strength (0-1)
    highpass_cutoff: float = 80.0        # High-pass filter cutoff (Hz)
    target_loudness: float = -23.0       # Target loudness in LUFS (EBU R128)
    
    # Quality Thresholds
    min_quality_score: float = 0.3       # Minimum quality to process segment
    store_quality_threshold: float = 0.6  # Minimum quality to store embedding
    min_speech_ratio: float = 0.3        # Minimum speech ratio (from VAD)
    
    # Transcription
    enable_transcription: bool = True
    whisper_model_size: str = "base"  # tiny, base, small, medium, large-v2
    transcription_language: str = "en"  # Language code or empty for auto-detect
    whisper_no_speech_threshold: float = 0.6  # Higher = stricter speech detection
    whisper_condition_on_previous: bool = False  # Prevent error propagation
    
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
