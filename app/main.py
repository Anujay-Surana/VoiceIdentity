"""FastAPI application entry point."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import audio, speakers, conversations, streaming

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Identity Platform",
    description="Speaker identification and diarization service",
    version="1.0.0",
)

# ML models loaded lazily on first request
app.state.diarization_pipeline = None
app.state.embedding_extractor = None
app.state.models_loaded = False


def get_ml_models():
    """Lazy load ML models on first use."""
    if not app.state.models_loaded:
        logger.info("Loading ML models (first request)...")
        from app.core.diarization import DiarizationPipeline
        from app.core.embeddings import EmbeddingExtractor
        settings = get_settings()
        
        app.state.diarization_pipeline = DiarizationPipeline(hf_token=settings.hf_token)
        app.state.embedding_extractor = EmbeddingExtractor()
        app.state.models_loaded = True
        logger.info("ML models loaded successfully!")
    
    return app.state.diarization_pipeline, app.state.embedding_extractor

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(speakers.router, prefix="/api/v1/speakers", tags=["speakers"])
app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["conversations"])
app.include_router(streaming.router, prefix="/api/v1", tags=["streaming"])


@app.get("/")
async def root():
    """Root endpoint - helps with ngrok browser warning."""
    return {"message": "Voice Identity API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": app.state.models_loaded,
    }


if __name__ == "__main__":
    import os
    import uvicorn
    settings = get_settings()
    # Railway sets PORT env var
    port = int(os.environ.get("PORT", settings.api_port))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
    )
