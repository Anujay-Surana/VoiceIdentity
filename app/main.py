"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import audio, speakers, conversations, streaming
from app.core.diarization import DiarizationPipeline
from app.core.embeddings import EmbeddingExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML models on startup."""
    settings = get_settings()
    
    # Initialize ML pipelines (loaded once, reused for all requests)
    app.state.diarization_pipeline = DiarizationPipeline(hf_token=settings.hf_token)
    app.state.embedding_extractor = EmbeddingExtractor()
    
    yield
    
    # Cleanup on shutdown
    del app.state.diarization_pipeline
    del app.state.embedding_extractor


app = FastAPI(
    title="Voice Identity Platform",
    description="Speaker identification and diarization service",
    version="1.0.0",
    lifespan=lifespan,
)

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
    return {"status": "healthy"}


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
