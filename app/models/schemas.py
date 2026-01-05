"""Pydantic schemas for API request and response models."""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


# ============================================================================
# Speaker Schemas
# ============================================================================

class Speaker(BaseModel):
    """Speaker representation."""
    id: UUID
    name: Optional[str] = None
    is_identified: bool = False
    first_seen: datetime
    last_seen: datetime
    embedding_count: int = 0


class SpeakerList(BaseModel):
    """Paginated list of speakers."""
    speakers: list[Speaker]
    total: int
    limit: int
    offset: int


class SpeakerUpdate(BaseModel):
    """Request to update a speaker."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)


class SpeakerMergeRequest(BaseModel):
    """Request to merge two speakers."""
    source_speaker_id: UUID = Field(
        ..., 
        description="ID of the speaker to merge INTO the target speaker"
    )


class SpeakerMergeResponse(BaseModel):
    """Response after merging speakers."""
    merged_speaker_id: str
    deleted_speaker_id: str
    segments_moved: int
    embeddings_moved: int


class MergeSuggestion(BaseModel):
    """A suggestion to merge two speakers that might be the same person."""
    speaker1_id: UUID
    speaker2_id: UUID
    speaker1_name: Optional[str] = None
    speaker2_name: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0)


class MergeSuggestionList(BaseModel):
    """List of merge suggestions."""
    suggestions: list[MergeSuggestion]
    threshold_used: float


# ============================================================================
# Audio Processing Schemas
# ============================================================================

class ProcessedSegment(BaseModel):
    """A processed audio segment with speaker identification."""
    segment_id: UUID
    speaker_id: UUID
    speaker_name: Optional[str] = None
    is_new_speaker: bool
    start_ms: int
    end_ms: int
    confidence: float = Field(..., ge=0.0, le=1.0)


class ProcessAudioResponse(BaseModel):
    """Response from audio processing."""
    conversation_id: UUID
    audio_url: str
    segments: list[ProcessedSegment]
    total_speakers: int
    new_speakers: int


# ============================================================================
# Conversation Schemas
# ============================================================================

class Conversation(BaseModel):
    """Conversation summary."""
    id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None
    audio_file_url: Optional[str] = None
    segment_count: int = 0


class ConversationList(BaseModel):
    """Paginated list of conversations."""
    conversations: list[Conversation]
    total: int
    limit: int
    offset: int


class SegmentDetail(BaseModel):
    """Detailed segment information including speaker."""
    id: UUID
    speaker_id: Optional[UUID] = None
    speaker_name: Optional[str] = None
    speaker_is_identified: bool = False
    start_ms: int
    end_ms: int
    transcript: Optional[str] = None
    confidence: float


class ConversationDetail(BaseModel):
    """Full conversation with all segments."""
    id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None
    audio_file_url: Optional[str] = None
    segments: list[SegmentDetail]
    total_speakers: int


# ============================================================================
# Streaming Schemas
# ============================================================================

class StreamingIdentification(BaseModel):
    """Real-time speaker identification result."""
    speaker_id: UUID
    speaker_name: Optional[str] = None
    is_new_speaker: bool
    start_ms: int
    end_ms: int
    confidence: float
    transcript: Optional[str] = None


class StreamingResponse(BaseModel):
    """WebSocket response message."""
    type: str  # "identification", "conversation_started", "conversation_ended"
    conversation_id: Optional[UUID] = None
    segments: Optional[list[StreamingIdentification]] = None


# ============================================================================
# Voice Matching Schemas (Internal)
# ============================================================================

class VoiceMatchResult(BaseModel):
    """Result of voice matching operation."""
    is_match: bool
    speaker_id: Optional[str] = None
    confidence: float = 0.0
    
    class Config:
        """Allow arbitrary types for numpy arrays in related operations."""
        arbitrary_types_allowed = True


# ============================================================================
# Diarization Schemas (Internal)
# ============================================================================

class DiarizedSegment(BaseModel):
    """A segment from speaker diarization."""
    start: float  # seconds
    end: float  # seconds
    label: str  # speaker label from diarization (e.g., "SPEAKER_00")
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start
