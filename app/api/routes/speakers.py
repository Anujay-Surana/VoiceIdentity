"""Speaker management endpoints."""

from typing import Annotated, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.dependencies import get_user_context, get_supabase, get_settings_dep
from app.config import Settings
from app.models.schemas import (
    Speaker,
    SpeakerList,
    SpeakerUpdate,
    SpeakerMergeRequest,
    SpeakerMergeResponse,
    MergeSuggestion,
    MergeSuggestionList,
)
from app.services.speaker_service import SpeakerService

router = APIRouter()


@router.get("", response_model=SpeakerList)
async def list_speakers(
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
    identified_only: bool = Query(False, description="Only return identified (named) speakers"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List all known speakers for the current user."""
    user_id = user_ctx["user"]["id"]
    
    query = supabase.table("speakers").select(
        "*, voice_embeddings(count)"
    ).eq("user_id", user_id)
    
    if identified_only:
        query = query.eq("is_identified", True)
    
    result = query.order(
        "last_seen", desc=True
    ).range(offset, offset + limit - 1).execute()
    
    # Get total count
    count_result = supabase.table("speakers").select(
        "id", count="exact"
    ).eq("user_id", user_id).execute()
    
    speakers = [
        Speaker(
            id=s["id"],
            name=s["name"],
            is_identified=s["is_identified"],
            first_seen=s["first_seen"],
            last_seen=s["last_seen"],
            embedding_count=s["voice_embeddings"][0]["count"] if s.get("voice_embeddings") else 0,
        )
        for s in result.data
    ]
    
    return SpeakerList(
        speakers=speakers,
        total=count_result.count,
        limit=limit,
        offset=offset,
    )


@router.get("/{speaker_id}", response_model=Speaker)
async def get_speaker(
    speaker_id: UUID,
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
):
    """Get a specific speaker by ID."""
    user_id = user_ctx["user"]["id"]
    
    result = supabase.table("speakers").select(
        "*, voice_embeddings(count)"
    ).eq("id", str(speaker_id)).eq("user_id", user_id).execute()
    
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Speaker not found",
        )
    
    s = result.data[0]
    return Speaker(
        id=s["id"],
        name=s["name"],
        is_identified=s["is_identified"],
        first_seen=s["first_seen"],
        last_seen=s["last_seen"],
        embedding_count=s["voice_embeddings"][0]["count"] if s.get("voice_embeddings") else 0,
    )


@router.patch("/{speaker_id}", response_model=Speaker)
async def update_speaker(
    speaker_id: UUID,
    update: SpeakerUpdate,
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
):
    """Update a speaker's information (e.g., assign a name)."""
    user_id = user_ctx["user"]["id"]
    
    # Verify speaker belongs to user
    existing = supabase.table("speakers").select("id").eq(
        "id", str(speaker_id)
    ).eq("user_id", user_id).execute()
    
    if not existing.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Speaker not found",
        )
    
    # Build update data
    update_data = {}
    if update.name is not None:
        update_data["name"] = update.name
        update_data["is_identified"] = bool(update.name)
    
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )
    
    result = supabase.table("speakers").update(update_data).eq(
        "id", str(speaker_id)
    ).execute()
    
    s = result.data[0]
    return Speaker(
        id=s["id"],
        name=s["name"],
        is_identified=s["is_identified"],
        first_seen=s["first_seen"],
        last_seen=s["last_seen"],
    )


@router.post("/{speaker_id}/merge", response_model=SpeakerMergeResponse)
async def merge_speakers(
    speaker_id: UUID,
    merge_request: SpeakerMergeRequest,
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
):
    """
    Merge another speaker into this one (when they're the same person).
    
    The source speaker will be deleted and all their data moved to the target.
    """
    user_id = user_ctx["user"]["id"]
    target_id = str(speaker_id)
    source_id = str(merge_request.source_speaker_id)
    
    if target_id == source_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot merge speaker with itself",
        )
    
    speaker_service = SpeakerService(supabase)
    
    # Verify both speakers belong to user
    target = await speaker_service.get_speaker(target_id)
    source = await speaker_service.get_speaker(source_id)
    
    if not target or target.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target speaker not found",
        )
    
    if not source or source.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source speaker not found",
        )
    
    # Merge speakers
    merged = await speaker_service.merge_speakers(target_id, source_id)
    
    return SpeakerMergeResponse(
        merged_speaker_id=target_id,
        deleted_speaker_id=source_id,
        segments_moved=merged["segments_moved"],
        embeddings_moved=merged["embeddings_moved"],
    )


@router.get("/debug/stats")
async def get_debug_stats(
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
):
    """
    Get debug statistics about speakers and embeddings for the current user.
    
    Useful for diagnosing why matching isn't working.
    """
    user_id = user_ctx["user"]["id"]
    
    # Get all speakers for user
    speakers = supabase.table("speakers").select("id, name, is_identified, first_seen, last_seen").eq(
        "user_id", user_id
    ).execute()
    
    speaker_stats = []
    total_embeddings = 0
    
    for speaker in (speakers.data or []):
        # Count embeddings for this speaker
        emb_count = supabase.table("voice_embeddings").select(
            "id", count="exact"
        ).eq("speaker_id", speaker["id"]).execute()
        
        count = emb_count.count or 0
        total_embeddings += count
        
        speaker_stats.append({
            "id": speaker["id"],
            "name": speaker.get("name"),
            "is_identified": speaker.get("is_identified"),
            "embedding_count": count,
            "first_seen": speaker.get("first_seen"),
            "last_seen": speaker.get("last_seen"),
        })
    
    return {
        "user_id": user_id,
        "total_speakers": len(speaker_stats),
        "total_embeddings": total_embeddings,
        "speakers": speaker_stats,
        "warning": "Speakers with 0 embeddings cannot be matched!" if any(s["embedding_count"] == 0 for s in speaker_stats) else None,
    }


@router.get("/suggestions/merge", response_model=MergeSuggestionList)
async def get_merge_suggestions(
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
    settings: Settings = Depends(get_settings_dep),
    threshold: float = Query(None, ge=0.5, le=1.0, description="Similarity threshold for suggestions"),
):
    """
    Get suggestions for speakers that might be the same person.
    
    Returns pairs of speakers with high voice similarity that could be merged.
    This is useful for cleaning up duplicate speakers after processing.
    """
    user_id = user_ctx["user"]["id"]
    
    # Use provided threshold or default from settings
    effective_threshold = threshold if threshold is not None else settings.voice_merge_threshold
    
    try:
        # Call the database function to find merge candidates
        result = supabase.rpc(
            "find_merge_candidates",
            {
                "target_user_id": user_id,
                "similarity_threshold": effective_threshold,
            }
        ).execute()
        
        suggestions = [
            MergeSuggestion(
                speaker1_id=row["speaker1_id"],
                speaker2_id=row["speaker2_id"],
                speaker1_name=row.get("speaker1_name"),
                speaker2_name=row.get("speaker2_name"),
                similarity=row["similarity"],
            )
            for row in (result.data or [])
        ]
        
        return MergeSuggestionList(
            suggestions=suggestions,
            threshold_used=effective_threshold,
        )
    except Exception as e:
        # If the function doesn't exist yet, return empty list
        print(f"Merge suggestions query failed: {e}")
        return MergeSuggestionList(
            suggestions=[],
            threshold_used=effective_threshold,
        )


@router.delete("/{speaker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_speaker(
    speaker_id: UUID,
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
):
    """Delete a speaker and all associated data."""
    user_id = user_ctx["user"]["id"]
    
    # Verify speaker belongs to user
    existing = supabase.table("speakers").select("id").eq(
        "id", str(speaker_id)
    ).eq("user_id", user_id).execute()
    
    if not existing.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Speaker not found",
        )
    
    speaker_service = SpeakerService(supabase)
    await speaker_service.delete_speaker(str(speaker_id))


@router.get("/{speaker_id}/transcripts")
async def get_speaker_transcripts(
    speaker_id: UUID,
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Get all transcripts/segments for a specific speaker.
    
    Returns segments with transcripts ordered by most recent first.
    """
    user_id = user_ctx["user"]["id"]
    
    # Verify speaker belongs to user
    existing = supabase.table("speakers").select("id, name").eq(
        "id", str(speaker_id)
    ).eq("user_id", user_id).execute()
    
    if not existing.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Speaker not found",
        )
    
    speaker = existing.data[0]
    
    # Get segments with transcripts for this speaker
    # Join with conversations to get the timestamp
    result = supabase.table("segments").select(
        "id, conversation_id, start_ms, end_ms, confidence, transcript, conversations(started_at)"
    ).eq(
        "speaker_id", str(speaker_id)
    ).not_.is_(
        "transcript", "null"  # Only get segments that have transcripts
    ).order(
        "start_ms", desc=True
    ).range(offset, offset + limit - 1).execute()
    
    # Get total count
    count_result = supabase.table("segments").select(
        "id", count="exact"
    ).eq(
        "speaker_id", str(speaker_id)
    ).not_.is_("transcript", "null").execute()
    
    return {
        "speaker_id": str(speaker_id),
        "speaker_name": speaker.get("name"),
        "transcripts": [
            {
                "id": seg["id"],
                "conversation_id": seg["conversation_id"],
                "transcript": seg["transcript"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "duration_ms": seg["end_ms"] - seg["start_ms"],
                "confidence": seg["confidence"],
                "created_at": seg.get("conversations", {}).get("started_at") if seg.get("conversations") else None,
            }
            for seg in (result.data or [])
        ],
        "total": count_result.count or 0,
        "limit": limit,
        "offset": offset,
    }
