"""Conversation history endpoints."""

from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.dependencies import get_user_context, get_supabase
from app.models.schemas import (
    Conversation,
    ConversationList,
    ConversationDetail,
    SegmentDetail,
)

router = APIRouter()


@router.get("", response_model=ConversationList)
async def list_conversations(
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List all conversations for the current user."""
    user_id = user_ctx["user"]["id"]
    
    result = supabase.table("conversations").select(
        "*, segments(count)"
    ).eq("user_id", user_id).order(
        "started_at", desc=True
    ).range(offset, offset + limit - 1).execute()
    
    # Get total count
    count_result = supabase.table("conversations").select(
        "id", count="exact"
    ).eq("user_id", user_id).execute()
    
    conversations = [
        Conversation(
            id=c["id"],
            started_at=c["started_at"],
            ended_at=c["ended_at"],
            audio_file_url=c["audio_file_url"],
            segment_count=c["segments"][0]["count"] if c.get("segments") else 0,
        )
        for c in result.data
    ]
    
    return ConversationList(
        conversations=conversations,
        total=count_result.count,
        limit=limit,
        offset=offset,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: UUID,
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
):
    """Get a specific conversation with all segments and speaker info."""
    user_id = user_ctx["user"]["id"]
    
    # Get conversation
    conv_result = supabase.table("conversations").select("*").eq(
        "id", str(conversation_id)
    ).eq("user_id", user_id).execute()
    
    if not conv_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    
    conversation = conv_result.data[0]
    
    # Get segments with speaker info
    segments_result = supabase.table("segments").select(
        "*, speakers(id, name, is_identified)"
    ).eq("conversation_id", str(conversation_id)).order("start_ms").execute()
    
    segments = [
        SegmentDetail(
            id=s["id"],
            speaker_id=s["speaker_id"],
            speaker_name=s["speakers"]["name"] if s.get("speakers") else None,
            speaker_is_identified=s["speakers"]["is_identified"] if s.get("speakers") else False,
            start_ms=s["start_ms"],
            end_ms=s["end_ms"],
            transcript=s.get("transcript"),
            confidence=s["confidence"],
        )
        for s in segments_result.data
    ]
    
    # Count unique speakers
    unique_speakers = set(s.speaker_id for s in segments)
    
    return ConversationDetail(
        id=conversation["id"],
        started_at=conversation["started_at"],
        ended_at=conversation["ended_at"],
        audio_file_url=conversation["audio_file_url"],
        segments=segments,
        total_speakers=len(unique_speakers),
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
):
    """Delete a conversation and all associated segments."""
    user_id = user_ctx["user"]["id"]
    
    # Verify conversation belongs to user
    existing = supabase.table("conversations").select("id").eq(
        "id", str(conversation_id)
    ).eq("user_id", user_id).execute()
    
    if not existing.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    
    # Delete segments first (foreign key constraint)
    supabase.table("segments").delete().eq(
        "conversation_id", str(conversation_id)
    ).execute()
    
    # Delete conversation
    supabase.table("conversations").delete().eq(
        "id", str(conversation_id)
    ).execute()
