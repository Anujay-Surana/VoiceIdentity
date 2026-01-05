"""Speaker management service."""

import logging
from typing import Optional
from supabase import Client
import numpy as np

logger = logging.getLogger(__name__)


class SpeakerService:
    """Service for managing speaker records and operations."""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
    
    async def create_speaker(
        self,
        user_id: str,
        name: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Create a new speaker record.
        
        If embedding is provided, it will also create the first voice embedding.
        """
        speaker_data = {
            "user_id": user_id,
            "name": name,
            "is_identified": bool(name),
        }
        
        result = self.supabase.table("speakers").insert(speaker_data).execute()
        speaker = result.data[0]
        logger.info(f"[SPEAKER] Created new speaker {speaker['id'][:8]}... for user {user_id[:8]}...")
        
        # Store initial embedding if provided - THIS IS CRITICAL for matching to work
        if embedding is not None:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    embed_result = self.supabase.table("voice_embeddings").insert({
                        "speaker_id": speaker["id"],
                        "embedding": embedding.tolist(),
                        "quality_score": 1.0,
                    }).execute()
                    logger.info(f"[SPEAKER] Stored initial embedding for speaker {speaker['id'][:8]}...")
                    break
                except Exception as e:
                    logger.error(f"[SPEAKER] Failed to store initial embedding (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        # Last attempt failed - log critical error but continue
                        # The speaker exists but has no embeddings - this is bad!
                        logger.critical(f"[SPEAKER] CRITICAL: Speaker {speaker['id'][:8]}... created WITHOUT embedding!")
        
        return speaker
    
    async def get_speaker(self, speaker_id: str) -> Optional[dict]:
        """Get a speaker by ID."""
        result = self.supabase.table("speakers").select("*").eq(
            "id", speaker_id
        ).execute()
        
        return result.data[0] if result.data else None
    
    async def update_speaker(self, speaker_id: str, **updates) -> dict:
        """Update a speaker's fields."""
        result = self.supabase.table("speakers").update(updates).eq(
            "id", speaker_id
        ).execute()
        
        return result.data[0]
    
    async def update_last_seen(self, speaker_id: str) -> None:
        """Update a speaker's last_seen timestamp."""
        self.supabase.table("speakers").update({
            "last_seen": "now()",
        }).eq("id", speaker_id).execute()
    
    async def delete_speaker(self, speaker_id: str) -> None:
        """Delete a speaker and all associated data."""
        # Delete embeddings first (foreign key)
        self.supabase.table("voice_embeddings").delete().eq(
            "speaker_id", speaker_id
        ).execute()
        
        # Update segments to remove speaker reference (or could delete)
        self.supabase.table("segments").update({
            "speaker_id": None,
        }).eq("speaker_id", speaker_id).execute()
        
        # Delete speaker
        self.supabase.table("speakers").delete().eq(
            "id", speaker_id
        ).execute()
    
    async def merge_speakers(self, target_id: str, source_id: str) -> dict:
        """
        Merge source speaker into target speaker.
        
        - Moves all embeddings from source to target
        - Updates all segments from source to target
        - Deletes the source speaker
        - Updates target's first_seen if source is older
        
        Returns stats about the merge operation.
        """
        # Get both speakers to compare dates
        target = await self.get_speaker(target_id)
        source = await self.get_speaker(source_id)
        
        # Move embeddings
        embed_result = self.supabase.table("voice_embeddings").update({
            "speaker_id": target_id,
        }).eq("speaker_id", source_id).execute()
        embeddings_moved = len(embed_result.data)
        
        # Move segments
        seg_result = self.supabase.table("segments").update({
            "speaker_id": target_id,
        }).eq("speaker_id", source_id).execute()
        segments_moved = len(seg_result.data)
        
        # Update first_seen if source is older
        if source["first_seen"] < target["first_seen"]:
            self.supabase.table("speakers").update({
                "first_seen": source["first_seen"],
            }).eq("id", target_id).execute()
        
        # Update last_seen if source is newer
        if source["last_seen"] > target["last_seen"]:
            self.supabase.table("speakers").update({
                "last_seen": source["last_seen"],
            }).eq("id", target_id).execute()
        
        # Delete source speaker
        self.supabase.table("speakers").delete().eq(
            "id", source_id
        ).execute()
        
        return {
            "embeddings_moved": embeddings_moved,
            "segments_moved": segments_moved,
        }
    
    async def get_speakers_for_user(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get all speakers for a user."""
        result = self.supabase.table("speakers").select("*").eq(
            "user_id", user_id
        ).order("last_seen", desc=True).range(
            offset, offset + limit - 1
        ).execute()
        
        return result.data
