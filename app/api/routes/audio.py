"""Audio upload and processing endpoints."""

from typing import Annotated
from collections import defaultdict
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status

from app.api.dependencies import (
    get_user_context,
    get_supabase,
    get_diarization_pipeline,
    get_embedding_extractor,
    get_settings_dep,
)
from app.config import Settings
from app.models.schemas import ProcessAudioResponse, ProcessedSegment
from app.services.speaker_service import SpeakerService
from app.services.storage_service import StorageService
from app.core.audio_utils import load_audio_from_bytes
from app.core.voice_matcher import VoiceMatcher

router = APIRouter()


def compute_quality_score(duration: float, min_duration: float = 0.5) -> float:
    """
    Compute a quality score based on segment duration.
    
    Longer segments produce more reliable embeddings.
    - < 1s: lower quality (0.5-0.7)
    - 1-3s: medium quality (0.7-0.9)
    - > 3s: high quality (0.9-1.0)
    """
    if duration < min_duration:
        return 0.3
    elif duration < 1.0:
        return 0.5 + (duration - min_duration) * 0.4
    elif duration < 3.0:
        return 0.7 + (duration - 1.0) * 0.1
    else:
        return min(1.0, 0.9 + (duration - 3.0) * 0.02)


def consolidate_diarization_labels(
    label_embeddings: dict[str, list[np.ndarray]],
    merge_threshold: float = 0.70,
) -> dict[str, str]:
    """
    Consolidate diarization labels that likely belong to the same speaker.
    
    Diarization often assigns different labels to the same speaker,
    especially across different parts of the audio. This function
    identifies which labels should be merged.
    
    Args:
        label_embeddings: Dict mapping labels to their embeddings
        merge_threshold: Similarity threshold for merging
        
    Returns:
        Dict mapping original labels to canonical labels
    """
    labels = list(label_embeddings.keys())
    
    if len(labels) <= 1:
        return {label: label for label in labels}
    
    # Build a union-find structure for merging
    parent = {label: label for label in labels}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Compare all pairs of labels
    for i, label1 in enumerate(labels):
        for label2 in labels[i + 1:]:
            embs1 = label_embeddings[label1]
            embs2 = label_embeddings[label2]
            
            # Compute similarity between embedding sets
            similarities = []
            for e1 in embs1:
                for e2 in embs2:
                    sim = float(np.dot(e1, e2))
                    similarities.append(sim)
            
            if not similarities:
                continue
            
            # Use max and top-k average for decision
            max_sim = max(similarities)
            top_k = min(5, len(similarities))
            top_avg = sum(sorted(similarities, reverse=True)[:top_k]) / top_k
            
            # Combined score emphasizes best matches
            combined = max_sim * 0.6 + top_avg * 0.4
            
            if combined >= merge_threshold:
                union(label1, label2)
    
    # Build final mapping
    return {label: find(label) for label in labels}


@router.post("/process", response_model=ProcessAudioResponse)
async def process_audio(
    file: Annotated[UploadFile, File(description="Audio file to process")],
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
    diarization_pipeline=Depends(get_diarization_pipeline),
    embedding_extractor=Depends(get_embedding_extractor),
    settings: Settings = Depends(get_settings_dep),
):
    """
    Process an audio file for speaker identification.
    
    This endpoint:
    1. Uploads the audio to storage
    2. Performs speaker diarization (who spoke when)
    3. Extracts voice embeddings for each speaker segment
    4. Matches voices against known speakers in the database
    5. Creates new speaker profiles for unknown voices
    6. Returns the processed segments with speaker identifications
    """
    user = user_ctx["user"]
    user_id = user["id"]
    
    # Read audio file
    audio_bytes = await file.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file",
        )
    
    # Load and preprocess audio
    try:
        audio, sr = load_audio_from_bytes(audio_bytes, target_sr=settings.sample_rate)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load audio file: {str(e)}",
        )
    
    # Upload audio to storage
    storage = StorageService(supabase)
    audio_url = await storage.upload_audio(
        audio_bytes=audio_bytes,
        user_id=user_id,
        file_extension=file.filename.split(".")[-1] if file.filename else "wav",
    )
    
    # Create conversation record
    conversation = supabase.table("conversations").insert({
        "user_id": user_id,
        "audio_file_url": audio_url,
    }).execute()
    conversation_id = conversation.data[0]["id"]
    
    # Run speaker diarization
    diarization_result = diarization_pipeline.process(audio, sr)
    
    # Initialize voice matcher with both thresholds
    voice_matcher = VoiceMatcher(
        supabase=supabase,
        embedding_extractor=embedding_extractor,
        threshold=settings.voice_match_threshold,
        merge_threshold=settings.voice_merge_threshold,
    )
    
    # Initialize speaker service
    speaker_service = SpeakerService(supabase)
    
    # Group segments by diarization label (same label = same speaker within this audio)
    segments_by_label: dict[str, list] = defaultdict(list)
    for segment in diarization_result:
        segment_audio = audio[int(segment.start * sr):int(segment.end * sr)]
        
        # Skip very short segments
        if len(segment_audio) / sr < settings.min_segment_duration:
            continue
        
        segments_by_label[segment.label].append({
            "segment": segment,
            "audio": segment_audio,
        })
    
    # STEP 1: Extract multiple embeddings per label for robust matching
    label_embeddings: dict[str, list[np.ndarray]] = {}
    
    for label, segments in segments_by_label.items():
        # Sort by duration (longest first) and take up to 3 best segments
        sorted_segments = sorted(segments, key=lambda s: len(s["audio"]), reverse=True)
        best_segments = sorted_segments[:3]
        
        embeddings = []
        for seg_data in best_segments:
            duration = len(seg_data["audio"]) / sr
            # Only use segments >= 0.7s for initial matching (more reliable)
            if duration >= 0.7:
                emb = embedding_extractor.extract(seg_data["audio"], sr)
                embeddings.append(emb)
        
        # If no segments >= 0.7s, use the longest one anyway
        if not embeddings and sorted_segments:
            emb = embedding_extractor.extract(sorted_segments[0]["audio"], sr)
            embeddings.append(emb)
        
        label_embeddings[label] = embeddings
    
    # STEP 2: Consolidate labels that might be the same speaker
    # This fixes the issue of diarization splitting the same speaker
    label_mapping = consolidate_diarization_labels(
        label_embeddings, 
        merge_threshold=settings.voice_merge_threshold
    )
    
    # Group by canonical labels
    canonical_labels = set(label_mapping.values())
    canonical_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    
    for label, canonical in label_mapping.items():
        canonical_embeddings[canonical].extend(label_embeddings.get(label, []))
    
    # STEP 3: Match each canonical label against database
    label_to_speaker: dict[str, str] = {}  # Maps canonical labels to speaker IDs
    new_speaker_labels: set[str] = set()
    
    for canonical_label in canonical_labels:
        embeddings = canonical_embeddings[canonical_label]
        
        if not embeddings:
            continue
        
        # Use multi-embedding matching for better accuracy
        if len(embeddings) >= 2:
            match_result = await voice_matcher.match_multiple_embeddings(
                embeddings=embeddings,
                user_id=user_id,
            )
        else:
            # Single embedding - use standard match
            match_result = await voice_matcher.match(
                embedding=embeddings[0],
                user_id=user_id,
            )
        
        if match_result.is_match:
            label_to_speaker[canonical_label] = match_result.speaker_id
            await speaker_service.update_last_seen(match_result.speaker_id)
        else:
            # Create new speaker with the best embedding
            # Use the centroid of embeddings for the initial embedding
            if len(embeddings) > 1:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)  # Re-normalize
            else:
                centroid = embeddings[0]
            
            speaker = await speaker_service.create_speaker(
                user_id=user_id,
                name=None,
                embedding=centroid,
            )
            label_to_speaker[canonical_label] = speaker["id"]
            new_speaker_labels.add(canonical_label)
    
    # Now process all segments with their assigned speaker IDs
    processed_segments = []
    
    for original_label, segments in segments_by_label.items():
        # Map original label to canonical label, then to speaker ID
        canonical_label = label_mapping.get(original_label, original_label)
        speaker_id = label_to_speaker.get(canonical_label)
        
        if not speaker_id:
            continue  # Skip if no speaker assigned (shouldn't happen)
        
        is_new = canonical_label in new_speaker_labels
        
        for seg_data in segments:
            segment = seg_data["segment"]
            segment_audio = seg_data["audio"]
            duration = len(segment_audio) / sr
            
            # Extract embedding for this specific segment
            embedding = embedding_extractor.extract(segment_audio, sr)
            
            # Compute quality score based on duration
            quality_score = compute_quality_score(duration, settings.min_segment_duration)
            
            # Store embedding (adds more voice samples for this speaker)
            await voice_matcher.store_embedding(
                speaker_id=speaker_id,
                embedding=embedding,
                quality_score=quality_score,
            )
            
            # Create segment record
            segment_record = supabase.table("segments").insert({
                "conversation_id": conversation_id,
                "speaker_id": speaker_id,
                "start_ms": int(segment.start * 1000),
                "end_ms": int(segment.end * 1000),
                "confidence": quality_score,
            }).execute()
            
            # Get speaker info for response
            speaker_info = await speaker_service.get_speaker(speaker_id)
            
            processed_segments.append(ProcessedSegment(
                segment_id=segment_record.data[0]["id"],
                speaker_id=speaker_id,
                speaker_name=speaker_info.get("name"),
                is_new_speaker=is_new,
                start_ms=int(segment.start * 1000),
                end_ms=int(segment.end * 1000),
                confidence=quality_score,
            ))
    
    # Sort segments by start time
    processed_segments.sort(key=lambda s: s.start_ms)
    
    # Count unique speakers (canonical labels that got matched/created)
    unique_speakers = len(set(label_to_speaker.values()))
    
    return ProcessAudioResponse(
        conversation_id=conversation_id,
        audio_url=audio_url,
        segments=processed_segments,
        total_speakers=unique_speakers,
        new_speakers=len(new_speaker_labels),
    )
