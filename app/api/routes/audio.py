"""Audio upload and processing endpoints."""

from typing import Annotated, Optional
from collections import defaultdict
import numpy as np
import io
import base64
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Query
from fastapi.responses import Response

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
from app.core.audio_utils import load_audio_from_bytes, load_and_preprocess
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


@router.post("/test-preprocessing")
async def test_preprocessing(
    file: Annotated[UploadFile, File(description="Audio file to test preprocessing")],
    enable_preprocessing: bool = Query(True, description="Enable noise reduction and normalization"),
    enable_vad: bool = Query(False, description="Enable voice activity detection"),
    settings: Settings = Depends(get_settings_dep),
):
    """
    Test audio preprocessing pipeline.
    
    Returns detailed metrics about the audio before and after preprocessing,
    along with the processed audio as base64 for playback.
    
    This endpoint is for testing/debugging the audio pipeline.
    """
    import soundfile as sf
    
    audio_bytes = await file.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file",
        )
    
    try:
        # Load original audio
        original_audio, original_sr = load_audio_from_bytes(audio_bytes, target_sr=settings.sample_rate)
        
        # Compute original metrics
        original_rms = float(np.sqrt(np.mean(original_audio ** 2)))
        original_peak = float(np.abs(original_audio).max())
        original_duration = len(original_audio) / original_sr
        
        # Compute original spectral characteristics
        original_silent_ratio = float(np.mean(np.abs(original_audio) < 0.01))
        
        result = {
            "original": {
                "duration_seconds": round(original_duration, 3),
                "sample_rate": original_sr,
                "samples": len(original_audio),
                "rms_level": round(original_rms, 4),
                "peak_level": round(original_peak, 4),
                "silent_ratio": round(original_silent_ratio, 3),
                "clipping_detected": original_peak >= 0.99,
            },
            "processed": None,
            "processed_audio_base64": None,
            "original_audio_base64": None,
        }
        
        # Encode original audio for playback
        original_buffer = io.BytesIO()
        sf.write(original_buffer, original_audio, original_sr, format='WAV')
        original_buffer.seek(0)
        result["original_audio_base64"] = base64.b64encode(original_buffer.read()).decode('utf-8')
        
        if enable_preprocessing:
            try:
                # Run preprocessing pipeline
                processed = load_and_preprocess(
                    audio_bytes,
                    target_sr=settings.sample_rate,
                    enable_preprocessing=True,
                    enable_vad=enable_vad,
                    min_quality_score=0.0,  # Don't reject anything for testing
                )
                
                processed_audio = processed.audio
                processed_rms = float(np.sqrt(np.mean(processed_audio ** 2)))
                processed_peak = float(np.abs(processed_audio).max())
                processed_silent_ratio = float(np.mean(np.abs(processed_audio) < 0.01))
                
                result["processed"] = {
                    "duration_seconds": round(processed.duration, 3),
                    "sample_rate": processed.sample_rate,
                    "samples": len(processed_audio),
                    "rms_level": round(processed_rms, 4),
                    "peak_level": round(processed_peak, 4),
                    "silent_ratio": round(processed_silent_ratio, 3),
                    "quality_score": round(processed.quality_score, 3),
                    "speech_ratio": round(processed.speech_ratio, 3),
                    "is_acceptable": processed.is_acceptable,
                    "clipping_detected": processed_peak >= 0.99,
                }
                
                # Compute improvement metrics
                result["improvements"] = {
                    "rms_change": round((processed_rms - original_rms) / max(original_rms, 0.001), 3),
                    "noise_reduction": round(original_silent_ratio - processed_silent_ratio, 3) if enable_vad else None,
                    "normalized": processed_peak > original_peak * 1.1 or processed_peak < original_peak * 0.9,
                }
                
                # Encode processed audio for playback
                processed_buffer = io.BytesIO()
                sf.write(processed_buffer, processed_audio, processed.sample_rate, format='WAV')
                processed_buffer.seek(0)
                result["processed_audio_base64"] = base64.b64encode(processed_buffer.read()).decode('utf-8')
                
            except Exception as e:
                result["preprocessing_error"] = str(e)
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process audio: {str(e)}",
        )


@router.post("/test-embedding")
async def test_embedding(
    file: Annotated[UploadFile, File(description="Audio file to extract embedding from")],
    user_ctx: Annotated[dict, Depends(get_user_context)],
    supabase=Depends(get_supabase),
    embedding_extractor=Depends(get_embedding_extractor),
    settings: Settings = Depends(get_settings_dep),
):
    """
    Test voice embedding extraction and matching.
    
    Returns the embedding and top matches from the database without creating
    any new speakers or storing the embedding.
    """
    user = user_ctx["user"]
    user_id = user["id"]
    
    audio_bytes = await file.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file",
        )
    
    try:
        # Load and optionally preprocess
        try:
            processed = load_and_preprocess(
                audio_bytes,
                target_sr=settings.sample_rate,
                enable_preprocessing=True,
                enable_vad=False,
                min_quality_score=0.0,
            )
            audio = processed.audio
            sr = processed.sample_rate
            quality_score = processed.quality_score
        except Exception:
            audio, sr = load_audio_from_bytes(audio_bytes, target_sr=settings.sample_rate)
            quality_score = 0.5
        
        # Extract embedding
        embedding = embedding_extractor.extract(audio, sr)
        
        # Initialize voice matcher
        voice_matcher = VoiceMatcher(
            supabase=supabase,
            embedding_extractor=embedding_extractor,
            threshold=settings.voice_match_threshold,
            merge_threshold=settings.voice_merge_threshold,
        )
        
        # Get match candidates
        match_result, candidates = await voice_matcher.match_with_candidates(
            embedding=embedding,
            user_id=user_id,
        )
        
        # Get speaker names for candidates
        candidate_info = []
        for candidate in candidates[:10]:
            try:
                speaker_info = supabase.table("speakers").select("id, name").eq(
                    "id", candidate["speaker_id"]
                ).execute()
                name = speaker_info.data[0].get("name") if speaker_info.data else None
            except Exception:
                name = None
            
            candidate_info.append({
                "speaker_id": candidate["speaker_id"],
                "speaker_name": name,
                "similarity": round(candidate["similarity"], 4),
            })
        
        return {
            "duration_seconds": round(len(audio) / sr, 3),
            "quality_score": round(quality_score, 3),
            "embedding_dimensions": len(embedding),
            "match_result": {
                "is_match": match_result.is_match,
                "speaker_id": match_result.speaker_id,
                "confidence": round(match_result.confidence, 4) if match_result.confidence else None,
            },
            "top_candidates": candidate_info,
            "threshold_used": settings.voice_match_threshold,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process audio: {str(e)}",
        )
