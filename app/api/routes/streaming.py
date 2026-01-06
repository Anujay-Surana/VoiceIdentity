"""WebSocket streaming endpoint for real-time voice identification."""

import json
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
import numpy as np

from app.api.dependencies import get_supabase, get_settings_dep
from app.config import Settings, get_settings
from app.services.supabase_client import get_supabase_client
from app.core.diarization import DiarizationPipeline
from app.core.embeddings import EmbeddingExtractor
from app.core.voice_matcher import VoiceMatcher
from app.core.audio_utils import load_audio_from_bytes
from app.services.speaker_service import SpeakerService
from app.core.transcription import TranscriptionService

router = APIRouter()
logger = logging.getLogger(__name__)


def compute_quality_score(duration: float, min_duration: float = 0.5) -> float:
    """Compute quality score based on segment duration."""
    if duration < min_duration:
        return 0.3
    elif duration < 1.0:
        return 0.5 + (duration - min_duration) * 0.4
    elif duration < 3.0:
        return 0.7 + (duration - 1.0) * 0.1
    else:
        return min(1.0, 0.9 + (duration - 3.0) * 0.02)


class StreamingSession:
    """Manages a real-time voice identification streaming session."""
    
    def __init__(
        self,
        user_id: str,
        supabase,
        diarization_pipeline: DiarizationPipeline,
        embedding_extractor: EmbeddingExtractor,
        settings: Settings,
        transcription_service: Optional[TranscriptionService] = None,
    ):
        self.user_id = user_id
        self.supabase = supabase
        self.diarization_pipeline = diarization_pipeline
        self.embedding_extractor = embedding_extractor
        self.settings = settings
        self.transcription_service = transcription_service
        
        # IMPORTANT: Use low thresholds for real-world streaming audio
        # Config may be cached, so we override here to ensure it works
        effective_threshold = min(settings.voice_match_threshold, 0.45)
        effective_merge_threshold = min(settings.voice_merge_threshold, 0.55)
        
        self.voice_matcher = VoiceMatcher(
            supabase=supabase,
            embedding_extractor=embedding_extractor,
            threshold=effective_threshold,
            merge_threshold=effective_merge_threshold,
        )
        logger.info(f"[SESSION] Using thresholds: match={effective_threshold}, merge={effective_merge_threshold}")
        logger.info(f"[SESSION] Transcription enabled: {transcription_service is not None}")
        self.speaker_service = SpeakerService(supabase)
        
        # Audio buffer for accumulating chunks
        self.audio_buffer: list[np.ndarray] = []
        self.buffer_duration = 0.0
        
        # Processing window settings
        self.window_duration = 5.0  # seconds
        self.overlap_duration = 2.5  # 50% overlap
        
        # Track current conversation
        self.conversation_id: Optional[str] = None
        
        # Track speaker assignments within session for continuity
        # Maps diarization label -> speaker_id
        self.session_speaker_map: dict[str, str] = {}
        
        # Store embeddings per session label for cross-label verification
        # Maps diarization label -> list of embeddings
        self.session_label_embeddings: dict[str, list[np.ndarray]] = {}
        
        # Threshold for merging labels within session
        self.label_merge_threshold = settings.voice_merge_threshold
    
    async def start_conversation(self) -> str:
        """Create a new conversation record."""
        result = self.supabase.table("conversations").insert({
            "user_id": self.user_id,
        }).execute()
        self.conversation_id = result.data[0]["id"]
        
        # Debug: Check how many speakers and embeddings exist for this user
        try:
            speakers = self.supabase.table("speakers").select("id").eq("user_id", self.user_id).execute()
            speaker_count = len(speakers.data) if speakers.data else 0
            
            if speaker_count > 0:
                speaker_ids = [s["id"] for s in speakers.data]
                embeddings = self.supabase.table("voice_embeddings").select(
                    "id", count="exact"
                ).in_("speaker_id", speaker_ids).execute()
                embedding_count = embeddings.count or 0
            else:
                embedding_count = 0
            
            logger.info(f"[SESSION] Started conversation. User {self.user_id[:8]}... has {speaker_count} speakers and {embedding_count} embeddings")
        except Exception as e:
            logger.error(f"[SESSION] Error checking user stats: {e}")
        
        return self.conversation_id
    
    async def process_chunk(self, audio_bytes: bytes) -> list[dict]:
        """
        Process an incoming audio chunk.
        
        Returns list of identified speaker segments.
        """
        logger.info(f"[CHUNK] Processing {len(audio_bytes)} bytes...")
        
        # Load and add to buffer
        try:
            audio, sr = load_audio_from_bytes(audio_bytes, target_sr=self.settings.sample_rate)
            logger.info(f"[CHUNK] Loaded audio: {len(audio)} samples, {len(audio)/sr:.2f}s, sr={sr}")
        except Exception as e:
            logger.warning(f"[CHUNK] Could not load audio: {e}")
            return []  # Skip this chunk gracefully
        
        # Skip if audio is too short or empty
        if len(audio) < 100:
            logger.warning(f"[CHUNK] Audio too short: {len(audio)} samples")
            return []
            
        self.audio_buffer.append(audio)
        self.buffer_duration += len(audio) / sr
        
        results = []
        
        # Process if we have enough audio
        while self.buffer_duration >= self.window_duration:
            # Concatenate buffer
            full_audio = np.concatenate(self.audio_buffer)
            
            # Extract window
            window_samples = int(self.window_duration * sr)
            window_audio = full_audio[:window_samples]
            
            # Process window
            window_results = await self._process_window(window_audio, sr)
            results.extend(window_results)
            
            # Slide buffer (keep overlap)
            overlap_samples = int(self.overlap_duration * sr)
            keep_samples = len(full_audio) - window_samples + overlap_samples
            
            if keep_samples > 0:
                self.audio_buffer = [full_audio[-keep_samples:]]
                self.buffer_duration = keep_samples / sr
            else:
                self.audio_buffer = []
                self.buffer_duration = 0.0
        
        return results
    
    def _find_matching_session_label(self, embedding: np.ndarray) -> Optional[str]:
        """
        Check if this embedding matches any existing session label.
        
        This helps merge different diarization labels that are actually
        the same speaker within a session.
        
        Only returns labels that are mapped to real speakers (not temp_* IDs).
        """
        best_label = None
        best_similarity = 0.0
        
        for label, embeddings in self.session_label_embeddings.items():
            if not embeddings:
                continue
            
            # Skip labels mapped to temporary speakers
            speaker_id = self.session_speaker_map.get(label)
            if speaker_id and isinstance(speaker_id, str) and speaker_id.startswith("temp_"):
                continue
            
            # Compute similarity against stored embeddings for this label
            similarities = [float(np.dot(embedding, e)) for e in embeddings[-10:]]  # Last 10 for efficiency
            
            # Use max and average for decision
            max_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            combined = max_sim * 0.6 + avg_sim * 0.4
            
            if combined > best_similarity and combined >= self.label_merge_threshold:
                best_similarity = combined
                best_label = label
        
        return best_label
    
    async def _process_window(self, audio: np.ndarray, sr: int) -> list[dict]:
        """Process a single audio window for speaker identification."""
        results = []
        
        try:
            # Run diarization
            diarization_result = self.diarization_pipeline.process(audio, sr)
        except Exception as e:
            print(f"Diarization error: {e}")
            return []
        
        for segment in diarization_result:
            try:
                duration = segment.end - segment.start
                
                # Skip very short segments (unreliable)
                if duration < self.settings.min_segment_duration:
                    logger.debug(f"Skipping short segment ({duration:.2f}s < {self.settings.min_segment_duration}s)")
                    continue
                
                # Extract segment audio
                start_sample = int(segment.start * sr)
                end_sample = int(segment.end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Log audio stats for debugging
                audio_max = np.abs(segment_audio).max()
                audio_rms = np.sqrt(np.mean(segment_audio**2))
                logger.debug(f"Segment audio: duration={duration:.2f}s, max={audio_max:.4f}, rms={audio_rms:.4f}")
                
                # Skip silent or very quiet segments
                if audio_rms < 0.01:
                    logger.debug(f"Skipping silent segment (rms={audio_rms:.4f})")
                    continue
                
                # Get embedding - pass sample rate for proper resampling
                embedding = self.embedding_extractor.extract(segment_audio, sr)
                
                # Compute quality score
                quality_score = compute_quality_score(duration, self.settings.min_segment_duration)
                
                # Use diarization label
                diar_label = segment.label
                is_new = False
                
                # STEP 1: Check if this exact diarization label is already known
                if diar_label in self.session_speaker_map:
                    cached_speaker_id = self.session_speaker_map[diar_label]
                    is_cached_temp = isinstance(cached_speaker_id, str) and cached_speaker_id.startswith("temp_")
                    
                    # If cached speaker is temporary and we now have a longer segment,
                    # try to upgrade it to a real speaker from DB
                    min_match_duration = getattr(self.settings, 'min_match_duration', 1.5)
                    if is_cached_temp and duration >= min_match_duration:
                        logger.info(f"[STREAM] Upgrading temp speaker for label '{diar_label}' with longer segment ({duration:.2f}s)")
                        # Remove the temp mapping so we go through full detection
                        del self.session_speaker_map[diar_label]
                        # Fall through to re-detection below
                    else:
                        # CRITICAL FIX: Verify the embedding actually matches the cached speaker!
                        # Diarization labels reset between windows, so SPEAKER_00 in one window
                        # might be a completely different person than SPEAKER_00 in another window.
                        cached_embeddings = self.session_label_embeddings.get(diar_label, [])
                        
                        if cached_embeddings:
                            # Check similarity against cached embeddings for this label
                            similarities = [float(np.dot(embedding, e)) for e in cached_embeddings[-5:]]  # Last 5
                            avg_sim = sum(similarities) / len(similarities)
                            max_sim = max(similarities)
                            
                            # If embedding doesn't match cached speaker, treat as new speaker detection
                            if max_sim < 0.50 or avg_sim < 0.40:
                                logger.info(f"[STREAM] Label '{diar_label}' embedding mismatch! max_sim={max_sim:.3f}, avg_sim={avg_sim:.3f}")
                                logger.info(f"[STREAM] Speaker likely changed - re-evaluating...")
                                # Don't use cached mapping - fall through to re-detection
                                del self.session_speaker_map[diar_label]
                                # Keep embeddings but they'll be replaced if different speaker
                            else:
                                # Embedding matches - use cached speaker
                                speaker_id = cached_speaker_id
                                self.session_label_embeddings[diar_label].append(embedding)
                                # Continue to result generation
                                logger.debug(f"[STREAM] Label '{diar_label}' verified (sim={max_sim:.3f}), using cached speaker")
                        else:
                            # No cached embeddings to compare, use cached speaker
                            speaker_id = cached_speaker_id
                            self.session_label_embeddings[diar_label] = [embedding]
                
                # If we don't have a valid cached mapping, go through detection
                if diar_label not in self.session_speaker_map:
                    # STEP 2: Check if embedding matches another session label
                    # This handles cases where diarization assigns different labels
                    # to the same speaker
                    matching_label = self._find_matching_session_label(embedding)
                    
                    if matching_label:
                        # Found a match - use that speaker
                        speaker_id = self.session_speaker_map[matching_label]
                        # Also map this new label to the same speaker
                        self.session_speaker_map[diar_label] = speaker_id
                        # Store embedding under the original matching label
                        self.session_label_embeddings[matching_label].append(embedding)
                    else:
                        # STEP 3: Try to match against known speakers in DB
                        # Only do this for longer segments (more reliable embeddings)
                        min_match_duration = getattr(self.settings, 'min_match_duration', 1.5)
                        
                        if duration < min_match_duration:
                            # Short segment - don't try DB match, just track locally
                            logger.info(f"[STREAM] Short segment ({duration:.2f}s) - skipping DB match, tracking locally")
                            # Create a temporary local speaker for this session
                            # It will be matched/merged with DB on next longer segment
                            self.session_speaker_map[diar_label] = f"temp_{diar_label}"
                            self.session_label_embeddings[diar_label] = [embedding]
                            continue  # Don't create a result for short segments
                        
                        try:
                            logger.info(f"[STREAM] New label '{diar_label}' ({duration:.2f}s) - checking DB for matches...")
                            
                            match_result = await self.voice_matcher.match(
                                embedding=embedding,
                                user_id=self.user_id,
                            )
                            
                            logger.info(f"[STREAM] Match result: is_match={match_result.is_match}, confidence={match_result.confidence:.4f}")
                            
                            if match_result.is_match:
                                speaker_id = match_result.speaker_id
                                is_new = False
                                logger.info(f"[STREAM] Matched to existing speaker {speaker_id[:8]}...")
                            else:
                                # Create new speaker
                                logger.info(f"[STREAM] No match found (confidence={match_result.confidence:.4f}) - creating new speaker")
                                speaker = await self.speaker_service.create_speaker(
                                    user_id=self.user_id,
                                    name=None,
                                    embedding=embedding,
                                )
                                speaker_id = speaker["id"]
                                is_new = True
                                logger.info(f"[STREAM] Created new speaker {speaker_id[:8]}...")
                            
                            # Remember this speaker for session continuity
                            self.session_speaker_map[diar_label] = speaker_id
                            # Initialize embeddings list for this label
                            self.session_label_embeddings[diar_label] = [embedding]
                        except Exception as e:
                            logger.error(f"[STREAM] DB match error: {e}")
                            continue
                
                # Skip database operations for temporary speakers
                is_temp_speaker = isinstance(speaker_id, str) and speaker_id.startswith("temp_")
                
                # Transcribe the segment
                transcript = None
                if self.transcription_service and duration >= 0.5:  # Only transcribe segments >= 0.5s
                    try:
                        transcript = self.transcription_service.transcribe_simple(
                            segment_audio,
                            sample_rate=sr,
                            language=self.settings.transcription_language or None,
                        )
                        if transcript:
                            logger.info(f"[TRANSCRIBE] '{transcript[:50]}...' ({duration:.1f}s)")
                    except Exception as e:
                        logger.error(f"[TRANSCRIBE] Error: {e}")
                
                if not is_temp_speaker:
                    # Store embedding with quality score (non-blocking, skip on error)
                    try:
                        await self.voice_matcher.store_embedding(
                            speaker_id=speaker_id,
                            embedding=embedding,
                            quality_score=quality_score,
                        )
                    except Exception as e:
                        print(f"Store embedding error: {e}")
                    
                    # Create segment record with transcript (non-blocking, skip on error)
                    if self.conversation_id:
                        try:
                            self.supabase.table("segments").insert({
                                "conversation_id": self.conversation_id,
                                "speaker_id": speaker_id,
                                "start_ms": int(segment.start * 1000),
                                "end_ms": int(segment.end * 1000),
                                "confidence": quality_score,
                                "transcript": transcript,
                            }).execute()
                        except Exception as e:
                            print(f"Store segment error: {e}")
                
                # Skip result for temporary speakers (not yet matched to DB)
                if is_temp_speaker:
                    continue
                
                # Get speaker name from cache or DB
                speaker_name = None
                try:
                    speaker_info = await self.speaker_service.get_speaker(speaker_id)
                    speaker_name = speaker_info.get("name") if speaker_info else None
                except Exception:
                    pass  # Use None for name
                
                results.append({
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "is_new_speaker": is_new,
                    "start_ms": int(segment.start * 1000),
                    "end_ms": int(segment.end * 1000),
                    "confidence": quality_score,
                    "transcript": transcript,
                })
            except Exception as e:
                print(f"Segment processing error: {e}")
                continue
        
        return results
    
    async def flush(self) -> list[dict]:
        """Process any remaining audio in the buffer."""
        if not self.audio_buffer or self.buffer_duration < self.settings.min_segment_duration:
            return []
        
        full_audio = np.concatenate(self.audio_buffer)
        sr = self.settings.sample_rate
        
        results = await self._process_window(full_audio, sr)
        
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        return results
    
    async def end_conversation(self):
        """Mark the conversation as ended."""
        if self.conversation_id:
            self.supabase.table("conversations").update({
                "ended_at": "now()",
            }).eq("id", self.conversation_id).execute()


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    x_api_key: str = Query(...),
    x_user_id: str = Query(...),
):
    """
    WebSocket endpoint for real-time voice identification.
    
    Connect with query params:
    - x_api_key: Your API key
    - x_user_id: External user ID
    
    Send audio chunks as binary messages.
    Receive JSON messages with speaker identifications.
    
    Control messages (JSON):
    - {"action": "start"} - Start a new conversation
    - {"action": "end"} - End current conversation and flush buffer
    """
    settings = get_settings()
    supabase = get_supabase_client(settings.supabase_url, settings.supabase_service_key)
    
    # Validate API key
    org_result = supabase.table("organizations").select("*").eq("api_key", x_api_key).execute()
    if not org_result.data:
        await websocket.close(code=4001, reason="Invalid API key")
        return
    
    org = org_result.data[0]
    
    # Get or create user
    user_result = supabase.table("users").select("*").eq(
        "org_id", org["id"]
    ).eq("external_user_id", x_user_id).execute()
    
    if user_result.data:
        user_id = user_result.data[0]["id"]
    else:
        new_user = supabase.table("users").insert({
            "org_id": org["id"],
            "external_user_id": x_user_id,
        }).execute()
        user_id = new_user.data[0]["id"]
    
    await websocket.accept()
    
    # Get shared ML pipelines (lazy loaded on first use)
    from app.main import get_ml_models
    diarization_pipeline, embedding_extractor = get_ml_models()
    
    # Initialize transcription service if enabled
    transcription_service = None
    if settings.enable_transcription:
        try:
            transcription_service = TranscriptionService(
                model_size=settings.whisper_model_size,
            )
        except Exception as e:
            logger.error(f"Failed to initialize transcription: {e}")
    
    # Create session
    session = StreamingSession(
        user_id=user_id,
        supabase=supabase,
        diarization_pipeline=diarization_pipeline,
        embedding_extractor=embedding_extractor,
        settings=settings,
        transcription_service=transcription_service,
    )
    
    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
            
            if "text" in message:
                # Control message
                data = json.loads(message["text"])
                action = data.get("action")
                
                if action == "start":
                    conversation_id = await session.start_conversation()
                    await websocket.send_json({
                        "type": "conversation_started",
                        "conversation_id": conversation_id,
                    })
                
                elif action == "end":
                    # Flush remaining audio
                    final_results = await session.flush()
                    if final_results:
                        await websocket.send_json({
                            "type": "identification",
                            "segments": final_results,
                        })
                    
                    await session.end_conversation()
                    await websocket.send_json({
                        "type": "conversation_ended",
                        "conversation_id": session.conversation_id,
                    })
            
            elif "bytes" in message:
                # Audio chunk
                audio_bytes = message["bytes"]
                logger.info(f"[STREAM] Received audio chunk: {len(audio_bytes)} bytes")
                
                try:
                    results = await session.process_chunk(audio_bytes)
                    logger.info(f"[STREAM] Processed chunk, got {len(results)} results")
                    
                    if results:
                        await websocket.send_json({
                            "type": "identification",
                            "segments": results,
                        })
                except Exception as e:
                    logger.error(f"[STREAM] Error processing chunk: {e}")
    
    except WebSocketDisconnect:
        # Clean up on disconnect
        await session.flush()
        await session.end_conversation()
