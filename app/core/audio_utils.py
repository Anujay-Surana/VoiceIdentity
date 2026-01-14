"""Audio preprocessing utilities with quality scoring.

This module provides:
- Audio loading from various formats
- Audio preprocessing (noise reduction, normalization)
- Quality scoring for embedding selection
- Voice activity detection integration
"""

import io
import tempfile
import os
import subprocess
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedAudio:
    """Container for processed audio with metadata."""
    audio: np.ndarray          # Preprocessed audio samples
    sample_rate: int           # Sample rate
    duration: float            # Duration in seconds
    quality_score: float       # Quality score (0-1)
    speech_ratio: float        # Ratio of speech to total duration
    snr_db: float              # Estimated SNR in dB
    is_acceptable: bool        # Whether quality is good enough to process
    
    @property
    def is_high_quality(self) -> bool:
        """Check if quality is high enough to store embedding."""
        return self.quality_score >= 0.6


def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = 16000,
    apply_preprocessing: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Load audio from bytes and resample to target sample rate.
    Supports WAV, MP3, WebM/Opus, and other formats via ffmpeg.
    
    Args:
        audio_bytes: Raw audio bytes.
        target_sr: Target sample rate.
        apply_preprocessing: Whether to apply noise reduction/normalization.
    
    Returns:
        tuple: (audio_array, sample_rate)
    """
    # Try soundfile first (fast, supports WAV, FLAC, OGG)
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        audio = audio.astype(np.float32)
    except Exception:
        # Fall back to ffmpeg for WebM/Opus and other formats
        try:
            audio, sr = _load_with_ffmpeg(audio_bytes, target_sr)
        except Exception as e:
            raise ValueError(f"Could not load audio: {e}")
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Apply preprocessing if requested
    if apply_preprocessing:
        from app.core.audio_preprocessing import get_preprocessor
        preprocessor = get_preprocessor()
        audio = preprocessor.preprocess(audio, sr)
    
    return audio.astype(np.float32), sr


def load_and_preprocess(
    audio_bytes: bytes,
    target_sr: int = 16000,
    enable_preprocessing: bool = True,
    enable_vad: bool = True,
    min_quality_score: float = 0.3,
) -> ProcessedAudio:
    """
    Load audio with full preprocessing and quality analysis.
    
    This is the recommended method for processing audio for speaker
    identification, as it applies all quality improvements and provides
    detailed quality metrics.
    
    Args:
        audio_bytes: Raw audio bytes.
        target_sr: Target sample rate.
        enable_preprocessing: Whether to apply noise reduction/normalization.
        enable_vad: Whether to compute speech ratio using VAD.
        min_quality_score: Minimum quality score to accept.
        
    Returns:
        ProcessedAudio object with preprocessed audio and quality metrics.
    """
    # Load raw audio
    try:
        audio, sr = load_audio_from_bytes(audio_bytes, target_sr, apply_preprocessing=False)
    except Exception as e:
        logger.error(f"[AUDIO] Failed to load audio: {e}")
        return ProcessedAudio(
            audio=np.array([], dtype=np.float32),
            sample_rate=target_sr,
            duration=0.0,
            quality_score=0.0,
            speech_ratio=0.0,
            snr_db=0.0,
            is_acceptable=False,
        )
    
    duration = len(audio) / sr
    
    # Apply preprocessing
    if enable_preprocessing:
        from app.core.audio_preprocessing import get_preprocessor
        preprocessor = get_preprocessor()
        audio = preprocessor.preprocess(audio, sr)
    
    # Compute speech ratio using VAD
    speech_ratio = 1.0
    if enable_vad:
        try:
            from app.core.vad import get_vad
            vad = get_vad()
            speech_ratio = vad.get_speech_ratio(audio, sr)
        except Exception as e:
            logger.warning(f"[AUDIO] VAD failed: {e}")
            speech_ratio = 1.0  # Assume all speech
    
    # Compute quality score
    from app.core.audio_preprocessing import compute_quality_score
    quality = compute_quality_score(audio, sr, speech_ratio)
    
    is_acceptable = quality.overall_score >= min_quality_score
    
    logger.debug(f"[AUDIO] Processed: duration={duration:.2f}s, quality={quality.overall_score:.2f}, "
                f"speech_ratio={speech_ratio:.2f}, snr={quality.snr_db:.1f}dB, acceptable={is_acceptable}")
    
    return ProcessedAudio(
        audio=audio,
        sample_rate=sr,
        duration=duration,
        quality_score=quality.overall_score,
        speech_ratio=speech_ratio,
        snr_db=quality.snr_db,
        is_acceptable=is_acceptable,
    )


def _load_with_ffmpeg(audio_bytes: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio using ffmpeg subprocess.
    This handles WebM/Opus, MP4, and other formats reliably.
    """
    # Write input to temp file
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f_in:
        f_in.write(audio_bytes)
        input_path = f_in.name
    
    # Output to temp WAV file
    output_path = input_path.replace('.webm', '.wav')
    
    try:
        # Use ffmpeg to convert to WAV
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', input_path,
            '-ar', str(target_sr),  # Resample to target
            '-ac', '1',  # Mono
            '-f', 'wav',
            output_path
        ], capture_output=True, timeout=30)
        
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            raise ValueError(f"ffmpeg error: {stderr[:200]}")
        
        # Read the converted WAV
        audio, sr = sf.read(output_path)
        return audio.astype(np.float32), sr
        
    finally:
        # Clean up temp files
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def extract_segment(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """Extract a segment from audio array given start and end times in seconds."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return audio[start_sample:end_sample]


def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return buffer.read()


def get_audio_duration(audio: np.ndarray, sr: int) -> float:
    """Get duration of audio in seconds."""
    return len(audio) / sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio
