"""Audio preprocessing utilities."""

import io
import tempfile
import os
import subprocess
import numpy as np
import librosa
import soundfile as sf


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio from bytes and resample to target sample rate.
    Supports WAV, MP3, WebM/Opus, and other formats via ffmpeg.
    
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
    
    return audio.astype(np.float32), sr


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
