"""Audio preprocessing utilities."""

import io
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio from bytes and resample to target sample rate.
    Supports WAV, MP3, WebM/Opus, and other formats.
    
    Returns:
        tuple: (audio_array, sample_rate)
    """
    # Try soundfile first (fast, supports WAV, FLAC, OGG)
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        audio = audio.astype(np.float32)
    except Exception:
        # Fall back to torchaudio for WebM/Opus and other formats
        try:
            audio, sr = _load_with_torchaudio(audio_bytes)
        except Exception as e:
            # Last resort: try with pydub if ffmpeg is available
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
                audio = audio / (2**15)  # Normalize to [-1, 1]
                sr = audio_segment.frame_rate
                
                if audio_segment.channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
            except Exception:
                raise ValueError(f"Could not load audio: {e}")
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio.astype(np.float32), sr


def _load_with_torchaudio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load audio using torchaudio via a temporary file.
    This handles WebM/Opus and other formats that need file access.
    """
    # Write to temp file (torchaudio needs file path for some formats)
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        # Load with torchaudio
        waveform, sr = torchaudio.load(temp_path)
        
        # Convert to numpy
        audio = waveform.numpy()
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0)
        else:
            audio = audio.squeeze(0)
        
        return audio.astype(np.float32), sr
    finally:
        # Clean up temp file
        os.unlink(temp_path)


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
