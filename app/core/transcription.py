"""Speech-to-text transcription using OpenAI Whisper."""

import numpy as np
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# Target sample rate for Whisper (16kHz)
WHISPER_SAMPLE_RATE = 16000


class TranscriptionService:
    """
    Transcribe audio segments using OpenAI's Whisper.
    
    Whisper is a robust speech recognition model that works well
    across many languages and audio conditions.
    """
    
    def __init__(
        self,
        model_size: str = "base",  # Options: tiny, base, small, medium, large
        device: str = "auto",  # auto, cpu, cuda
    ):
        """
        Initialize the transcription service.
        
        Args:
            model_size: Whisper model size (base is good balance of speed/accuracy)
            device: Device to run on (auto-detects GPU if available)
        """
        import whisper
        
        logger.info(f"[TRANSCRIBE] Loading Whisper model: {model_size}")
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
        logger.info(f"[TRANSCRIBE] Model loaded on {device}")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,  # None = auto-detect
    ) -> dict:
        """
        Transcribe an audio segment.
        
        Args:
            audio: Audio waveform (mono, float32)
            sample_rate: Sample rate of input audio
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            
        Returns:
            Dict with transcript text and metadata
        """
        import librosa
        
        # Resample if needed
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Pad/trim to 30 seconds as Whisper expects
        import whisper
        audio = whisper.pad_or_trim(audio)
        
        # Create mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        # Detect language if not specified
        if language is None:
            _, probs = self.model.detect_language(mel)
            language = max(probs, key=probs.get)
            language_probability = probs[language]
        else:
            language_probability = 1.0
        
        # Decode
        options = whisper.DecodingOptions(
            language=language,
            fp16=(self.device == "cuda"),
        )
        result = whisper.decode(self.model, mel, options)
        
        return {
            "text": result.text.strip(),
            "language": language,
            "language_probability": language_probability,
        }
    
    def transcribe_simple(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
    ) -> str:
        """
        Simple transcription - just returns the text.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            language: Language code
            
        Returns:
            Transcribed text string
        """
        result = self.transcribe(audio, sample_rate, language)
        return result["text"]
    
    def transcribe_full(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
    ) -> dict:
        """
        Full transcription with word-level timestamps.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate  
            language: Language code
            
        Returns:
            Dict with text, segments, and word timestamps
        """
        import librosa
        
        # Resample if needed
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Use the full transcribe method for longer audio
        result = self.model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
        )
        
        return {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": result["segments"],
        }
