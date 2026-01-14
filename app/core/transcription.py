"""Speech-to-text transcription using OpenAI Whisper.

Enhanced features:
- VAD integration for improved accuracy on noisy audio
- Configurable decoding options for quality/speed tradeoff
- Preprocessing integration
"""

import numpy as np
import logging
import torch
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Target sample rate for Whisper (16kHz)
WHISPER_SAMPLE_RATE = 16000


class TranscriptionService:
    """
    Transcribe audio segments using OpenAI's Whisper.
    
    Whisper is a robust speech recognition model that works well
    across many languages and audio conditions.
    
    Enhanced with:
    - VAD preprocessing for noisy audio
    - Configurable parameters for quality vs. speed
    - Better handling of short segments
    """
    
    def __init__(
        self,
        model_size: str = "base",  # Options: tiny, base, small, medium, large
        device: str = "auto",  # auto, cpu, cuda
        download_root: Optional[str] = None,  # Cache location (auto-detected if None)
        no_speech_threshold: float = 0.6,  # Higher = stricter speech detection
        condition_on_previous_text: bool = False,  # Disable to prevent error propagation
    ):
        """
        Initialize the transcription service.
        
        Args:
            model_size: Whisper model size (base is good balance of speed/accuracy)
            device: Device to run on (auto-detects GPU if available)
            download_root: Directory to cache Whisper models (auto-detected if None)
            no_speech_threshold: Threshold for detecting no-speech segments
            condition_on_previous_text: Whether to use previous text as context
        """
        import whisper
        import os
        
        logger.info(f"[TRANSCRIBE] Loading Whisper model: {model_size}")
        
        # Store configuration
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        
        # Auto-detect download root
        if download_root is None:
            # Docker path
            docker_path = "/app/.cache/whisper"
            if os.path.exists("/app") and os.access("/app", os.W_OK):
                download_root = docker_path
            else:
                # Local development: use default Whisper cache (~/.cache/whisper)
                download_root = None  # Let Whisper use its default
        
        self._download_root = download_root
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with explicit cache path if specified
        if self._download_root:
            self.model = whisper.load_model(model_size, device=device, download_root=self._download_root)
        else:
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
    
    def transcribe_with_vad(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
        apply_preprocessing: bool = True,
    ) -> dict:
        """
        Transcribe audio with VAD preprocessing for better accuracy.
        
        This method:
        1. Applies audio preprocessing (noise reduction, normalization)
        2. Uses VAD to identify speech regions
        3. Transcribes only speech regions for better accuracy
        
        Args:
            audio: Audio waveform.
            sample_rate: Sample rate.
            language: Language code.
            apply_preprocessing: Whether to apply noise reduction.
            
        Returns:
            Dict with text and quality metrics.
        """
        import librosa
        
        # Resample if needed
        if sample_rate != WHISPER_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=WHISPER_SAMPLE_RATE)
            sample_rate = WHISPER_SAMPLE_RATE
        
        audio = audio.astype(np.float32)
        
        # Apply preprocessing
        if apply_preprocessing:
            try:
                from app.core.audio_preprocessing import get_preprocessor
                preprocessor = get_preprocessor()
                audio = preprocessor.preprocess(audio, sample_rate)
            except ImportError:
                logger.warning("[TRANSCRIBE] Preprocessing not available")
        
        # Check if audio contains speech using VAD
        speech_ratio = 1.0
        try:
            from app.core.vad import get_vad
            vad = get_vad()
            speech_ratio = vad.get_speech_ratio(audio, sample_rate)
            
            if speech_ratio < 0.1:
                logger.debug("[TRANSCRIBE] Very little speech detected, returning empty")
                return {
                    "text": "",
                    "language": language,
                    "speech_ratio": speech_ratio,
                    "confidence": 0.0,
                }
        except ImportError:
            logger.warning("[TRANSCRIBE] VAD not available")
        
        # Transcribe
        result = self.transcribe(audio, sample_rate, language)
        result["speech_ratio"] = speech_ratio
        result["confidence"] = speech_ratio * result.get("language_probability", 1.0)
        
        return result
    
    def transcribe_segment_robust(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
        min_duration: float = 0.5,
        max_duration: float = 30.0,
    ) -> str:
        """
        Robust transcription for variable-length segments.
        
        Handles short and long segments appropriately:
        - Short segments (<0.5s): Return empty (too short to transcribe reliably)
        - Medium segments (0.5-30s): Use standard transcription
        - Long segments (>30s): Split and transcribe in chunks
        
        Args:
            audio: Audio waveform.
            sample_rate: Sample rate.
            language: Language code.
            min_duration: Minimum duration to attempt transcription.
            max_duration: Maximum duration before splitting.
            
        Returns:
            Transcribed text string.
        """
        duration = len(audio) / sample_rate
        
        # Too short
        if duration < min_duration:
            logger.debug(f"[TRANSCRIBE] Segment too short ({duration:.2f}s), skipping")
            return ""
        
        # Standard length
        if duration <= max_duration:
            try:
                return self.transcribe_simple(audio, sample_rate, language)
            except Exception as e:
                logger.warning(f"[TRANSCRIBE] Failed: {e}")
                return ""
        
        # Long segment - split into chunks
        logger.debug(f"[TRANSCRIBE] Long segment ({duration:.2f}s), splitting")
        
        chunk_duration = 25.0  # Leave some margin from 30s limit
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(2.0 * sample_rate)  # 2s overlap
        
        transcripts: List[str] = []
        offset = 0
        
        while offset < len(audio):
            end = min(offset + chunk_samples, len(audio))
            chunk = audio[offset:end]
            
            try:
                text = self.transcribe_simple(chunk, sample_rate, language)
                if text:
                    transcripts.append(text)
            except Exception as e:
                logger.warning(f"[TRANSCRIBE] Chunk failed: {e}")
            
            offset += chunk_samples - overlap_samples
        
        return " ".join(transcripts)
