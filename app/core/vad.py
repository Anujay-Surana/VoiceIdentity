"""Voice Activity Detection using Silero VAD.

Silero VAD is a fast, accurate voice activity detector that helps:
- Filter out silence and non-speech segments
- Improve embedding quality by focusing on speech
- Speed up processing by skipping non-speech
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Target sample rate for Silero VAD
VAD_SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    
    @property
    def duration(self) -> float:
        return self.end - self.start


class VoiceActivityDetector:
    """
    Voice Activity Detection using Silero VAD.
    
    Silero VAD is trained on a large dataset and provides:
    - High accuracy speech detection
    - Fast inference (works on CPU)
    - Robust to noise
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.1,
        window_size_samples: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize the VAD.
        
        Args:
            threshold: Speech probability threshold (0-1). Higher = stricter.
            min_speech_duration: Minimum speech segment duration in seconds.
            min_silence_duration: Minimum silence to split segments.
            window_size_samples: Processing window size (512 for 16kHz).
            device: Device to run on (auto-detected if None).
        """
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.window_size_samples = window_size_samples
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load Silero VAD model
        logger.info(f"[VAD] Loading Silero VAD model on {device}...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        self.model = self.model.to(device)
        
        # Extract utility functions
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils
        
        logger.info("[VAD] Silero VAD loaded successfully")
    
    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int = VAD_SAMPLE_RATE,
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Audio samples as numpy array (float32, mono).
            sample_rate: Sample rate of audio.
            
        Returns:
            List of SpeechSegment objects with start/end times.
        """
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        # Ensure 1D
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        
        # Resample if needed
        if sample_rate != VAD_SAMPLE_RATE:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, VAD_SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor)
            sample_rate = VAD_SAMPLE_RATE
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=int(self.min_speech_duration * 1000),
            min_silence_duration_ms=int(self.min_silence_duration * 1000),
            window_size_samples=self.window_size_samples,
            return_seconds=False,
        )
        
        # Convert to SpeechSegment objects
        segments = []
        for ts in speech_timestamps:
            start_sec = ts['start'] / sample_rate
            end_sec = ts['end'] / sample_rate
            segments.append(SpeechSegment(start=start_sec, end=end_sec))
        
        logger.debug(f"[VAD] Detected {len(segments)} speech segments")
        return segments
    
    def get_speech_ratio(
        self,
        audio: np.ndarray,
        sample_rate: int = VAD_SAMPLE_RATE,
    ) -> float:
        """
        Get the ratio of speech to total audio duration.
        
        Args:
            audio: Audio samples.
            sample_rate: Sample rate.
            
        Returns:
            Ratio of speech duration to total duration (0-1).
        """
        total_duration = len(audio) / sample_rate
        if total_duration == 0:
            return 0.0
        
        segments = self.detect_speech(audio, sample_rate)
        speech_duration = sum(seg.duration for seg in segments)
        
        return min(1.0, speech_duration / total_duration)
    
    def extract_speech(
        self,
        audio: np.ndarray,
        sample_rate: int = VAD_SAMPLE_RATE,
        padding: float = 0.1,
    ) -> Tuple[np.ndarray, List[SpeechSegment]]:
        """
        Extract only speech portions from audio.
        
        Args:
            audio: Input audio samples.
            sample_rate: Sample rate.
            padding: Padding around speech segments in seconds.
            
        Returns:
            Tuple of (concatenated speech audio, list of segments).
        """
        segments = self.detect_speech(audio, sample_rate)
        
        if not segments:
            return np.array([], dtype=np.float32), []
        
        # Extract audio for each segment with padding
        speech_chunks = []
        padding_samples = int(padding * sample_rate)
        
        for seg in segments:
            start_sample = max(0, int(seg.start * sample_rate) - padding_samples)
            end_sample = min(len(audio), int(seg.end * sample_rate) + padding_samples)
            speech_chunks.append(audio[start_sample:end_sample])
        
        # Concatenate all speech
        if speech_chunks:
            concatenated = np.concatenate(speech_chunks)
        else:
            concatenated = np.array([], dtype=np.float32)
        
        return concatenated, segments
    
    def filter_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = VAD_SAMPLE_RATE,
        min_speech_ratio: float = 0.3,
    ) -> Optional[np.ndarray]:
        """
        Filter audio - return None if too little speech.
        
        Args:
            audio: Input audio.
            sample_rate: Sample rate.
            min_speech_ratio: Minimum speech ratio to accept.
            
        Returns:
            Audio if speech ratio is sufficient, None otherwise.
        """
        speech_ratio = self.get_speech_ratio(audio, sample_rate)
        
        if speech_ratio < min_speech_ratio:
            logger.debug(f"[VAD] Rejecting audio: speech_ratio={speech_ratio:.2f} < {min_speech_ratio}")
            return None
        
        return audio


# Singleton instance for reuse
_vad_instance: Optional[VoiceActivityDetector] = None


def get_vad() -> VoiceActivityDetector:
    """Get or create the global VAD instance."""
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = VoiceActivityDetector()
    return _vad_instance
