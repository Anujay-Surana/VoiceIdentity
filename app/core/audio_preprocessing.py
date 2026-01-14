"""Audio preprocessing pipeline for improving quality.

This module provides a comprehensive preprocessing pipeline:
- Noise reduction (spectral gating)
- Audio normalization (EBU R128)
- High-pass filtering (remove low-frequency rumble)
- Quality scoring
"""

import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy import signal

logger = logging.getLogger(__name__)

# Default sample rate
DEFAULT_SAMPLE_RATE = 16000


@dataclass
class AudioQuality:
    """Audio quality metrics."""
    snr_db: float           # Signal-to-noise ratio in dB
    rms_energy: float       # RMS energy level
    duration: float         # Duration in seconds
    speech_ratio: float     # Ratio of speech to total (from VAD)
    overall_score: float    # Combined quality score (0-1)
    
    def is_acceptable(self, min_score: float = 0.3) -> bool:
        """Check if quality is acceptable for processing."""
        return self.overall_score >= min_score
    
    def is_high_quality(self, min_score: float = 0.6) -> bool:
        """Check if quality is high enough to store embedding."""
        return self.overall_score >= min_score


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for improving audio quality.
    
    Pipeline stages:
    1. DC offset removal
    2. High-pass filter (remove rumble)
    3. Noise reduction
    4. Normalization
    """
    
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        highpass_cutoff: float = 80.0,
        noise_reduce_strength: float = 0.75,
        target_loudness: float = -23.0,  # EBU R128 target
    ):
        """
        Initialize the preprocessor.
        
        Args:
            sample_rate: Target sample rate.
            highpass_cutoff: High-pass filter cutoff frequency (Hz).
            noise_reduce_strength: Noise reduction strength (0-1).
            target_loudness: Target loudness in LUFS.
        """
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff
        self.noise_reduce_strength = noise_reduce_strength
        self.target_loudness = target_loudness
        
        # Pre-compute high-pass filter coefficients
        self._highpass_b, self._highpass_a = signal.butter(
            4,  # 4th order
            highpass_cutoff / (sample_rate / 2),
            btype='high',
        )
    
    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        apply_noise_reduction: bool = True,
        apply_normalization: bool = True,
        apply_highpass: bool = True,
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            audio: Input audio samples (float32).
            sample_rate: Sample rate (uses default if None).
            apply_noise_reduction: Whether to apply noise reduction.
            apply_normalization: Whether to normalize loudness.
            apply_highpass: Whether to apply high-pass filter.
            
        Returns:
            Preprocessed audio.
        """
        if len(audio) == 0:
            return audio
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Step 1: Remove DC offset
        audio = self._remove_dc_offset(audio)
        
        # Step 2: High-pass filter
        if apply_highpass:
            audio = self._apply_highpass(audio)
        
        # Step 3: Noise reduction
        if apply_noise_reduction:
            audio = self._reduce_noise(audio, sample_rate or self.sample_rate)
        
        # Step 4: Normalization
        if apply_normalization:
            audio = self._normalize(audio, sample_rate or self.sample_rate)
        
        return audio
    
    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio."""
        return audio - np.mean(audio)
    
    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency rumble."""
        try:
            filtered = signal.filtfilt(self._highpass_b, self._highpass_a, audio)
            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning(f"[PREPROCESS] High-pass filter failed: {e}")
            return audio
    
    def _reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Apply noise reduction using spectral gating."""
        try:
            import noisereduce as nr
            
            # Apply noise reduction
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                prop_decrease=self.noise_reduce_strength,
                stationary=False,  # Adaptive to non-stationary noise
                n_fft=512,
                win_length=400,
                hop_length=100,
            )
            
            return reduced.astype(np.float32)
        except ImportError:
            logger.warning("[PREPROCESS] noisereduce not installed, skipping")
            return audio
        except Exception as e:
            logger.warning(f"[PREPROCESS] Noise reduction failed: {e}")
            return audio
    
    def _normalize(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Normalize audio loudness to target LUFS."""
        try:
            import pyloudnorm as pyln
            
            # Create meter
            meter = pyln.Meter(sample_rate)
            
            # Measure current loudness
            current_loudness = meter.integrated_loudness(audio)
            
            # Skip if audio is silent
            if np.isinf(current_loudness) or np.isnan(current_loudness):
                return audio
            
            # Normalize to target
            normalized = pyln.normalize.loudness(
                audio,
                current_loudness,
                self.target_loudness,
            )
            
            # Clip to prevent distortion
            normalized = np.clip(normalized, -1.0, 1.0)
            
            return normalized.astype(np.float32)
        except ImportError:
            logger.warning("[PREPROCESS] pyloudnorm not installed, using peak normalization")
            return self._peak_normalize(audio)
        except Exception as e:
            logger.warning(f"[PREPROCESS] Loudness normalization failed: {e}")
            return self._peak_normalize(audio)
    
    def _peak_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Simple peak normalization fallback."""
        peak = np.abs(audio).max()
        if peak > 0:
            return (audio / peak * 0.9).astype(np.float32)
        return audio


def compute_quality_score(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speech_ratio: Optional[float] = None,
) -> AudioQuality:
    """
    Compute comprehensive quality score for audio.
    
    Args:
        audio: Audio samples.
        sample_rate: Sample rate.
        speech_ratio: Pre-computed speech ratio from VAD (optional).
        
    Returns:
        AudioQuality object with metrics and overall score.
    """
    if len(audio) == 0:
        return AudioQuality(
            snr_db=0, rms_energy=0, duration=0,
            speech_ratio=0, overall_score=0
        )
    
    duration = len(audio) / sample_rate
    
    # Compute RMS energy
    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    
    # Estimate SNR (signal-to-noise ratio)
    snr_db = estimate_snr(audio, sample_rate)
    
    # Use provided speech ratio or default to 1.0
    if speech_ratio is None:
        speech_ratio = 1.0
    
    # Compute overall score (weighted combination)
    score = compute_weighted_score(
        snr_db=snr_db,
        rms_energy=rms_energy,
        duration=duration,
        speech_ratio=speech_ratio,
    )
    
    return AudioQuality(
        snr_db=snr_db,
        rms_energy=rms_energy,
        duration=duration,
        speech_ratio=speech_ratio,
        overall_score=score,
    )


def estimate_snr(audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate signal-to-noise ratio in dB.
    
    Uses a simple method: compare signal power in speech-likely
    frequencies vs. noise-likely frequencies.
    """
    try:
        from scipy import signal as scipy_signal
        
        # Compute power spectrum
        freqs, psd = scipy_signal.welch(audio, sample_rate, nperseg=1024)
        
        # Speech frequencies: 300-3400 Hz
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        # Noise frequencies: below 100 Hz and above 4000 Hz
        noise_mask = (freqs < 100) | (freqs > 4000)
        
        if not np.any(speech_mask) or not np.any(noise_mask):
            return 20.0  # Default to reasonable value
        
        speech_power = np.mean(psd[speech_mask])
        noise_power = np.mean(psd[noise_mask])
        
        if noise_power > 0:
            snr_db = 10 * np.log10(speech_power / noise_power)
        else:
            snr_db = 40.0  # Very clean signal
        
        # Clamp to reasonable range
        return float(np.clip(snr_db, 0, 40))
    except Exception:
        return 20.0  # Default fallback


def compute_weighted_score(
    snr_db: float,
    rms_energy: float,
    duration: float,
    speech_ratio: float,
) -> float:
    """
    Compute weighted quality score from individual metrics.
    
    Returns a score from 0 to 1.
    """
    # Normalize each metric to 0-1 range
    
    # SNR: 5 dB = 0, 30 dB = 1
    snr_score = np.clip((snr_db - 5) / 25, 0, 1)
    
    # Energy: 0.01 = 0, 0.3 = 1
    energy_score = np.clip((rms_energy - 0.01) / 0.29, 0, 1)
    
    # Duration: 0.5s = 0, 3.0s = 1
    duration_score = np.clip((duration - 0.5) / 2.5, 0, 1)
    
    # Speech ratio: already 0-1
    speech_score = speech_ratio
    
    # Weighted combination
    score = (
        0.30 * snr_score +
        0.20 * energy_score +
        0.25 * duration_score +
        0.25 * speech_score
    )
    
    return float(np.clip(score, 0, 1))


def get_adaptive_threshold(
    quality_score: float,
    base_threshold: float = 0.45,
) -> float:
    """
    Get adaptive matching threshold based on audio quality.
    
    - High quality audio: Lower threshold (more permissive)
    - Low quality audio: Higher threshold (more strict)
    
    Args:
        quality_score: Audio quality score (0-1).
        base_threshold: Base matching threshold.
        
    Returns:
        Adjusted threshold.
    """
    if quality_score > 0.7:
        # High quality - be more permissive
        return base_threshold - 0.05
    elif quality_score < 0.4:
        # Low quality - be more strict to avoid false matches
        return base_threshold + 0.10
    else:
        # Medium quality - use base threshold
        return base_threshold


# Singleton instance
_preprocessor: Optional[AudioPreprocessor] = None


def get_preprocessor() -> AudioPreprocessor:
    """Get or create the global preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = AudioPreprocessor()
    return _preprocessor
