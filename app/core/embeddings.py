"""Voice embedding extraction using SpeechBrain ECAPA-TDNN."""

import torch
import torchaudio
import numpy as np
import logging
from typing import Optional, Union
from pathlib import Path

from speechbrain.inference.speaker import EncoderClassifier

logger = logging.getLogger(__name__)

# Target sample rate for ECAPA-TDNN (CRITICAL - model trained on 16kHz)
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.5  # Minimum 0.5 seconds for embedding
OPTIMAL_AUDIO_DURATION = 2.0  # Optimal 2+ seconds for best quality


class EmbeddingExtractor:
    """
    Extract voice embeddings using SpeechBrain's ECAPA-TDNN model.
    
    The ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
    in TDNN) model produces 192-dimensional embeddings that capture unique
    voice characteristics for speaker recognition.
    
    This model achieves state-of-the-art performance on VoxCeleb benchmarks.
    
    CRITICAL: Model expects 16kHz audio. Audio at other sample rates will
    be automatically resampled.
    """
    
    # Pre-trained model from SpeechBrain hub
    MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
    EMBEDDING_DIM = 192
    
    def __init__(
        self,
        device: Optional[str] = None,
        savedir: str = "pretrained_models/spkrec-ecapa-voxceleb",
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            savedir: Directory to cache the downloaded model
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load pre-trained model
        self.model = EncoderClassifier.from_hparams(
            source=self.MODEL_SOURCE,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        
        # Resampler cache for efficiency
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
    
    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Get or create a resampler for the given sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=TARGET_SAMPLE_RATE,
            ).to(self.device)
        return self._resamplers[orig_sr]
    
    def _preprocess_audio(
        self, 
        waveform: torch.Tensor, 
        sample_rate: int
    ) -> torch.Tensor:
        """
        Preprocess audio for optimal embedding extraction.
        
        - Resample to 16kHz if needed (CRITICAL!)
        - Normalize amplitude
        - Remove DC offset
        """
        # CRITICAL: Resample to 16kHz if not already
        if sample_rate != TARGET_SAMPLE_RATE:
            logger.debug(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)
        
        # Remove DC offset (center around zero)
        waveform = waveform - waveform.mean()
        
        # Normalize amplitude to [-1, 1] for consistent embeddings
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val
        
        return waveform
    
    def extract(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract voice embedding from audio.
        
        Args:
            audio: Audio waveform (mono, float32)
            sample_rate: Sample rate of input audio (will be resampled to 16kHz)
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            192-dimensional embedding as numpy array
        """
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
        else:
            waveform = audio.float()
        
        # Ensure correct shape (batch, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # CRITICAL: Preprocess audio (resample, normalize)
        waveform = self._preprocess_audio(waveform, sample_rate)
        
        # Check minimum duration
        duration = waveform.shape[1] / TARGET_SAMPLE_RATE
        if duration < MIN_AUDIO_DURATION:
            logger.warning(f"Audio too short ({duration:.2f}s < {MIN_AUDIO_DURATION}s) - embedding may be unreliable")
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
        
        # Convert to numpy and squeeze batch dimension
        embedding = embedding.squeeze().cpu().numpy()
        
        # L2 normalize for cosine similarity
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def extract_batch(
        self,
        audio_list: list[np.ndarray],
        sample_rate: int = 16000,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for multiple audio segments efficiently.
        
        Args:
            audio_list: List of audio waveforms
            sample_rate: Sample rate
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Array of shape (batch_size, 192) with embeddings
        """
        # Pad to same length for batching
        max_len = max(len(a) for a in audio_list)
        
        batch = []
        for audio in audio_list:
            # Pad with zeros
            padded = np.zeros(max_len, dtype=np.float32)
            padded[:len(audio)] = audio
            batch.append(padded)
        
        # Stack into batch tensor
        waveforms = torch.from_numpy(np.stack(batch)).float().to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model.encode_batch(waveforms)
        
        # Convert to numpy
        embeddings = embeddings.squeeze(1).cpu().numpy()
        
        # Normalize
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            embeddings = embeddings / norms
        
        return embeddings
    
    def extract_from_file(
        self,
        file_path: Union[str, Path],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract embedding directly from an audio file.
        
        Args:
            file_path: Path to audio file
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            192-dimensional embedding as numpy array
        """
        # SpeechBrain can load files directly
        with torch.no_grad():
            embedding = self.model.encode_batch(
                self.model.load_audio(str(file_path))
            )
        
        embedding = embedding.squeeze().cpu().numpy()
        
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        For normalized embeddings, this is equivalent to dot product.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score in range [-1, 1], higher = more similar
        """
        return float(np.dot(embedding1, embedding2))
    
    @staticmethod
    def compute_centroid(embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Compute centroid (mean) of multiple embeddings.
        
        Useful for aggregating multiple samples from the same speaker.
        
        Args:
            embeddings: Array of shape (n_samples, 192)
            normalize: Whether to L2-normalize the centroid
            
        Returns:
            192-dimensional centroid embedding
        """
        centroid = embeddings.mean(axis=0)
        
        if normalize:
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
        
        return centroid
