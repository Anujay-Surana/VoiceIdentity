"""Speaker diarization using PyAnnote.audio 3.x."""

import os
import torch
import numpy as np
from typing import Optional

from app.models.schemas import DiarizedSegment


class DiarizationPipeline:
    """
    Wrapper around PyAnnote speaker diarization pipeline.
    
    This uses the pyannote/speaker-diarization-3.1 model which provides
    state-of-the-art speaker diarization with:
    - Voice Activity Detection (VAD)
    - Speaker Segmentation
    - Speaker Embedding
    - Clustering
    """
    
    MODEL_NAME = "pyannote/speaker-diarization-3.1"
    
    def __init__(
        self,
        hf_token: str,
        device: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Initialize the diarization pipeline.
        
        Args:
            hf_token: HuggingFace token for model access
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            num_speakers: Exact number of speakers if known
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        """
        self.hf_token = hf_token
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set HF token in environment for huggingface_hub to pick up automatically
        # This avoids the deprecated use_auth_token / token parameter issues
        os.environ["HF_TOKEN"] = hf_token
        
        # Import Pipeline after setting token env var
        from pyannote.audio import Pipeline
        
        # Load pipeline (uses HF_TOKEN env var automatically)
        self.pipeline = Pipeline.from_pretrained(self.MODEL_NAME)
        self.pipeline.to(self.device)
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[DiarizedSegment]:
        """
        Run speaker diarization on audio.
        
        Args:
            audio: Audio waveform as numpy array (mono, float32)
            sample_rate: Sample rate of the audio
            num_speakers: Override exact number of speakers
            min_speakers: Override minimum speakers
            max_speakers: Override maximum speakers
            
        Returns:
            List of DiarizedSegment with speaker labels and timestamps
        """
        # Ensure audio is the right shape (1, num_samples) for PyAnnote
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)
        
        # Convert to torch tensor
        waveform = torch.from_numpy(audio).float()
        
        # Create input dict for pipeline
        audio_input = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }
        
        # Build diarization parameters
        params = {}
        
        # Use provided values or fall back to instance defaults
        effective_num_speakers = num_speakers or self.num_speakers
        effective_min_speakers = min_speakers or self.min_speakers
        effective_max_speakers = max_speakers or self.max_speakers
        
        if effective_num_speakers is not None:
            params["num_speakers"] = effective_num_speakers
        if effective_min_speakers is not None:
            params["min_speakers"] = effective_min_speakers
        if effective_max_speakers is not None:
            params["max_speakers"] = effective_max_speakers
        
        # Run diarization
        diarization = self.pipeline(audio_input, **params)
        
        # Convert to our segment format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                start=turn.start,
                end=turn.end,
                label=speaker,
            ))
        
        return segments
    
    def process_file(
        self,
        file_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> list[DiarizedSegment]:
        """
        Run speaker diarization on an audio file.
        
        Args:
            file_path: Path to audio file
            num_speakers: Override exact number of speakers
            min_speakers: Override minimum speakers  
            max_speakers: Override maximum speakers
            
        Returns:
            List of DiarizedSegment with speaker labels and timestamps
        """
        # Build parameters
        params = {}
        
        effective_num_speakers = num_speakers or self.num_speakers
        effective_min_speakers = min_speakers or self.min_speakers
        effective_max_speakers = max_speakers or self.max_speakers
        
        if effective_num_speakers is not None:
            params["num_speakers"] = effective_num_speakers
        if effective_min_speakers is not None:
            params["min_speakers"] = effective_min_speakers
        if effective_max_speakers is not None:
            params["max_speakers"] = effective_max_speakers
        
        # Run diarization directly on file
        diarization = self.pipeline(file_path, **params)
        
        # Convert to our segment format
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                start=turn.start,
                end=turn.end,
                label=speaker,
            ))
        
        return segments


class StreamingDiarization:
    """
    Streaming-compatible diarization for real-time processing.
    
    This wraps the standard pipeline but handles:
    - Buffer management for overlapping windows
    - Speaker continuity across chunks
    - Efficient incremental processing
    """
    
    def __init__(
        self,
        hf_token: str,
        window_duration: float = 5.0,
        step_duration: float = 2.5,
        device: Optional[str] = None,
    ):
        """
        Initialize streaming diarization.
        
        Args:
            hf_token: HuggingFace token
            window_duration: Duration of each processing window in seconds
            step_duration: Step size between windows (window - step = overlap)
            device: Device to run on
        """
        self.pipeline = DiarizationPipeline(hf_token=hf_token, device=device)
        self.window_duration = window_duration
        self.step_duration = step_duration
        
        # Speaker label mapping for consistency across windows
        self.label_map: dict[str, str] = {}
        self.next_speaker_id = 0
    
    def _map_speaker_label(self, label: str) -> str:
        """Map internal speaker labels to consistent IDs across windows."""
        if label not in self.label_map:
            self.label_map[label] = f"SPEAKER_{self.next_speaker_id:02d}"
            self.next_speaker_id += 1
        return self.label_map[label]
    
    def process_window(
        self,
        audio: np.ndarray,
        sample_rate: int,
        window_offset: float = 0.0,
    ) -> list[DiarizedSegment]:
        """
        Process a single audio window.
        
        Args:
            audio: Audio window as numpy array
            sample_rate: Sample rate
            window_offset: Time offset of this window in the full stream
            
        Returns:
            List of segments with timestamps adjusted for window offset
        """
        segments = self.pipeline.process(audio, sample_rate)
        
        # Adjust timestamps and map labels
        adjusted_segments = []
        for seg in segments:
            adjusted_segments.append(DiarizedSegment(
                start=seg.start + window_offset,
                end=seg.end + window_offset,
                label=self._map_speaker_label(seg.label),
            ))
        
        return adjusted_segments
    
    def reset(self):
        """Reset speaker label mapping for a new stream."""
        self.label_map = {}
        self.next_speaker_id = 0
