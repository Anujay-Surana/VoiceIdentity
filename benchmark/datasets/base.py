"""Base classes for benchmark datasets."""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """
    Speaker annotation for a segment.
    
    Attributes:
        start: Start time in seconds
        end: End time in seconds
        speaker_id: Speaker identifier (from annotation)
        text: Optional transcript text
    """
    start: float
    end: float
    speaker_id: str
    text: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def to_tuple(self) -> Tuple[float, float, str]:
        """Convert to (start, end, speaker_id) tuple for metrics."""
        return (self.start, self.end, self.speaker_id)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "text": self.text,
        }


@dataclass
class BenchmarkSample:
    """
    A single sample from a benchmark dataset.
    
    Attributes:
        audio_path: Path to audio file
        annotations: List of speaker annotations
        sample_id: Unique identifier for this sample
        duration: Total audio duration in seconds
        metadata: Additional metadata (recording info, etc.)
    """
    audio_path: Path
    annotations: List[Annotation]
    sample_id: str
    duration: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    @property
    def speakers(self) -> List[str]:
        """Get unique speakers in this sample."""
        return list(set(ann.speaker_id for ann in self.annotations))
    
    @property
    def n_speakers(self) -> int:
        """Number of unique speakers."""
        return len(self.speakers)
    
    def get_annotations_as_tuples(self) -> List[Tuple[float, float, str]]:
        """Get annotations as list of tuples for metrics computation."""
        return [ann.to_tuple() for ann in self.annotations]


class BenchmarkDataset(ABC):
    """
    Abstract base class for benchmark datasets.
    
    Subclasses implement loading logic for specific datasets
    (VoxConverse, AMI, etc.)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split (train/dev/test)
            max_samples: Optional limit on number of samples to load
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_samples = max_samples
        self.samples: List[BenchmarkSample] = []
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {root_dir}")
        
        self._load_samples()
    
    @abstractmethod
    def _load_samples(self) -> None:
        """Load samples from disk. Implemented by subclasses."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name for reporting."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> BenchmarkSample:
        return self.samples[idx]
    
    def get_summary(self) -> Dict:
        """Get summary statistics for the dataset."""
        if not self.samples:
            return {"name": self.name, "n_samples": 0}
        
        total_duration = sum(s.duration for s in self.samples)
        all_speakers = set()
        for s in self.samples:
            all_speakers.update(s.speakers)
        
        speakers_per_sample = [s.n_speakers for s in self.samples]
        
        return {
            "name": self.name,
            "split": self.split,
            "n_samples": len(self.samples),
            "total_duration_hours": round(total_duration / 3600, 2),
            "n_unique_speakers": len(all_speakers),
            "avg_speakers_per_sample": round(sum(speakers_per_sample) / len(speakers_per_sample), 2),
            "avg_sample_duration": round(total_duration / len(self.samples), 2),
        }


def parse_rttm(rttm_path: Path) -> List[Annotation]:
    """
    Parse an RTTM (Rich Transcription Time Marked) file.
    
    RTTM format:
    SPEAKER <file> <channel> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    
    Args:
        rttm_path: Path to RTTM file
        
    Returns:
        List of Annotation objects
    """
    annotations = []
    
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) < 8:
                continue
            
            if parts[0] != "SPEAKER":
                continue
            
            try:
                start = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]
                
                annotations.append(Annotation(
                    start=start,
                    end=start + duration,
                    speaker_id=speaker_id,
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse RTTM line: {line.strip()}: {e}")
    
    return annotations


def parse_ctm(ctm_path: Path) -> List[Annotation]:
    """
    Parse a CTM (Conversation Time Marked) file with transcripts.
    
    CTM format:
    <file> <channel> <start> <dur> <word>
    
    Args:
        ctm_path: Path to CTM file
        
    Returns:
        List of Annotation objects
    """
    annotations = []
    
    with open(ctm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) < 5:
                continue
            
            try:
                start = float(parts[2])
                duration = float(parts[3])
                word = parts[4]
                speaker_id = parts[1] if len(parts) > 5 else "unknown"
                
                annotations.append(Annotation(
                    start=start,
                    end=start + duration,
                    speaker_id=speaker_id,
                    text=word,
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse CTM line: {line.strip()}: {e}")
    
    return annotations


def annotations_to_rttm(
    annotations: List[Annotation],
    file_id: str,
    output_path: Path,
) -> None:
    """
    Write annotations to RTTM format.
    
    Args:
        annotations: List of annotations to write
        file_id: File identifier for RTTM
        output_path: Path to write RTTM file
    """
    with open(output_path, 'w') as f:
        for ann in annotations:
            duration = ann.end - ann.start
            f.write(f"SPEAKER {file_id} 1 {ann.start:.3f} {duration:.3f} <NA> <NA> {ann.speaker_id} <NA> <NA>\n")
