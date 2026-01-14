"""AMI Meeting Corpus dataset loader.

The AMI Meeting Corpus is a multi-modal dataset of meeting recordings
used for evaluating speech and speaker recognition systems.

Dataset: https://groups.inf.ed.ac.uk/ami/corpus/
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

from benchmark.datasets.base import (
    BenchmarkDataset,
    BenchmarkSample,
    Annotation,
    parse_rttm,
)

logger = logging.getLogger(__name__)


class AMIDataset(BenchmarkDataset):
    """
    AMI Meeting Corpus dataset loader.
    
    Expected directory structure:
    root_dir/
        amicorpus/
            <meeting_id>/
                audio/
                    <meeting_id>.Mix-Headset.wav
        annotations/
            <meeting_id>.rttm
    
    Or with pyannote-style annotations:
    root_dir/
        audio/
            <meeting_id>.wav
        rttm/
            <meeting_id>.rttm
    """
    
    # Standard AMI meeting IDs
    TEST_MEETINGS = [
        "EN2001a", "EN2001b", "EN2001d", "EN2001e",
        "EN2003a", "EN2004a", "EN2005a", "EN2006a",
        "EN2006b", "EN2009b", "EN2009c", "EN2009d",
        "ES2004a", "ES2004b", "ES2004c", "ES2004d",
        "ES2014a", "ES2014b", "ES2014c", "ES2014d",
        "IS1009a", "IS1009b", "IS1009c", "IS1009d",
        "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    ]
    
    DEV_MEETINGS = [
        "ES2002a", "ES2002b", "ES2002c", "ES2002d",
        "ES2005a", "ES2005b", "ES2005c", "ES2005d",
        "ES2006a", "ES2006b", "ES2006c", "ES2006d",
        "ES2007a", "ES2007b", "ES2007c", "ES2007d",
        "IS1003a", "IS1003b", "IS1003c", "IS1003d",
    ]
    
    @property
    def name(self) -> str:
        return "AMI"
    
    def _load_samples(self) -> None:
        """Load AMI corpus samples."""
        # Determine which meetings to load
        if self.split == "test":
            target_meetings = set(self.TEST_MEETINGS)
        elif self.split == "dev":
            target_meetings = set(self.DEV_MEETINGS)
        else:
            target_meetings = None  # Load all
        
        # Find audio and annotation files
        audio_files = self._find_audio_files()
        rttm_files = self._find_rttm_files()
        
        if not audio_files:
            logger.warning(f"No audio files found in {self.root_dir}")
            return
        
        if not rttm_files:
            logger.warning(f"No RTTM files found in {self.root_dir}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files, {len(rttm_files)} RTTM files")
        
        # Match audio with annotations
        count = 0
        for meeting_id, audio_path in sorted(audio_files.items()):
            if self.max_samples and count >= self.max_samples:
                break
            
            if target_meetings and meeting_id not in target_meetings:
                continue
            
            if meeting_id not in rttm_files:
                logger.warning(f"No RTTM for meeting {meeting_id}")
                continue
            
            # Parse annotations
            annotations = parse_rttm(rttm_files[meeting_id])
            
            if not annotations:
                logger.warning(f"No annotations for {meeting_id}")
                continue
            
            # Calculate duration
            duration = max(ann.end for ann in annotations)
            
            sample = BenchmarkSample(
                audio_path=audio_path,
                annotations=annotations,
                sample_id=meeting_id,
                duration=duration,
                metadata={
                    "rttm_file": str(rttm_files[meeting_id]),
                    "dataset": "ami",
                    "split": self.split,
                },
            )
            
            self.samples.append(sample)
            count += 1
        
        logger.info(f"Loaded {len(self.samples)} AMI samples")
    
    def _find_audio_files(self) -> Dict[str, Path]:
        """Find all audio files and map to meeting IDs."""
        audio_files = {}
        
        # Search patterns
        patterns = [
            "**/*.Mix-Headset.wav",
            "**/*.wav",
            "audio/*.wav",
        ]
        
        for pattern in patterns:
            for audio_path in self.root_dir.glob(pattern):
                # Extract meeting ID from filename
                meeting_id = self._extract_meeting_id(audio_path.stem)
                if meeting_id and meeting_id not in audio_files:
                    audio_files[meeting_id] = audio_path
        
        return audio_files
    
    def _find_rttm_files(self) -> Dict[str, Path]:
        """Find all RTTM files and map to meeting IDs."""
        rttm_files = {}
        
        for rttm_path in self.root_dir.glob("**/*.rttm"):
            meeting_id = self._extract_meeting_id(rttm_path.stem)
            if meeting_id and meeting_id not in rttm_files:
                rttm_files[meeting_id] = rttm_path
        
        return rttm_files
    
    def _extract_meeting_id(self, filename: str) -> Optional[str]:
        """Extract AMI meeting ID from filename."""
        # AMI meeting IDs are like: EN2001a, ES2004b, IS1009c, TS3003d
        import re
        
        # Remove common suffixes
        for suffix in [".Mix-Headset", ".Array1", ".Headset"]:
            filename = filename.replace(suffix, "")
        
        # Match AMI meeting ID pattern
        match = re.match(r"([A-Z]{2}\d{4}[a-d])", filename)
        if match:
            return match.group(1)
        
        return filename


def download_ami_subset(
    output_dir: str,
    meetings: Optional[List[str]] = None,
) -> Path:
    """
    Download a subset of AMI corpus.
    
    Note: Full AMI corpus is large (~100GB). This downloads
    a small subset for testing.
    
    Args:
        output_dir: Directory to save dataset
        meetings: List of meeting IDs to download (default: first 4 test meetings)
        
    Returns:
        Path to downloaded dataset
    """
    logger.info("AMI corpus must be downloaded manually from:")
    logger.info("https://groups.inf.ed.ac.uk/ami/download/")
    logger.info("")
    logger.info("For quick testing, you can use pyannote's prepared version:")
    logger.info("pip install pyannote.audio")
    logger.info("from pyannote.database import get_protocol")
    logger.info('protocol = get_protocol("AMI.SpeakerDiarization.only_words")')
    
    return Path(output_dir)
