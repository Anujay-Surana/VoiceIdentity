"""VoxConverse dataset loader.

VoxConverse is a popular benchmark for speaker diarization consisting
of multi-speaker conversations extracted from YouTube videos.

Dataset: https://github.com/joonson/voxconverse
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

from benchmark.datasets.base import (
    BenchmarkDataset,
    BenchmarkSample,
    Annotation,
    parse_rttm,
)

logger = logging.getLogger(__name__)


class VoxConverseDataset(BenchmarkDataset):
    """
    VoxConverse dataset loader.
    
    Expected directory structure:
    root_dir/
        audio/
            test/
                *.wav
        dev/
            *.rttm
        test/
            *.rttm
    
    Or alternative structure:
    root_dir/
        voxconverse_test_wav/
            *.wav
        test/
            *.rttm
    """
    
    @property
    def name(self) -> str:
        return "VoxConverse"
    
    def _load_samples(self) -> None:
        """Load VoxConverse samples."""
        # Find audio directory
        audio_dir = self._find_audio_dir()
        if audio_dir is None:
            logger.warning(f"Could not find audio directory in {self.root_dir}")
            return
        
        # Find RTTM directory
        rttm_dir = self._find_rttm_dir()
        if rttm_dir is None:
            logger.warning(f"Could not find RTTM directory in {self.root_dir}")
            return
        
        logger.info(f"Loading VoxConverse from {self.root_dir}")
        logger.info(f"  Audio dir: {audio_dir}")
        logger.info(f"  RTTM dir: {rttm_dir}")
        
        # Load samples
        count = 0
        for rttm_file in sorted(rttm_dir.glob("*.rttm")):
            if self.max_samples and count >= self.max_samples:
                break
            
            # Find matching audio file
            sample_id = rttm_file.stem
            audio_path = self._find_audio_file(audio_dir, sample_id)
            
            if audio_path is None:
                logger.warning(f"No audio found for {sample_id}")
                continue
            
            # Parse annotations
            annotations = parse_rttm(rttm_file)
            
            if not annotations:
                logger.warning(f"No annotations in {rttm_file}")
                continue
            
            # Calculate duration from annotations
            duration = max(ann.end for ann in annotations)
            
            sample = BenchmarkSample(
                audio_path=audio_path,
                annotations=annotations,
                sample_id=sample_id,
                duration=duration,
                metadata={
                    "rttm_file": str(rttm_file),
                    "dataset": "voxconverse",
                    "split": self.split,
                },
            )
            
            self.samples.append(sample)
            count += 1
        
        logger.info(f"Loaded {len(self.samples)} VoxConverse samples")
    
    def _find_audio_dir(self) -> Optional[Path]:
        """Find the audio directory."""
        # Try common directory structures
        candidates = [
            self.root_dir / "audio" / self.split,
            self.root_dir / f"voxconverse_{self.split}_wav",
            self.root_dir / self.split / "audio",
            self.root_dir / "wav",
            self.root_dir,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                wav_files = list(candidate.glob("*.wav"))
                if wav_files:
                    return candidate
        
        return None
    
    def _find_rttm_dir(self) -> Optional[Path]:
        """Find the RTTM annotation directory."""
        # Try common directory structures
        candidates = [
            self.root_dir / self.split,
            self.root_dir / "rttm" / self.split,
            self.root_dir / f"{self.split}_rttm",
            self.root_dir,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                rttm_files = list(candidate.glob("*.rttm"))
                if rttm_files:
                    return candidate
        
        return None
    
    def _find_audio_file(
        self,
        audio_dir: Path,
        sample_id: str,
    ) -> Optional[Path]:
        """Find audio file for a given sample ID."""
        # Try common extensions
        extensions = [".wav", ".flac", ".mp3", ".m4a"]
        
        for ext in extensions:
            audio_path = audio_dir / f"{sample_id}{ext}"
            if audio_path.exists():
                return audio_path
        
        # Try without extension
        for file in audio_dir.iterdir():
            if file.stem == sample_id:
                return file
        
        return None


def download_voxconverse(
    output_dir: str,
    split: str = "test",
) -> Path:
    """
    Download VoxConverse dataset.
    
    Note: This downloads the annotations. Audio must be obtained
    separately due to YouTube licensing.
    
    Args:
        output_dir: Directory to save dataset
        split: Dataset split (dev/test)
        
    Returns:
        Path to downloaded dataset
    """
    import urllib.request
    import zipfile
    import tempfile
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # VoxConverse RTTM annotations URL
    rttm_url = "https://github.com/joonson/voxconverse/archive/refs/heads/master.zip"
    
    logger.info(f"Downloading VoxConverse annotations to {output_dir}")
    
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        urllib.request.urlretrieve(rttm_url, f.name)
        
        with zipfile.ZipFile(f.name, 'r') as zip_ref:
            zip_ref.extractall(output_path)
    
    # Move files to expected structure
    extracted_dir = output_path / "voxconverse-master"
    if extracted_dir.exists():
        for item in extracted_dir.iterdir():
            item.rename(output_path / item.name)
        extracted_dir.rmdir()
    
    logger.info("VoxConverse annotations downloaded")
    logger.info("Note: Audio files must be downloaded separately from YouTube")
    
    return output_path
