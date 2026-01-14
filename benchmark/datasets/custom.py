"""Custom dataset loader for user-provided audio and annotations.

Supports flexible annotation formats:
- RTTM (Rich Transcription Time Marked)
- JSON (custom format)
- Simple text (start end speaker per line)
"""

import json
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


class CustomDataset(BenchmarkDataset):
    """
    Custom dataset loader for user-provided data.
    
    Expected directory structure:
    root_dir/
        audio/
            sample1.wav
            sample2.wav
            ...
        annotations/
            sample1.rttm (or .json or .txt)
            sample2.rttm
            ...
    
    Or flat structure:
    root_dir/
        sample1.wav
        sample1.rttm
        sample2.wav
        sample2.rttm
        ...
    
    JSON annotation format:
    {
        "segments": [
            {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_01", "text": "Hello"},
            ...
        ]
    }
    
    Simple text format:
    start end speaker [text]
    0.0 1.5 SPEAKER_01 Hello
    """
    
    @property
    def name(self) -> str:
        return "Custom"
    
    def _load_samples(self) -> None:
        """Load custom dataset samples."""
        # Find audio directory
        audio_dir = self._find_audio_dir()
        if audio_dir is None:
            logger.warning(f"No audio directory found in {self.root_dir}")
            return
        
        # Find annotation directory
        ann_dir = self._find_annotation_dir()
        if ann_dir is None:
            logger.warning(f"No annotation directory found in {self.root_dir}")
            return
        
        logger.info(f"Loading custom dataset from {self.root_dir}")
        logger.info(f"  Audio dir: {audio_dir}")
        logger.info(f"  Annotation dir: {ann_dir}")
        
        # Find all audio files
        audio_files = self._find_audio_files(audio_dir)
        
        # Load samples
        count = 0
        for sample_id, audio_path in sorted(audio_files.items()):
            if self.max_samples and count >= self.max_samples:
                break
            
            # Find matching annotation file
            ann_path = self._find_annotation_file(ann_dir, sample_id)
            
            if ann_path is None:
                logger.warning(f"No annotation found for {sample_id}")
                continue
            
            # Parse annotations based on format
            annotations = self._parse_annotations(ann_path)
            
            if not annotations:
                logger.warning(f"No annotations parsed from {ann_path}")
                continue
            
            # Calculate duration
            duration = max(ann.end for ann in annotations)
            
            sample = BenchmarkSample(
                audio_path=audio_path,
                annotations=annotations,
                sample_id=sample_id,
                duration=duration,
                metadata={
                    "annotation_file": str(ann_path),
                    "dataset": "custom",
                },
            )
            
            self.samples.append(sample)
            count += 1
        
        logger.info(f"Loaded {len(self.samples)} custom samples")
    
    def _find_audio_dir(self) -> Optional[Path]:
        """Find directory containing audio files."""
        candidates = [
            self.root_dir / "audio",
            self.root_dir / "wav",
            self.root_dir,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                # Check if it has audio files
                extensions = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]
                for ext in extensions:
                    if list(candidate.glob(f"*{ext}")):
                        return candidate
        
        return None
    
    def _find_annotation_dir(self) -> Optional[Path]:
        """Find directory containing annotation files."""
        candidates = [
            self.root_dir / "annotations",
            self.root_dir / "rttm",
            self.root_dir / "labels",
            self.root_dir,
        ]
        
        for candidate in candidates:
            if candidate.exists():
                # Check if it has annotation files
                extensions = [".rttm", ".json", ".txt"]
                for ext in extensions:
                    if list(candidate.glob(f"*{ext}")):
                        return candidate
        
        return None
    
    def _find_audio_files(self, audio_dir: Path) -> Dict[str, Path]:
        """Find all audio files and map to sample IDs."""
        audio_files = {}
        extensions = [".wav", ".flac", ".mp3", ".m4a", ".ogg"]
        
        for ext in extensions:
            for audio_path in audio_dir.glob(f"*{ext}"):
                sample_id = audio_path.stem
                if sample_id not in audio_files:
                    audio_files[sample_id] = audio_path
        
        return audio_files
    
    def _find_annotation_file(
        self,
        ann_dir: Path,
        sample_id: str,
    ) -> Optional[Path]:
        """Find annotation file for a sample."""
        extensions = [".rttm", ".json", ".txt"]
        
        for ext in extensions:
            ann_path = ann_dir / f"{sample_id}{ext}"
            if ann_path.exists():
                return ann_path
        
        return None
    
    def _parse_annotations(self, ann_path: Path) -> List[Annotation]:
        """Parse annotations from file based on format."""
        ext = ann_path.suffix.lower()
        
        if ext == ".rttm":
            return parse_rttm(ann_path)
        elif ext == ".json":
            return self._parse_json_annotations(ann_path)
        elif ext == ".txt":
            return self._parse_text_annotations(ann_path)
        else:
            logger.warning(f"Unknown annotation format: {ext}")
            return []
    
    def _parse_json_annotations(self, json_path: Path) -> List[Annotation]:
        """Parse JSON annotation file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        annotations = []
        
        # Support multiple JSON formats
        segments = data.get("segments", data.get("annotations", []))
        
        for seg in segments:
            try:
                start = float(seg.get("start", seg.get("start_time", 0)))
                end = float(seg.get("end", seg.get("end_time", 0)))
                speaker = seg.get("speaker", seg.get("speaker_id", "unknown"))
                text = seg.get("text", seg.get("transcript"))
                
                annotations.append(Annotation(
                    start=start,
                    end=end,
                    speaker_id=speaker,
                    text=text,
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse JSON segment: {seg}: {e}")
        
        return annotations
    
    def _parse_text_annotations(self, txt_path: Path) -> List[Annotation]:
        """Parse simple text annotation file."""
        annotations = []
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts) < 3:
                    continue
                
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    speaker = parts[2]
                    text = " ".join(parts[3:]) if len(parts) > 3 else None
                    
                    annotations.append(Annotation(
                        start=start,
                        end=end,
                        speaker_id=speaker,
                        text=text,
                    ))
                except ValueError as e:
                    logger.warning(f"Failed to parse line: {line.strip()}: {e}")
        
        return annotations


def create_custom_dataset(
    output_dir: str,
    audio_files: List[str],
    create_templates: bool = True,
) -> Path:
    """
    Create a custom dataset structure with template annotations.
    
    Args:
        output_dir: Directory to create dataset in
        audio_files: List of paths to audio files to include
        create_templates: Whether to create template annotation files
        
    Returns:
        Path to created dataset
    """
    import shutil
    
    output_path = Path(output_dir)
    audio_dir = output_path / "audio"
    ann_dir = output_path / "annotations"
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    for audio_file in audio_files:
        audio_path = Path(audio_file)
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_file}")
            continue
        
        # Copy audio file
        dest_audio = audio_dir / audio_path.name
        shutil.copy2(audio_path, dest_audio)
        
        # Create template annotation
        if create_templates:
            sample_id = audio_path.stem
            ann_path = ann_dir / f"{sample_id}.json"
            
            template = {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "SPEAKER_01",
                        "text": "Replace with actual annotation"
                    }
                ]
            }
            
            with open(ann_path, 'w') as f:
                json.dump(template, f, indent=2)
    
    logger.info(f"Created custom dataset at {output_path}")
    logger.info(f"  Audio: {audio_dir}")
    logger.info(f"  Annotations: {ann_dir} (templates created)")
    
    return output_path
