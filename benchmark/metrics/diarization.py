"""Diarization metrics using pyannote.metrics.

Standard metrics for evaluating speaker diarization:
- DER (Diarization Error Rate): Primary metric
- JER (Jaccard Error Rate): Per-speaker metric
- Coverage: Percentage of reference covered by hypothesis
- Purity: How "pure" each cluster is
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiarizationErrorRate:
    """
    Diarization Error Rate breakdown.
    
    DER = (False Alarm + Missed Speech + Speaker Confusion) / Total Reference Duration
    
    Attributes:
        false_alarm: Time marked as speech in hypothesis but not in reference
        missed_speech: Time marked as speech in reference but not in hypothesis  
        confusion: Time where both have speech but wrong speaker assigned
        total_reference: Total speech duration in reference
        der: Overall DER percentage
    """
    false_alarm: float = 0.0
    missed_speech: float = 0.0
    confusion: float = 0.0
    total_reference: float = 0.0
    der: float = 0.0
    jer: float = 0.0  # Jaccard Error Rate
    
    def __post_init__(self):
        """Calculate DER if not provided."""
        if self.total_reference > 0 and self.der == 0:
            self.der = (self.false_alarm + self.missed_speech + self.confusion) / self.total_reference * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "der": round(self.der, 2),
            "jer": round(self.jer, 2),
            "false_alarm": round(self.false_alarm, 3),
            "missed_speech": round(self.missed_speech, 3),
            "confusion": round(self.confusion, 3),
            "total_reference": round(self.total_reference, 3),
        }


def compute_der(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> DiarizationErrorRate:
    """
    Compute Diarization Error Rate.
    
    Args:
        reference: List of (start, end, speaker_id) tuples (ground truth)
        hypothesis: List of (start, end, speaker_id) tuples (system output)
        collar: Forgiveness collar around reference boundaries (seconds)
        skip_overlap: Whether to skip overlapping speech regions
        
    Returns:
        DiarizationErrorRate object with detailed breakdown
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate as PyaDER
        
        # Convert to pyannote Annotation format
        ref_annotation = Annotation()
        for start, end, speaker in reference:
            ref_annotation[Segment(start, end)] = speaker
        
        hyp_annotation = Annotation()
        for start, end, speaker in hypothesis:
            hyp_annotation[Segment(start, end)] = speaker
        
        # Create metric with options
        metric = PyaDER(collar=collar, skip_overlap=skip_overlap)
        
        # Compute DER
        der_value = metric(ref_annotation, hyp_annotation)
        
        # Get detailed breakdown
        details = metric.report().iloc[0]
        
        return DiarizationErrorRate(
            false_alarm=float(details.get("false alarm", 0)),
            missed_speech=float(details.get("missed detection", 0)),
            confusion=float(details.get("confusion", 0)),
            total_reference=float(details.get("total", 1)),
            der=der_value * 100,
        )
        
    except ImportError:
        logger.warning("pyannote.metrics not available, using fallback implementation")
        return _compute_der_fallback(reference, hypothesis, collar)


def _compute_der_fallback(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
    collar: float = 0.0,
) -> DiarizationErrorRate:
    """
    Fallback DER computation without pyannote.metrics.
    
    Uses a simpler frame-based approach.
    """
    # Frame resolution (10ms)
    frame_size = 0.01
    
    # Find total duration
    if not reference:
        return DiarizationErrorRate()
    
    max_time = max(max(end for _, end, _ in reference),
                   max(end for _, end, _ in hypothesis) if hypothesis else 0)
    
    n_frames = int(np.ceil(max_time / frame_size))
    
    # Create frame-level labels
    ref_frames = [""] * n_frames
    hyp_frames = [""] * n_frames
    
    for start, end, speaker in reference:
        start_frame = int(start / frame_size)
        end_frame = int(end / frame_size)
        for i in range(start_frame, min(end_frame, n_frames)):
            ref_frames[i] = speaker
    
    for start, end, speaker in hypothesis:
        start_frame = int(start / frame_size)
        end_frame = int(end / frame_size)
        for i in range(start_frame, min(end_frame, n_frames)):
            hyp_frames[i] = speaker
    
    # Count errors
    fa = 0  # False alarm
    ms = 0  # Missed speech
    cf = 0  # Confusion
    total_ref = 0
    
    for ref, hyp in zip(ref_frames, hyp_frames):
        if ref:  # Reference has speech
            total_ref += 1
            if not hyp:
                ms += 1  # Missed
            elif ref != hyp:
                cf += 1  # Wrong speaker
        else:  # Reference has no speech
            if hyp:
                fa += 1  # False alarm
    
    # Convert to seconds
    fa_sec = fa * frame_size
    ms_sec = ms * frame_size
    cf_sec = cf * frame_size
    total_sec = total_ref * frame_size
    
    der = (fa_sec + ms_sec + cf_sec) / total_sec * 100 if total_sec > 0 else 0
    
    return DiarizationErrorRate(
        false_alarm=fa_sec,
        missed_speech=ms_sec,
        confusion=cf_sec,
        total_reference=total_sec,
        der=der,
    )


def compute_jer(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
) -> float:
    """
    Compute Jaccard Error Rate (JER).
    
    JER is a per-speaker metric that measures the overlap between
    reference and hypothesis speaker segments.
    
    Args:
        reference: List of (start, end, speaker_id) tuples
        hypothesis: List of (start, end, speaker_id) tuples
        
    Returns:
        JER as a percentage (lower is better)
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import JaccardErrorRate
        
        ref_annotation = Annotation()
        for start, end, speaker in reference:
            ref_annotation[Segment(start, end)] = speaker
        
        hyp_annotation = Annotation()
        for start, end, speaker in hypothesis:
            hyp_annotation[Segment(start, end)] = speaker
        
        metric = JaccardErrorRate()
        return float(metric(ref_annotation, hyp_annotation) * 100)
        
    except ImportError:
        logger.warning("pyannote.metrics not available for JER")
        return 0.0


def compute_coverage(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
) -> float:
    """
    Compute coverage: what fraction of reference is covered by hypothesis.
    
    Args:
        reference: Ground truth segments
        hypothesis: System output segments
        
    Returns:
        Coverage percentage (0-100)
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.segmentation import SegmentationCoverage
        
        ref_annotation = Annotation()
        for start, end, speaker in reference:
            ref_annotation[Segment(start, end)] = speaker
        
        hyp_annotation = Annotation()
        for start, end, speaker in hypothesis:
            hyp_annotation[Segment(start, end)] = speaker
        
        metric = SegmentationCoverage()
        return float(metric(ref_annotation, hyp_annotation) * 100)
        
    except ImportError:
        # Simple fallback
        if not reference:
            return 0.0
        
        ref_duration = sum(end - start for start, end, _ in reference)
        if ref_duration == 0:
            return 0.0
        
        # Simple overlap calculation
        covered = 0.0
        for r_start, r_end, _ in reference:
            for h_start, h_end, _ in hypothesis:
                overlap_start = max(r_start, h_start)
                overlap_end = min(r_end, h_end)
                if overlap_end > overlap_start:
                    covered += overlap_end - overlap_start
        
        return min(100.0, covered / ref_duration * 100)


def compute_purity(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
) -> float:
    """
    Compute purity: how "pure" each hypothesis cluster is.
    
    A cluster is pure if all its segments belong to the same reference speaker.
    
    Args:
        reference: Ground truth segments
        hypothesis: System output segments
        
    Returns:
        Purity percentage (0-100)
    """
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.segmentation import SegmentationPurity
        
        ref_annotation = Annotation()
        for start, end, speaker in reference:
            ref_annotation[Segment(start, end)] = speaker
        
        hyp_annotation = Annotation()
        for start, end, speaker in hypothesis:
            hyp_annotation[Segment(start, end)] = speaker
        
        metric = SegmentationPurity()
        return float(metric(ref_annotation, hyp_annotation) * 100)
        
    except ImportError:
        logger.warning("pyannote.metrics not available for purity")
        return 0.0
