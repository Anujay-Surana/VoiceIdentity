"""Speaker identification and verification metrics.

Metrics for evaluating how well the system identifies known speakers:
- Identification Accuracy: How often the correct speaker is identified
- EER (Equal Error Rate): Verification threshold where FAR = FRR
- Confusion Matrix: Per-speaker identification breakdown
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """
    Speaker identification evaluation result.
    
    Attributes:
        accuracy: Overall identification accuracy (%)
        top_k_accuracy: Accuracy when correct speaker is in top K predictions
        per_speaker_accuracy: Accuracy breakdown per speaker
        total_correct: Number of correct identifications
        total_samples: Total number of samples evaluated
    """
    accuracy: float = 0.0
    top_k_accuracy: Dict[int, float] = None
    per_speaker_accuracy: Dict[str, float] = None
    total_correct: int = 0
    total_samples: int = 0
    
    def __post_init__(self):
        if self.top_k_accuracy is None:
            self.top_k_accuracy = {}
        if self.per_speaker_accuracy is None:
            self.per_speaker_accuracy = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": round(self.accuracy, 2),
            "top_k_accuracy": {k: round(v, 2) for k, v in self.top_k_accuracy.items()},
            "per_speaker_accuracy": {k: round(v, 2) for k, v in self.per_speaker_accuracy.items()},
            "total_correct": self.total_correct,
            "total_samples": self.total_samples,
        }


def compute_identification_accuracy(
    predictions: List[Dict],
    references: List[str],
    top_k: List[int] = [1, 3, 5],
) -> IdentificationResult:
    """
    Compute speaker identification accuracy.
    
    Args:
        predictions: List of prediction dicts with 'speaker_id' and optionally
                    'candidates' (ranked list of (speaker_id, score) tuples)
        references: List of ground truth speaker IDs
        top_k: List of K values for top-K accuracy
        
    Returns:
        IdentificationResult with accuracy metrics
    """
    if not predictions or not references:
        return IdentificationResult()
    
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length")
    
    # Count correct predictions
    correct = 0
    top_k_correct = defaultdict(int)
    per_speaker_correct = defaultdict(int)
    per_speaker_total = defaultdict(int)
    
    for pred, ref in zip(predictions, references):
        per_speaker_total[ref] += 1
        
        # Top-1 accuracy
        pred_speaker = pred.get("speaker_id")
        if pred_speaker == ref:
            correct += 1
            per_speaker_correct[ref] += 1
        
        # Top-K accuracy
        candidates = pred.get("candidates", [])
        if candidates:
            for k in top_k:
                top_k_speakers = [c[0] if isinstance(c, tuple) else c.get("speaker_id") 
                                for c in candidates[:k]]
                if ref in top_k_speakers:
                    top_k_correct[k] += 1
        else:
            # If no candidates, use top-1 result for all K
            for k in top_k:
                if pred_speaker == ref:
                    top_k_correct[k] += 1
    
    total = len(predictions)
    
    # Calculate accuracies
    accuracy = correct / total * 100 if total > 0 else 0
    
    top_k_accuracy = {k: top_k_correct[k] / total * 100 for k in top_k}
    
    per_speaker_accuracy = {
        speaker: per_speaker_correct[speaker] / per_speaker_total[speaker] * 100
        for speaker in per_speaker_total
    }
    
    return IdentificationResult(
        accuracy=accuracy,
        top_k_accuracy=top_k_accuracy,
        per_speaker_accuracy=per_speaker_accuracy,
        total_correct=correct,
        total_samples=total,
    )


def compute_eer(
    scores: List[float],
    labels: List[bool],
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) for speaker verification.
    
    EER is the point where False Acceptance Rate (FAR) equals 
    False Rejection Rate (FRR).
    
    Args:
        scores: List of similarity scores
        labels: List of boolean labels (True = same speaker)
        
    Returns:
        Tuple of (EER percentage, threshold at EER)
    """
    if not scores or not labels:
        return 0.0, 0.0
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Sort by score
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    # Count positives and negatives
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0
    
    # Calculate FAR and FRR at each threshold
    far_list = []
    frr_list = []
    thresholds = []
    
    for i, threshold in enumerate(sorted_scores):
        # FAR: fraction of negatives with score >= threshold
        far = np.sum((~sorted_labels) & (sorted_scores >= threshold)) / n_neg
        # FRR: fraction of positives with score < threshold
        frr = np.sum(sorted_labels & (sorted_scores < threshold)) / n_pos
        
        far_list.append(far)
        frr_list.append(frr)
        thresholds.append(threshold)
    
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)
    thresholds = np.array(thresholds)
    
    # Find where FAR and FRR cross
    diff = np.abs(far_list - frr_list)
    eer_idx = np.argmin(diff)
    
    eer = (far_list[eer_idx] + frr_list[eer_idx]) / 2 * 100
    eer_threshold = thresholds[eer_idx]
    
    return float(eer), float(eer_threshold)


def compute_confusion_matrix(
    predictions: List[str],
    references: List[str],
    speaker_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute confusion matrix for speaker identification.
    
    Args:
        predictions: List of predicted speaker IDs
        references: List of ground truth speaker IDs
        speaker_names: Optional list of speaker names (for labeling)
        
    Returns:
        Dict with confusion matrix and summary statistics
    """
    if not predictions or not references:
        return {"matrix": [], "speakers": [], "accuracy_per_speaker": {}}
    
    # Get unique speakers
    all_speakers = sorted(set(references) | set(predictions))
    speaker_to_idx = {s: i for i, s in enumerate(all_speakers)}
    n_speakers = len(all_speakers)
    
    # Build confusion matrix
    matrix = np.zeros((n_speakers, n_speakers), dtype=int)
    
    for pred, ref in zip(predictions, references):
        if ref in speaker_to_idx and pred in speaker_to_idx:
            matrix[speaker_to_idx[ref], speaker_to_idx[pred]] += 1
    
    # Calculate per-speaker accuracy
    accuracy_per_speaker = {}
    for speaker in all_speakers:
        idx = speaker_to_idx[speaker]
        total = matrix[idx, :].sum()
        correct = matrix[idx, idx]
        accuracy_per_speaker[speaker] = correct / total * 100 if total > 0 else 0
    
    # Use speaker names if provided
    if speaker_names:
        display_speakers = speaker_names[:n_speakers]
    else:
        display_speakers = all_speakers
    
    return {
        "matrix": matrix.tolist(),
        "speakers": display_speakers,
        "accuracy_per_speaker": {k: round(v, 2) for k, v in accuracy_per_speaker.items()},
        "overall_accuracy": round(np.trace(matrix) / matrix.sum() * 100 if matrix.sum() > 0 else 0, 2),
    }


def compute_rank_statistics(
    predictions: List[Dict],
    references: List[str],
) -> Dict:
    """
    Compute ranking statistics for speaker identification.
    
    This shows how often the correct speaker appears at each rank
    in the candidate list.
    
    Args:
        predictions: List of prediction dicts with 'candidates' field
        references: List of ground truth speaker IDs
        
    Returns:
        Dict with rank statistics
    """
    if not predictions or not references:
        return {"mean_rank": 0, "median_rank": 0, "rank_histogram": {}}
    
    ranks = []
    
    for pred, ref in zip(predictions, references):
        candidates = pred.get("candidates", [])
        
        if not candidates:
            continue
        
        # Find rank of correct speaker
        for rank, candidate in enumerate(candidates, start=1):
            speaker_id = candidate[0] if isinstance(candidate, tuple) else candidate.get("speaker_id")
            if speaker_id == ref:
                ranks.append(rank)
                break
        else:
            # Not found in candidates
            ranks.append(len(candidates) + 1)
    
    if not ranks:
        return {"mean_rank": 0, "median_rank": 0, "rank_histogram": {}}
    
    ranks = np.array(ranks)
    
    # Compute histogram
    max_rank = min(10, max(ranks))
    histogram = {i: int(np.sum(ranks == i)) for i in range(1, max_rank + 1)}
    histogram["not_found"] = int(np.sum(ranks > max_rank))
    
    return {
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "min_rank": int(np.min(ranks)),
        "max_rank": int(np.max(ranks)),
        "rank_histogram": histogram,
    }
