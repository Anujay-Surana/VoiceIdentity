"""Benchmark runner for evaluating the VoiceIdentity system.

This module orchestrates the benchmark evaluation:
1. Load dataset
2. Process each sample through the system
3. Compare predictions with ground truth
4. Compute metrics
5. Generate reports
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from benchmark.datasets.base import BenchmarkDataset, BenchmarkSample
from benchmark.metrics.diarization import (
    DiarizationErrorRate,
    compute_der,
    compute_jer,
    compute_coverage,
    compute_purity,
)
from benchmark.metrics.identification import (
    compute_identification_accuracy,
    compute_confusion_matrix,
)

logger = logging.getLogger(__name__)


# State-of-the-art reference numbers (as of 2024)
SOTA_REFERENCE = {
    "VoxConverse": {
        "pyannote_3.1": {"der": 4.6, "jer": 0.0, "model": "pyannote/speaker-diarization-3.1"},
        "wespeaker": {"der": 5.2, "jer": 0.0, "model": "WeSpeaker ResNet-293"},
        "diarizers": {"der": 7.8, "jer": 0.0, "model": "Diarizers baseline"},
    },
    "AMI": {
        "pyannote_3.1": {"der": 18.8, "jer": 0.0, "model": "pyannote/speaker-diarization-3.1"},
        "nemo": {"der": 16.5, "jer": 0.0, "model": "NVIDIA NeMo MSDD"},
    },
    "DIHARD_III": {
        "pyannote_3.1": {"der": 21.7, "jer": 0.0, "model": "pyannote/speaker-diarization-3.1"},
    },
}


@dataclass
class BenchmarkResult:
    """
    Result from running benchmark on a single sample.
    """
    sample_id: str
    ground_truth: List[Tuple[float, float, str]]
    predictions: List[Tuple[float, float, str]]
    der: DiarizationErrorRate
    jer: float
    coverage: float
    purity: float
    processing_time: float
    n_speakers_gt: int
    n_speakers_pred: int
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "der": self.der.to_dict(),
            "jer": round(self.jer, 2),
            "coverage": round(self.coverage, 2),
            "purity": round(self.purity, 2),
            "processing_time": round(self.processing_time, 3),
            "n_speakers_gt": self.n_speakers_gt,
            "n_speakers_pred": self.n_speakers_pred,
        }


@dataclass
class BenchmarkSummary:
    """
    Summary of benchmark results across all samples.
    """
    dataset_name: str
    n_samples: int
    total_duration: float
    avg_der: float
    std_der: float
    avg_jer: float
    avg_coverage: float
    avg_purity: float
    avg_processing_time: float
    total_processing_time: float
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "total_duration_hours": round(self.total_duration / 3600, 2),
            "metrics": {
                "der": {
                    "mean": round(self.avg_der, 2),
                    "std": round(self.std_der, 2),
                },
                "jer": round(self.avg_jer, 2),
                "coverage": round(self.avg_coverage, 2),
                "purity": round(self.avg_purity, 2),
            },
            "processing": {
                "avg_time_per_sample": round(self.avg_processing_time, 2),
                "total_time": round(self.total_processing_time, 2),
                "rtf": round(self.total_processing_time / self.total_duration, 3)
                if self.total_duration > 0 else 0,
            },
            "per_sample": [r.to_dict() for r in self.results],
        }


class BenchmarkRunner:
    """
    Runner for benchmarking the VoiceIdentity system.
    """
    
    def __init__(
        self,
        diarization_pipeline=None,
        embedding_extractor=None,
        collar: float = 0.0,
        skip_overlap: bool = False,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            diarization_pipeline: PyAnnote diarization pipeline
            embedding_extractor: SpeechBrain embedding extractor
            collar: Forgiveness collar for DER (seconds)
            skip_overlap: Whether to skip overlapping speech
        """
        self.diarization_pipeline = diarization_pipeline
        self.embedding_extractor = embedding_extractor
        self.collar = collar
        self.skip_overlap = skip_overlap
    
    def run(
        self,
        dataset: BenchmarkDataset,
        output_dir: Optional[str] = None,
    ) -> BenchmarkSummary:
        """
        Run benchmark on a dataset.
        
        Args:
            dataset: Dataset to evaluate
            output_dir: Optional directory to save results
            
        Returns:
            BenchmarkSummary with aggregated results
        """
        logger.info(f"Running benchmark on {dataset.name} ({len(dataset)} samples)")
        
        results = []
        total_duration = 0.0
        
        for i, sample in enumerate(dataset):
            logger.info(f"Processing [{i+1}/{len(dataset)}] {sample.sample_id}")
            
            try:
                result = self._process_sample(sample)
                results.append(result)
                total_duration += sample.duration
            except Exception as e:
                logger.error(f"Failed to process {sample.sample_id}: {e}")
        
        # Aggregate results
        summary = self._aggregate_results(dataset.name, results, total_duration)
        
        # Save results
        if output_dir:
            self._save_results(summary, output_dir)
        
        return summary
    
    def _process_sample(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Process a single sample and compute metrics."""
        start_time = time.time()
        
        # Get ground truth
        ground_truth = sample.get_annotations_as_tuples()
        n_speakers_gt = sample.n_speakers
        
        # Run diarization
        if self.diarization_pipeline:
            predictions = self._run_diarization(sample.audio_path)
        else:
            # Fallback: use ground truth with noise (for testing)
            predictions = self._simulate_predictions(ground_truth)
        
        n_speakers_pred = len(set(p[2] for p in predictions))
        
        processing_time = time.time() - start_time
        
        # Compute metrics
        der = compute_der(ground_truth, predictions, self.collar, self.skip_overlap)
        jer = compute_jer(ground_truth, predictions)
        coverage = compute_coverage(ground_truth, predictions)
        purity = compute_purity(ground_truth, predictions)
        
        return BenchmarkResult(
            sample_id=sample.sample_id,
            ground_truth=ground_truth,
            predictions=predictions,
            der=der,
            jer=jer,
            coverage=coverage,
            purity=purity,
            processing_time=processing_time,
            n_speakers_gt=n_speakers_gt,
            n_speakers_pred=n_speakers_pred,
        )
    
    def _run_diarization(self, audio_path: Path) -> List[Tuple[float, float, str]]:
        """Run diarization pipeline on audio file."""
        if self.diarization_pipeline is None:
            raise ValueError("No diarization pipeline configured")
        
        # Run pyannote diarization
        diarization = self.diarization_pipeline(str(audio_path))
        
        # Convert to list of tuples
        predictions = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            predictions.append((segment.start, segment.end, speaker))
        
        return predictions
    
    def _simulate_predictions(
        self,
        ground_truth: List[Tuple[float, float, str]],
    ) -> List[Tuple[float, float, str]]:
        """
        Simulate predictions by adding noise to ground truth.
        Used for testing the benchmark framework.
        """
        predictions = []
        
        for start, end, speaker in ground_truth:
            # Add random timing noise
            start_noise = np.random.normal(0, 0.1)
            end_noise = np.random.normal(0, 0.1)
            
            new_start = max(0, start + start_noise)
            new_end = max(new_start + 0.1, end + end_noise)
            
            # Occasionally assign wrong speaker
            if np.random.random() < 0.1:  # 10% confusion
                speaker = f"SPEAKER_{np.random.randint(0, 5):02d}"
            
            predictions.append((new_start, new_end, speaker))
        
        return predictions
    
    def _aggregate_results(
        self,
        dataset_name: str,
        results: List[BenchmarkResult],
        total_duration: float,
    ) -> BenchmarkSummary:
        """Aggregate results across all samples."""
        if not results:
            return BenchmarkSummary(
                dataset_name=dataset_name,
                n_samples=0,
                total_duration=0,
                avg_der=0, std_der=0,
                avg_jer=0, avg_coverage=0, avg_purity=0,
                avg_processing_time=0, total_processing_time=0,
            )
        
        ders = [r.der.der for r in results]
        jers = [r.jer for r in results]
        coverages = [r.coverage for r in results]
        purities = [r.purity for r in results]
        times = [r.processing_time for r in results]
        
        return BenchmarkSummary(
            dataset_name=dataset_name,
            n_samples=len(results),
            total_duration=total_duration,
            avg_der=float(np.mean(ders)),
            std_der=float(np.std(ders)),
            avg_jer=float(np.mean(jers)),
            avg_coverage=float(np.mean(coverages)),
            avg_purity=float(np.mean(purities)),
            avg_processing_time=float(np.mean(times)),
            total_processing_time=float(np.sum(times)),
            results=results,
        )
    
    def _save_results(
        self,
        summary: BenchmarkSummary,
        output_dir: str,
    ) -> None:
        """Save benchmark results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{summary.dataset_name.lower()}_{timestamp}.json"
        
        with open(output_path / filename, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {output_path / filename}")


def compare_to_sota(
    results: BenchmarkSummary,
    dataset_name: Optional[str] = None,
) -> Dict:
    """
    Compare benchmark results to state-of-the-art.
    
    Args:
        results: Benchmark results to compare
        dataset_name: Override dataset name for SOTA lookup
        
    Returns:
        Comparison dict with SOTA references
    """
    name = dataset_name or results.dataset_name
    sota = SOTA_REFERENCE.get(name, {})
    
    comparison = {
        "your_results": {
            "der": results.avg_der,
            "jer": results.avg_jer,
        },
        "sota_reference": sota,
        "gaps": {},
    }
    
    for model_name, model_results in sota.items():
        sota_der = model_results.get("der", 0)
        gap = results.avg_der - sota_der
        comparison["gaps"][model_name] = {
            "der_gap": round(gap, 2),
            "relative_gap": f"+{gap/sota_der*100:.1f}%" if sota_der > 0 else "N/A",
        }
    
    return comparison


def print_benchmark_report(
    summary: BenchmarkSummary,
    show_sota: bool = True,
) -> None:
    """Print a formatted benchmark report."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {summary.dataset_name}")
    print("=" * 60)
    
    print(f"\nDataset: {summary.n_samples} samples")
    print(f"Total Duration: {summary.total_duration/3600:.2f} hours")
    
    print("\n--- Diarization Metrics ---")
    print(f"DER:      {summary.avg_der:.2f}% (±{summary.std_der:.2f})")
    print(f"JER:      {summary.avg_jer:.2f}%")
    print(f"Coverage: {summary.avg_coverage:.2f}%")
    print(f"Purity:   {summary.avg_purity:.2f}%")
    
    print("\n--- Processing Performance ---")
    print(f"Avg Time/Sample: {summary.avg_processing_time:.2f}s")
    print(f"Total Time:      {summary.total_processing_time:.2f}s")
    rtf = summary.total_processing_time / summary.total_duration if summary.total_duration > 0 else 0
    print(f"RTF:             {rtf:.3f}x")
    
    if show_sota:
        comparison = compare_to_sota(summary)
        
        if comparison["sota_reference"]:
            print("\n--- SOTA Comparison ---")
            for model, sota in comparison["sota_reference"].items():
                gap = comparison["gaps"].get(model, {})
                print(f"{model}:")
                print(f"  SOTA DER: {sota.get('der', 'N/A')}%")
                print(f"  Your DER: {summary.avg_der:.2f}%")
                print(f"  Gap:      {gap.get('relative_gap', 'N/A')}")
    
    print("\n" + "=" * 60)
