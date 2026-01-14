"""Benchmark suite for speaker diarization and identification.

This module provides tools to evaluate the VoiceIdentity system against
standard academic benchmarks and compare with state-of-the-art results.

Metrics:
- DER (Diarization Error Rate): Standard metric for "who spoke when"
- JER (Jaccard Error Rate): Per-speaker error rate
- Speaker Identification Accuracy: How often correct speaker is identified
- EER (Equal Error Rate): Speaker verification metric

Datasets:
- VoxConverse: Multi-speaker conversations
- AMI Meeting Corpus: Meeting recordings
- Custom: User-provided audio with annotations

Usage:
    from benchmark import run_benchmark, compare_to_sota
    
    results = run_benchmark("voxconverse", audio_dir="/path/to/audio")
    compare_to_sota(results)
"""

__version__ = "0.1.0"
