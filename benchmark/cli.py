"""Command-line interface for running benchmarks.

Usage:
    python -m benchmark.cli run voxconverse --data-dir /path/to/voxconverse
    python -m benchmark.cli run custom --data-dir /path/to/custom
    python -m benchmark.cli compare results.json
    python -m benchmark.cli report results.json --format markdown
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import click

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool):
    """VoiceIdentity Benchmark Suite"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("dataset", type=click.Choice(["voxconverse", "ami", "custom"]))
@click.option("--data-dir", "-d", required=True, help="Path to dataset directory")
@click.option("--output-dir", "-o", default="benchmark/results", help="Output directory")
@click.option("--split", "-s", default="test", help="Dataset split (train/dev/test)")
@click.option("--max-samples", "-n", type=int, default=None, help="Max samples to process")
@click.option("--collar", "-c", type=float, default=0.0, help="Collar for DER (seconds)")
@click.option("--use-gpu", is_flag=True, help="Use GPU if available")
@click.option("--no-pipeline", is_flag=True, help="Skip diarization (for testing)")
def run(
    dataset: str,
    data_dir: str,
    output_dir: str,
    split: str,
    max_samples: Optional[int],
    collar: float,
    use_gpu: bool,
    no_pipeline: bool,
):
    """Run benchmark evaluation on a dataset."""
    from benchmark.runner import BenchmarkRunner, print_benchmark_report
    
    # Load dataset
    logger.info(f"Loading {dataset} dataset from {data_dir}")
    
    if dataset == "voxconverse":
        from benchmark.datasets.voxconverse import VoxConverseDataset
        ds = VoxConverseDataset(data_dir, split=split, max_samples=max_samples)
    elif dataset == "ami":
        from benchmark.datasets.ami import AMIDataset
        ds = AMIDataset(data_dir, split=split, max_samples=max_samples)
    elif dataset == "custom":
        from benchmark.datasets.custom import CustomDataset
        ds = CustomDataset(data_dir, split=split, max_samples=max_samples)
    else:
        raise click.BadParameter(f"Unknown dataset: {dataset}")
    
    if len(ds) == 0:
        logger.error("No samples loaded from dataset")
        sys.exit(1)
    
    logger.info(f"Loaded {len(ds)} samples")
    
    # Load diarization pipeline
    diarization_pipeline = None
    embedding_extractor = None
    
    if not no_pipeline:
        try:
            logger.info("Loading diarization pipeline...")
            from pyannote.audio import Pipeline
            import torch
            
            # Load pipeline from HuggingFace
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("HF_TOKEN not set, using cached model")
            
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            
            if use_gpu and torch.cuda.is_available():
                diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
                logger.info("Pipeline loaded on GPU")
            else:
                logger.info("Pipeline loaded on CPU")
                
        except Exception as e:
            logger.warning(f"Could not load pipeline: {e}")
            logger.info("Running in simulation mode (no actual diarization)")
    
    # Run benchmark
    runner = BenchmarkRunner(
        diarization_pipeline=diarization_pipeline,
        embedding_extractor=embedding_extractor,
        collar=collar,
    )
    
    results = runner.run(ds, output_dir)
    
    # Print report
    print_benchmark_report(results, show_sota=True)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
def compare(results_file: str):
    """Compare results to SOTA reference."""
    from benchmark.runner import compare_to_sota, BenchmarkSummary
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Reconstruct summary from JSON
    summary = BenchmarkSummary(
        dataset_name=data.get("dataset_name", "Unknown"),
        n_samples=data.get("n_samples", 0),
        total_duration=data.get("total_duration_hours", 0) * 3600,
        avg_der=data.get("metrics", {}).get("der", {}).get("mean", 0),
        std_der=data.get("metrics", {}).get("der", {}).get("std", 0),
        avg_jer=data.get("metrics", {}).get("jer", 0),
        avg_coverage=data.get("metrics", {}).get("coverage", 0),
        avg_purity=data.get("metrics", {}).get("purity", 0),
        avg_processing_time=data.get("processing", {}).get("avg_time_per_sample", 0),
        total_processing_time=data.get("processing", {}).get("total_time", 0),
    )
    
    comparison = compare_to_sota(summary)
    
    print("\n=== SOTA Comparison ===")
    print(f"\nYour Results:")
    print(f"  DER: {comparison['your_results']['der']:.2f}%")
    print(f"  JER: {comparison['your_results']['jer']:.2f}%")
    
    if comparison["sota_reference"]:
        print(f"\nSOTA Reference for {summary.dataset_name}:")
        for model, sota in comparison["sota_reference"].items():
            gap = comparison["gaps"].get(model, {})
            print(f"\n  {model} ({sota.get('model', 'N/A')}):")
            print(f"    DER: {sota.get('der', 'N/A')}%")
            print(f"    Gap: {gap.get('der_gap', 'N/A')} ({gap.get('relative_gap', 'N/A')})")
    else:
        print("\nNo SOTA reference available for this dataset")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["text", "markdown", "json"]), default="text")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file")
def report(results_file: str, format: str, output: Optional[str]):
    """Generate a report from benchmark results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if format == "text":
        report_content = generate_text_report(data)
    elif format == "markdown":
        report_content = generate_markdown_report(data)
    elif format == "json":
        report_content = json.dumps(data, indent=2)
    
    if output:
        with open(output, 'w') as f:
            f.write(report_content)
        logger.info(f"Report saved to {output}")
    else:
        print(report_content)


def generate_text_report(data: dict) -> str:
    """Generate a text report from benchmark data."""
    lines = [
        "=" * 60,
        f"BENCHMARK REPORT: {data.get('dataset_name', 'Unknown')}",
        "=" * 60,
        "",
        f"Samples: {data.get('n_samples', 0)}",
        f"Duration: {data.get('total_duration_hours', 0):.2f} hours",
        "",
        "--- Metrics ---",
    ]
    
    metrics = data.get("metrics", {})
    der = metrics.get("der", {})
    lines.append(f"DER: {der.get('mean', 0):.2f}% (±{der.get('std', 0):.2f})")
    lines.append(f"JER: {metrics.get('jer', 0):.2f}%")
    lines.append(f"Coverage: {metrics.get('coverage', 0):.2f}%")
    lines.append(f"Purity: {metrics.get('purity', 0):.2f}%")
    
    lines.extend([
        "",
        "--- Processing ---",
        f"Avg Time: {data.get('processing', {}).get('avg_time_per_sample', 0):.2f}s",
        f"Total Time: {data.get('processing', {}).get('total_time', 0):.2f}s",
        f"RTF: {data.get('processing', {}).get('rtf', 0):.3f}x",
        "",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def generate_markdown_report(data: dict) -> str:
    """Generate a Markdown report from benchmark data."""
    lines = [
        f"# Benchmark Report: {data.get('dataset_name', 'Unknown')}",
        "",
        "## Summary",
        "",
        f"- **Samples**: {data.get('n_samples', 0)}",
        f"- **Duration**: {data.get('total_duration_hours', 0):.2f} hours",
        "",
        "## Metrics",
        "",
    ]
    
    metrics = data.get("metrics", {})
    der = metrics.get("der", {})
    
    lines.extend([
        "| Metric | Value |",
        "|--------|-------|",
        f"| DER | {der.get('mean', 0):.2f}% (±{der.get('std', 0):.2f}) |",
        f"| JER | {metrics.get('jer', 0):.2f}% |",
        f"| Coverage | {metrics.get('coverage', 0):.2f}% |",
        f"| Purity | {metrics.get('purity', 0):.2f}% |",
        "",
        "## Processing Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Avg Time/Sample | {data.get('processing', {}).get('avg_time_per_sample', 0):.2f}s |",
        f"| Total Time | {data.get('processing', {}).get('total_time', 0):.2f}s |",
        f"| RTF | {data.get('processing', {}).get('rtf', 0):.3f}x |",
    ])
    
    # Per-sample results if available
    per_sample = data.get("per_sample", [])
    if per_sample:
        lines.extend([
            "",
            "## Per-Sample Results",
            "",
            "| Sample | DER | Speakers (GT/Pred) | Time |",
            "|--------|-----|-------------------|------|",
        ])
        
        for sample in per_sample[:20]:  # Limit to first 20
            der_val = sample.get("der", {}).get("der", 0)
            gt = sample.get("n_speakers_gt", 0)
            pred = sample.get("n_speakers_pred", 0)
            time_val = sample.get("processing_time", 0)
            lines.append(f"| {sample.get('sample_id', 'N/A')} | {der_val:.1f}% | {gt}/{pred} | {time_val:.2f}s |")
    
    return "\n".join(lines)


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True))
def info(data_dir: str):
    """Show information about a dataset."""
    from benchmark.datasets.voxconverse import VoxConverseDataset
    from benchmark.datasets.ami import AMIDataset
    from benchmark.datasets.custom import CustomDataset
    
    # Try each dataset type
    datasets = [
        ("VoxConverse", VoxConverseDataset),
        ("AMI", AMIDataset),
        ("Custom", CustomDataset),
    ]
    
    for name, DatasetClass in datasets:
        try:
            ds = DatasetClass(data_dir, max_samples=5)
            if len(ds) > 0:
                summary = ds.get_summary()
                print(f"\nDataset: {name}")
                print(f"  Samples: {summary.get('n_samples', 0)}")
                print(f"  Duration: {summary.get('total_duration_hours', 0):.2f} hours")
                print(f"  Speakers: {summary.get('n_unique_speakers', 0)}")
                print(f"  Avg speakers/sample: {summary.get('avg_speakers_per_sample', 0):.1f}")
                return
        except Exception:
            continue
    
    print(f"Could not identify dataset format in {data_dir}")


if __name__ == "__main__":
    cli()
