#!/usr/bin/env python3
"""
HuggingFace Dataset Fetcher for Vietnamese ASR

This is a thin CLI wrapper around DatasetManager for downloading datasets.
All core logic is in src.dataset_loader.DatasetManager.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataset_loader import DatasetManager, HF_AVAILABLE


def list_available_datasets(dataset_manager: DatasetManager):
    """List all available Vietnamese datasets."""
    print("\n" + "="*70)
    print("Available Vietnamese ASR Datasets on HuggingFace")
    print("="*70 + "\n")

    if not dataset_manager.hf_configs:
        print("[WARNING] No dataset configurations found")
        return

    for key, info in dataset_manager.hf_configs.items():
        print(f"-> {key}")
        print(f"   ID: {info['id']}")
        print(f"   Description: {info['description']}")
        print(f"   Splits: {', '.join(info['splits'])}")
        print()


def fetch_dataset(
    dataset_manager: DatasetManager,
    dataset_key: str,
    max_samples: Optional[int] = None,
    save_to_disk: bool = True
):
    """
    Fetch a single dataset from HuggingFace.

    Args:
        dataset_manager: DatasetManager instance
        dataset_key: Key of the dataset to fetch
        max_samples: Maximum samples per split (for testing)
        save_to_disk: Whether to save to disk (always True for this CLI)
    """
    try:
        print(f"\n{'='*70}")
        print(f"Fetching: {dataset_key}")
        print(f"{'='*70}\n")

        # Load dataset - this will download if not cached
        samples = dataset_manager.load_dataset(
            dataset_name=dataset_key,
            respect_predefined_splits=True
        )

        # Limit samples if requested
        if max_samples:
            print(f"\n[INFO] Limiting to {max_samples} samples per split...")
            for split in samples:
                if len(samples[split]) > max_samples:
                    samples[split] = samples[split][:max_samples]
                    print(f"  {split}: limited to {max_samples} samples")

        # Print summary
        total_samples = sum(len(s) for s in samples.values())
        print(f"\n[OK] Successfully loaded {dataset_key}")
        print(f"[INFO] Total samples: {total_samples}")
        for split, split_samples in samples.items():
            print(f"  - {split}: {len(split_samples)} samples")

        return samples

    except Exception as e:
        print(f"\n[FAILED] Failed to fetch {dataset_key}: {e}")
        return None


def fetch_multiple_datasets(
    dataset_manager: DatasetManager,
    dataset_keys: List[str],
    max_samples: Optional[int] = None
):
    """
    Fetch multiple datasets.

    Args:
        dataset_manager: DatasetManager instance
        dataset_keys: List of dataset keys to fetch
        max_samples: Maximum samples per split per dataset
    """
    print(f"\n[INFO] Fetching {len(dataset_keys)} datasets...")

    results = {}
    for dataset_key in dataset_keys:
        samples = fetch_dataset(
            dataset_manager,
            dataset_key,
            max_samples=max_samples,
            save_to_disk=True
        )
        if samples:
            results[dataset_key] = samples
            print(f"\n[OK] {dataset_key} complete\n")
        else:
            print(f"\n[FAILED] {dataset_key} failed\n")

    # Print summary
    print("\n" + "="*70)
    print("FETCH SUMMARY")
    print("="*70 + "\n")

    if not results:
        print("[WARNING] No datasets fetched successfully")
        return results

    for dataset_key, samples in results.items():
        total_samples = sum(len(s) for s in samples.values())
        num_splits = len(samples)
        print(f"[OK] {dataset_key}: {total_samples} samples across {num_splits} splits")

    print(f"\n[INFO] Cache directory: {dataset_manager.cache_dir}")
    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Fetch Vietnamese ASR datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python scripts/fetch_datasets.py --list

  # Fetch specific datasets
  python scripts/fetch_datasets.py --fetch vimd bud500

  # Fetch all datasets
  python scripts/fetch_datasets.py --fetch-all

  # Fetch with sample limit (for testing)
  python scripts/fetch_datasets.py --fetch vimd --max-samples 100

  # Use custom cache directory
  python scripts/fetch_datasets.py --fetch vimd --cache-dir ./my_data
        """
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )

    parser.add_argument(
        '--fetch',
        nargs='+',
        metavar='DATASET',
        help='Dataset(s) to fetch (e.g., vimd bud500 lsvsc)'
    )

    parser.add_argument(
        '--fetch-all',
        action='store_true',
        help='Fetch all available datasets'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Base data directory (default: ./data)'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Cache directory for downloads (default: <data-dir>/huggingface_cache)'
    )

    parser.add_argument(
        '--config-file',
        type=str,
        default='./configs/dataset_profile.json',
        help='Path to JSON file containing dataset configurations'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per split (for testing)'
    )

    args = parser.parse_args()

    # Check if HuggingFace is available
    if not HF_AVAILABLE:
        print("\n[FAILED] Error: 'datasets' library not installed")
        print("Install with: pip install datasets")
        return 1

    # Initialize DatasetManager
    try:
        dataset_manager = DatasetManager(
            base_data_dir=args.data_dir,
            config_file=args.config_file,
            use_huggingface=True,
            cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"\n[FAILED] Error initializing DatasetManager: {e}")
        return 1

    # List datasets
    if args.list:
        list_available_datasets(dataset_manager)
        return 0

    # Fetch all datasets
    if args.fetch_all:
        if not dataset_manager.hf_configs:
            print("[ERROR] No dataset configurations found")
            return 1
        dataset_keys = list(dataset_manager.hf_configs.keys())
        fetch_multiple_datasets(dataset_manager, dataset_keys, max_samples=args.max_samples)
        return 0

    # Fetch specific datasets
    if args.fetch:
        fetch_multiple_datasets(dataset_manager, args.fetch, max_samples=args.max_samples)
        return 0

    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    print("""


          HuggingFace Dataset Fetcher for Vietnamese ASR


    """)

    sys.exit(main())