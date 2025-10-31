#!/usr/bin/env python3
"""
HuggingFace Dataset Fetcher for Vietnamese ASR

This script downloads and prepares Vietnamese ASR datasets from HuggingFace Hub.
Supports multiple datasets and provides progress tracking.
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import json
from tqdm import tqdm
from datasets import load_dataset, Audio

HF_AVAILABLE = True


class HuggingFaceDatasetFetcher:
    """
    Fetcher for Vietnamese ASR datasets from HuggingFace Hub.
    """

    def __init__(self, cache_dir: str = "./data/huggingface_cache", config_file: str = "dataset_profile.json"):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory to cache downloaded datasets
            config_file: Path to JSON file containing dataset configurations
        """
        if not HF_AVAILABLE:
            raise ImportError("datasets library required. Install with: pip install datasets")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset configurations from JSON file
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset configuration file not found: {config_file}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.AVAILABLE_DATASETS = json.load(f)
        
    def list_available_datasets(self):
        """List all available Vietnamese datasets."""
        print("\n" + "="*70)
        print("Available Vietnamese ASR Datasets on HuggingFace")
        print("="*70 + "\n")
        
        for key, info in self.AVAILABLE_DATASETS.items():
            print(f"-> {key}")
            print(f"   ID: {info['id']}")
            print(f"   Description: {info['description']}")
            print(f"   Splits: {', '.join(info['splits'])}")
            print()
    
    def fetch_dataset(
        self,
        dataset_key: str,
        splits: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        save_to_disk: bool = True
    ) -> Dict:
        """
        Fetch a dataset from HuggingFace.
        
        Args:
            dataset_key: Key of the dataset to fetch
            splits: List of splits to download (default: all)
            max_samples: Maximum samples per split (for testing)
            save_to_disk: Whether to save to disk
            
        Returns:
            Dictionary of loaded dataset splits
        """
        if dataset_key not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}. Use list_available_datasets()")
        
        info = self.AVAILABLE_DATASETS[dataset_key]
        
        print(f"\n{'='*70}")
        print(f"Fetching: {info['description']}")
        print(f"{'='*70}\n")
        
        # Determine splits to download
        if splits is None:
            splits = info['splits']
        
        dataset_splits = {}
        
        # Download each split
        for split in splits:
            if split not in info['splits']:
                print(f"[WARNING] Split '{split}' not available for {dataset_key}")
                continue
            
            print(f"Downloading split: {split}")
            
            try:
                # Load dataset
                if info['config']:
                    dataset = load_dataset(
                        info['id'],
                        info['config'],
                        split=split,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        info['id'],
                        split=split,
                        cache_dir=str(self.cache_dir),
                        trust_remote_code=True
                    )
                
                # Limit samples if requested
                if max_samples and len(dataset) > max_samples:
                    print(f"  Limiting to {max_samples} samples (from {len(dataset)})")
                    dataset = dataset.select(range(max_samples))
                
                print(f"  [OK] Loaded {len(dataset)} samples")
                
                # Cast audio column to Audio type if needed
                if info['audio_column'] in dataset.column_names:
                    dataset = dataset.cast_column(info['audio_column'], Audio(sampling_rate=16000))
                
                dataset_splits[split] = dataset
                
            except Exception as e:
                print(f"  [FAILED] Failed to download {split}: {e}")
        
        # Save to disk if requested
        if save_to_disk and dataset_splits:
            output_dir = self.cache_dir / dataset_key
            output_dir.mkdir(exist_ok=True)
            
            print(f"\nSaving to disk: {output_dir}")
            for split, dataset in dataset_splits.items():
                split_dir = output_dir / split
                dataset.save_to_disk(str(split_dir))
                print(f"  [OK] Saved {split}")
            
            # Save metadata
            metadata = {
                'dataset_key': dataset_key,
                'dataset_id': info['id'],
                'config': info['config'],
                'splits': list(dataset_splits.keys()),
                'audio_column': info['audio_column'],
                'text_column': info['text_column'],
                'total_samples': sum(len(ds) for ds in dataset_splits.values())
            }
            
            metadata_path = output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  [OK] Saved metadata")
        
        return dataset_splits
    
    def fetch_multiple_datasets(
        self,
        dataset_keys: List[str],
        max_samples: Optional[int] = None
    ):
        """
        Fetch multiple datasets.
        
        Args:
            dataset_keys: List of dataset keys to fetch
            max_samples: Maximum samples per split per dataset
        """
        print(f"\nFetching {len(dataset_keys)} datasets...")
        
        results = {}
        for dataset_key in dataset_keys:
            try:
                datasets = self.fetch_dataset(
                    dataset_key,
                    max_samples=max_samples,
                    save_to_disk=True
                )
                results[dataset_key] = datasets
                print(f"\n[OK] {dataset_key} complete\n")
            except Exception as e:
                print(f"\n[FAILED] {dataset_key} failed: {e}\n")
        
        # Print summary
        print("\n" + "="*70)
        print("FETCH SUMMARY")
        print("="*70 + "\n")
        
        for dataset_key, datasets in results.items():
            total_samples = sum(len(ds) for ds in datasets.values())
            print(f"[OK] {dataset_key}: {total_samples} samples across {len(datasets)} splits")
        
        return results
    
    def create_metadata_summary(self):
        """Create a summary of all downloaded datasets."""
        summary = []
        
        for dataset_key in self.AVAILABLE_DATASETS.keys():
            dataset_dir = self.cache_dir / dataset_key
            metadata_path = dataset_dir / 'metadata.json'
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                summary.append(metadata)
        
        if summary:
            summary_path = self.cache_dir / 'datasets_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n[OK] Summary saved to: {summary_path}")
            return summary
        else:
            print("\n[WARNING] No datasets found in cache")
            return []


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Fetch Vietnamese ASR datasets from HuggingFace"
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
        help='Dataset(s) to fetch (e.g., common_voice_vi vivos)'
    )
    
    parser.add_argument(
        '--fetch-all',
        action='store_true',
        help='Fetch all available datasets'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./data',
        help='Cache directory for datasets'
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
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=None,
        help='Specific splits to download (e.g., train test)'
    )
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("\n[FAILED] Error: datasets library not installed")
        print("Install with: pip install datasets")
        return 1

    # Initialize fetcher
    fetcher = HuggingFaceDatasetFetcher(cache_dir=args.cache_dir, config_file=args.config_file)
    
    # List datasets
    if args.list:
        fetcher.list_available_datasets()
        return 0
    
    # Fetch all datasets
    if args.fetch_all:
        dataset_keys = list(fetcher.AVAILABLE_DATASETS.keys())
        fetcher.fetch_multiple_datasets(dataset_keys, max_samples=args.max_samples)
        fetcher.create_metadata_summary()
        return 0
    
    # Fetch specific datasets
    if args.fetch:
        fetcher.fetch_multiple_datasets(args.fetch, max_samples=args.max_samples)
        fetcher.create_metadata_summary()
        return 0
    
    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    
    print("""

                                                                              
              HuggingFace Dataset Fetcher for Vietnamese ASR                  
                                                                              

    """)
    
    sys.exit(main())
