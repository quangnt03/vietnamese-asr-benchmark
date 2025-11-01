# HuggingFace Dataset Integration Guide

## Overview

The Vietnamese ASR Benchmark now has unified HuggingFace dataset support, eliminating the duplication between `scripts/fetch_datasets.py` and `src/dataset_loader.py`.

## Key Changes

### 1. Unified Configuration
All datasets are now configured in a single file: [configs/dataset_profile.json](../../configs/dataset_profile.json)

```json
{
    "vimd": {
        "id": "nguyendv02/ViMD_Dataset",
        "config": "default",
        "description": "ViMD: Vietnamese Multi-Dialect Dataset (63 provincial dialects)",
        "splits": ["train", "test", "valid"],
        "audio_column": "audio",
        "text_column": "text"
    },
    ...
}
```

### 2. DatasetManager Now Uses HuggingFace by Default

The `DatasetManager` class now:
- Loads dataset configs from `configs/dataset_profile.json`
- Downloads from HuggingFace Hub automatically
- Caches datasets locally for reuse
- Respects predefined train/val/test splits (no more re-splitting!)

### 3. Backward Compatibility

Legacy local loaders (ViMDLoader, BUD500Loader, etc.) are still available but deprecated. Set `use_huggingface=False` to use them.

## Usage

### Basic Usage (Recommended)

```python
from src.dataset_loader import DatasetManager

# Initialize with HuggingFace support (default)
dataset_manager = DatasetManager(
    base_data_dir="./data",
    config_file="./configs/dataset_profile.json",
    use_huggingface=True  # Default
)

# Load a dataset - will download from HuggingFace if not cached
samples = dataset_manager.load_dataset(
    dataset_name="ViMD",
    respect_predefined_splits=True  # Use official splits, don't re-split
)

# Access splits
train_samples = samples['train']
val_samples = samples['val']
test_samples = samples['test']
```

### First-Time Dataset Download

```python
# First run will download from HuggingFace
dataset_manager = DatasetManager(base_data_dir="./data")
samples = dataset_manager.load_dataset("ViMD")
# Downloads and caches to ./data/huggingface_cache/vimd/

# Subsequent runs load from cache (much faster)
samples = dataset_manager.load_dataset("ViMD")
# Loads from ./data/huggingface_cache/vimd/
```

### Using Pre-Downloaded Datasets

If you already downloaded datasets using `scripts/fetch_datasets.py`:

```bash
# Download datasets first
python scripts/fetch_datasets.py --fetch vimd bud500 --cache-dir ./data
```

Then load in code:
```python
dataset_manager = DatasetManager(
    base_data_dir="./data",
    cache_dir="./data"  # Point to where fetch_datasets.py saved files
)
samples = dataset_manager.load_dataset("ViMD")
```

### Respecting Official Splits vs Custom Splits

```python
# Option 1: Use official HuggingFace splits (RECOMMENDED)
samples = dataset_manager.load_dataset(
    dataset_name="ViMD",
    respect_predefined_splits=True  # Default
)
# ViMD has: train, valid, test
# You get: samples['train'], samples['val'], samples['test']

# Option 2: Combine all data and re-split with custom ratios
samples = dataset_manager.load_dataset(
    dataset_name="ViMD",
    respect_predefined_splits=False,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
# Combines all splits and re-divides with your ratios
```

## Notebooks Integration

Update your notebooks to use the new DatasetManager:

```python
# Cell: Initialize dataset manager
from src.dataset_loader import DatasetManager

dataset_manager = DatasetManager(
    base_data_dir=str(PATHS['data_dir']),
    config_file="./configs/dataset_profile.json",
    use_huggingface=True  # Use HuggingFace datasets
)

# Cell: Load datasets
datasets_loaded = {}

for dataset_name in ["ViMD", "BUD500", "LSVSC", "VLSP2020", "VietMed"]:
    try:
        # Load with predefined splits
        samples = dataset_manager.load_dataset(
            dataset_name=dataset_name,
            respect_predefined_splits=True
        )

        # Get test split
        test_samples = samples['test']
        datasets_loaded[dataset_name] = test_samples

        print(f"[OK] {dataset_name}: {len(test_samples)} test samples loaded")
    except Exception as e:
        print(f"[WARNING] Failed to load {dataset_name}: {e}")
```

## Available Datasets

All datasets in [configs/dataset_profile.json](../../configs/dataset_profile.json):

| Dataset | HuggingFace ID | Splits | Description |
|---------|----------------|--------|-------------|
| ViMD | `nguyendv02/ViMD_Dataset` | train, test, valid | Vietnamese Multi-Dialect (63 provinces) |
| BUD500 | `linhtran92/viet_bud500` | train, validation, test | 500 hours Vietnamese Speech |
| LSVSC | `doof-ferb/LSVSC` | train, validation, test | Large-Scale Vietnamese Speech Corpus |
| VLSP2020 | `doof-ferb/vlsp2020_vinai_100h` | train | VLSP 2020-100H VinAI |
| VietMed | `leduckhai/VietMed` | train, test, dev | Vietnamese Medical Domain |

## Troubleshooting

### Issue: Datasets not downloading
```python
# Check HuggingFace availability
from src.dataset_loader import HF_AVAILABLE
print(f"HuggingFace available: {HF_AVAILABLE}")

# If False, install datasets library
# pip install datasets
```

### Issue: Authentication required
Some datasets may require HuggingFace authentication:
```bash
# Login to HuggingFace
huggingface-cli login
```

### Issue: Out of disk space
Datasets are cached. Clear cache if needed:
```bash
rm -rf ./data/huggingface_cache/*
```

### Issue: Want to use local files instead
```python
# Disable HuggingFace, use legacy loaders
dataset_manager = DatasetManager(
    base_data_dir="./data",
    use_huggingface=False
)
```

## Migration from Legacy Code

### Old Code (Using legacy loaders)
```python
# OLD - Required manual data download and file organization
dataset_manager = DatasetManager(base_data_dir="./data")
samples = dataset_manager.load_dataset("ViMD")
# Would fail with NotImplementedError if files not present
```

### New Code (Using HuggingFace)
```python
# NEW - Automatic download and caching
dataset_manager = DatasetManager(
    base_data_dir="./data",
    use_huggingface=True  # Default
)
samples = dataset_manager.load_dataset("ViMD")
# Downloads automatically if not cached, respects official splits
```

## Benefits

1. **No Duplication**: Single source of truth in `configs/dataset_profile.json`
2. **Automatic Downloads**: No manual dataset fetching required
3. **Respects Official Splits**: Uses competition/paper splits, ensures reproducibility
4. **Caching**: Downloads once, reuses forever
5. **Backward Compatible**: Legacy loaders still available if needed

## See Also

- [configs/dataset_profile.json](../../configs/dataset_profile.json) - Dataset configurations
- [scripts/fetch_datasets.py](../../scripts/fetch_datasets.py) - CLI tool for pre-downloading
- [src/dataset_loader.py](../../src/dataset_loader.py) - Source code
