# Configuration Utilities - Quick Reference

## Overview

Two new utility functions have been added to `notebook_utils.py` to simplify configuration file management in notebooks:

- **`load_config(name)`** - Load a configuration file from `configs/` directory
- **`list_available_configs()`** - List all available configuration files

---

## Quick Start

```python
from src.notebook_utils import load_config, list_available_configs

# List all configs
configs = list_available_configs()

# Load a config
dataset_config = load_config('dataset_profile')

# Use the config
for dataset_name, info in dataset_config.items():
    print(f"{dataset_name}: {info['id']}")
```

---

## Function Reference

### `list_available_configs(config_dir=None)`

Lists all JSON configuration files in the configs directory.

**Parameters:**
- `config_dir` (Path, optional): Custom config directory path. Defaults to `{project_root}/configs/`

**Returns:**
- `list`: List of config file names (without .json extension)

**Example:**
```python
configs = list_available_configs()
# Output: ['dataset_profile']
```

---

### `load_config(config_name, config_dir=None)`

Loads and parses a JSON configuration file.

**Parameters:**
- `config_name` (str): Config filename (with or without .json extension)
- `config_dir` (Path, optional): Custom config directory path

**Returns:**
- `dict`: Parsed configuration data

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `json.JSONDecodeError`: If config file is invalid JSON

**Example:**
```python
# Both work the same
config = load_config('dataset_profile')
config = load_config('dataset_profile.json')
```

---

## Current Configuration Files

### `dataset_profile.json`

Contains HuggingFace dataset configurations for Vietnamese ASR datasets.

**Structure:**
```json
{
  "dataset_name": {
    "id": "huggingface-org/dataset-id",
    "config": "config_name",
    "description": "Dataset description",
    "splits": ["train", "test", "validation"],
    "audio_column": "audio",
    "text_column": "transcription"
  }
}
```

**Available Datasets:**
- `vimd` - Vietnamese Multi-Dialect Dataset
- `bud500` - Viet BUD500 (500 hours)
- `lsvsc` - Large-Scale Vietnamese Speech Corpus
- `vlsp2020` - VLSP 2020-100H
- `vietmed` - Vietnamese Medical Domain Dataset

**Usage Example:**
```python
config = load_config('dataset_profile')

# Get all dataset names
datasets = list(config.keys())

# Access specific dataset
vimd_config = config['vimd']
hf_id = vimd_config['id']  # 'nguyendv02/ViMD_Dataset'
splits = vimd_config['splits']  # ['train', 'test', 'valid']
```

---

## Common Use Cases

### 1. List All Datasets

```python
from src.notebook_utils import load_config

config = load_config('dataset_profile')
print("Available datasets:", list(config.keys()))
```

### 2. Get Dataset Details

```python
config = load_config('dataset_profile')

for name, info in config.items():
    print(f"\n{name.upper()}:")
    print(f"  ID: {info['id']}")
    print(f"  Description: {info['description']}")
```

### 3. Use with DatasetManager

```python
from src.notebook_utils import setup_paths
from src.dataset_loader import DatasetManager

PATHS = setup_paths()
dataset_manager = DatasetManager(config_file=PATHS['config_file'])

# DatasetManager automatically uses the config
samples = dataset_manager.load_dataset('vimd')
```

### 4. Validate Dataset Availability

```python
from src.notebook_utils import load_config

config = load_config('dataset_profile')
required_datasets = ['vimd', 'vlsp2020']

available = [ds for ds in required_datasets if ds in config]
missing = [ds for ds in required_datasets if ds not in config]

print(f"Available: {available}")
print(f"Missing: {missing}")
```

---

## Error Handling

```python
from src.notebook_utils import load_config
import json

try:
    config = load_config('nonexistent')
except FileNotFoundError as e:
    print(f"Config not found: {e}")

try:
    config = load_config('invalid.json')
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

---

## Creating New Configs

To add your own configuration:

1. Create a JSON file in `configs/` directory
2. Use a descriptive name (e.g., `model_params.json`)
3. Load using `load_config('model_params')`

**Example: `configs/training_params.json`**
```json
{
  "batch_size": 8,
  "learning_rate": 1e-4,
  "epochs": 10,
  "optimizer": "adam"
}
```

**Load in notebook:**
```python
params = load_config('training_params')
batch_size = params['batch_size']
```

---

## Integration with Existing Notebooks

All evaluation notebooks (01-05) automatically use these utilities through:

```python
# Cell 1: Setup
PATHS = setup_paths()

# Cell 5: Dataset loading
dataset_manager = DatasetManager(config_file=PATHS['config_file'])
```

The `config_file` path is automatically set by `setup_paths()` to point to `configs/dataset_profile.json`.

---

## Benefits

- **Centralized Configuration**: All dataset configs in one place
- **Easy Updates**: Change HuggingFace IDs without modifying code
- **Type Safety**: JSON schema validation
- **Environment Agnostic**: Works in both local and Colab
- **Error Reporting**: Clear error messages with suggestions

---

## Documentation Files

- **`config_utils_examples.md`** - Comprehensive examples and use cases
- **`README.md`** - Main notebooks documentation (includes config utilities section)
- **`CONFIG_UTILS_README.md`** - This quick reference guide

---

## Testing

Run the test suite to verify functionality:

```bash
cd notebooks
python3 -c "
from src.notebook_utils import list_available_configs, load_config
configs = list_available_configs()
config = load_config('dataset_profile')
print(f'Test passed: {len(config)} datasets loaded')
"
```

---

**Version:** 1.0.0
**Created:** November 2024
**Updated:** November 2024
