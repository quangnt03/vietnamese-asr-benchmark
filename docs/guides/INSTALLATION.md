# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster model inference

## Installation Methods

### Method 1: Install from Source (Development)

This method is recommended if you want to modify the code or contribute to the project.

```bash
# Clone the repository
git clone https://github.com/quangnt03/vietnamese-asr-benchmark.git
cd vietnamese-asr-benchmark

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Method 2: Standard Installation

```bash
# Clone the repository
git clone https://github.com/quangnt03/vietnamese-asr-benchmark.git
cd vietnamese-asr-benchmark

# Install the package
pip install .
```

### Method 3: Install from requirements.txt (Basic)

If you only want to use the modules without installing the package:

```bash
# Clone the repository
git clone https://github.com/quangnt03/vietnamese-asr-benchmark.git
cd vietnamese-asr-benchmark

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

After installation, verify that everything is set up correctly:

```bash
# Using installed command
vietnamese-asr-benchmark-check

# Or run the check script directly
python scripts/check_setup.py
```

## Command-Line Tools

After installation, the following commands will be available:

- `vietnamese-asr-benchmark-eval` - Run full ASR evaluation pipeline
- `vietnamese-asr-benchmark-demo` - Run a quick demonstration with synthetic data
- `vietnamese-asr-benchmark-fetch` - Fetch datasets from HuggingFace
- `vietnamese-asr-benchmark-check` - Check system setup and dependencies

### Examples

```bash
# Run a quick demo
vietnamese-asr-benchmark-demo

# Run evaluation with specific models
vietnamese-asr-benchmark-eval --models phowhisper-small whisper-small --max-samples 10

# Fetch HuggingFace datasets
vietnamese-asr-benchmark-fetch --fetch common_voice_vi vivos --max-samples 100

# Check setup
vietnamese-asr-benchmark-check
```

## Using as a Python Package

After installation, you can import the package in your Python code:

```python
# Import the package
from src.metrics import ASRMetrics
from src.dataset_loader import DatasetManager
from src.model_evaluator import ModelEvaluator
from src.visualization import ASRVisualizer

# Use the metrics calculator
calculator = ASRMetrics()
metrics = calculator.calculate_all_metrics(
    reference="xin chào tôi là người việt nam",
    hypothesis="xin chào tôi là người việt"
)
print(metrics)

# Load datasets
manager = DatasetManager(base_data_dir="./data")
datasets = manager.load_all_datasets()

# Evaluate models
evaluator = ModelEvaluator(models_to_evaluate=['phowhisper-small'])
evaluator.load_models()
```

## Jupyter Notebooks

The notebooks are located in the `notebooks/` directory:

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or open a specific notebook
jupyter notebook notebooks/custom_analysis_example.ipynb
```

For Google Colab, the notebooks are already configured to automatically:
1. Clone the repository
2. Install dependencies
3. Set up the environment

Just upload the notebook to Colab and run!

## Troubleshooting

### CUDA/GPU Issues

If you encounter CUDA-related errors:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### HuggingFace Authentication

For private or gated models:

```bash
# Login to HuggingFace
huggingface-cli login
```

### Audio Processing Issues

If librosa fails to load audio files:

```bash
# Install ffmpeg (required for some audio formats)
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### Import Errors

If you get import errors after installation:

```bash
# Make sure you're in the right directory
cd /path/to/vietnamese-asr-benchmark

# Reinstall in development mode
pip install -e .

# Or add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/vnasrb/src"
```

## Uninstallation

```bash
# Uninstall the package
pip uninstall vietnamese-asr-benchmark

# Remove virtual environment
deactivate
rm -rf venv/
```

## Docker Installation (Coming Soon)

Docker support is planned for a future release, which will simplify installation and ensure consistency across different environments.

## Next Steps

- Read the [Quick Reference](QUICK_REFERENCE.md) for common commands
- Check the [Project Summary](PROJECT_SUMMARY.md) for technical details
- Run `vietnamese-asr-benchmark-demo` for a quick demonstration
- Explore the [notebooks](../notebooks/) for detailed examples
