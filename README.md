# Vietnamese ASR Evaluation Framework

A comprehensive, production-ready evaluation framework for Vietnamese Automatic Speech Recognition (ASR) systems with modular architecture and extensive metrics support.

## Features

- **Multi-Dataset Support**: ViMD, Viet BUD500, LSVSC, VLSP 2020, VietMed, and HuggingFace datasets
- **Multi-Model Evaluation**: PhoWhisper, OpenAI Whisper, Wav2Vec2-XLS-R, Wav2Vn
- **Comprehensive Metrics**: WER, CER, MER, WIL, WIP, SER, RTF with detailed error analysis
- **Automated Workflow**: End-to-end pipeline from data loading to visualization
- **Modular Design**: Installable Python package with reusable components
- **Rich Visualizations**: Heatmaps, radar charts, comparative plots, error breakdowns
- **CLI Commands**: Easy-to-use command-line interface
- **Jupyter Notebooks**: Interactive examples for custom analysis

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Module Documentation](#module-documentation)
- [Customization](#customization)
- [Output Examples](#output-examples)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-capable GPU (optional, for faster inference)
- 16GB+ RAM recommended

### Method 1: Install as Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/quangnt03/vietnamese-asr-benchmark.git
cd vietnamese-asr-benchmark

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Method 2: Install Dependencies Only

```bash
# Clone the repository
git clone https://github.com/quangnt03/vietnamese-asr-benchmark.git
cd vietnamese-asr-benchmark

# Install dependencies
pip install -r requirements.txt
```

### Optional: CUDA Support

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (default)
pip install torch torchvision torchaudio
```

### Verify Installation

```bash
# Using installed command
vietnamese-asr-benchmark-check

# Or run directly
python scripts/check_setup.py
```

For detailed installation instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Quick Start

### Option 1: Run Demo

The quickest way to see the system in action:

```bash
# Using installed command
vietnamese-asr-benchmark-demo

# Or run script directly
python scripts/demo.py
```

This runs a demonstration with synthetic data and mock models.

### Option 2: Quick Test (10 samples)

```bash
# Using installed command
vietnamese-asr-benchmark-eval --max-samples 10

# Or run script directly
python scripts/main_evaluation.py --max-samples 10
```

### Option 3: Full Evaluation

```bash
# Using installed command
vietnamese-asr-benchmark-eval --data-dir ./data --output-dir ./results

# Or run script directly
python scripts/main_evaluation.py --data-dir ./data --output-dir ./results
```

### Option 4: Specific Models and Datasets

```bash
# Using installed command
vietnamese-asr-benchmark-eval \
    --models phowhisper-small whisper-small \
    --datasets ViMD VLSP2020 \
    --max-samples 50

# Or run script directly
python scripts/main_evaluation.py \
    --models phowhisper-small whisper-small \
    --datasets ViMD VLSP2020 \
    --max-samples 50
```

### List Available Models

```bash
# Using installed command
vietnamese-asr-benchmark-eval --list-models

# Or run script directly
python scripts/main_evaluation.py --list-models
```

## Project Structure

```
vietnamese-asr-benchmark/
├── src/                     # Core library (importable package)
│   ├── __init__.py          # Package initialization
│   ├── metrics.py           # ASR metrics calculation
│   ├── dataset_loader.py    # Dataset loading & management
│   ├── model_evaluator.py   # Model loading & evaluation
│   └── visualization.py     # Results visualization
│
├── scripts/                  # Executable scripts
│   ├── main_evaluation.py   # Main evaluation pipeline
│   ├── demo.py              # Quick demonstration
│   ├── fetch_datasets.py    # HuggingFace dataset fetcher
│   └── check_setup.py       # Setup verification
│
├── notebooks/                # Jupyter notebooks
│   ├── custom_analysis_example.ipynb
│   └── huggingface_integration_example.ipynb
│
├── docs/                     # Documentation
│   ├── INSTALLATION.md      # Installation guide
│   ├── PROJECT_STRUCTURE.md # Project structure details
│   ├── QUICK_REFERENCE.md   # Command cheat sheet
│   └── PROJECT_SUMMARY.md   # Technical overview
│
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── data/                     # Datasets (gitignored)
├── results/                  # Output results (gitignored)
│
├── setup.py                  # Package installation
├── pyproject.toml            # Modern packaging config
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

For detailed structure information, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Usage

### Command-Line Interface

After installation, the following commands are available:

```bash
# Run evaluation
vietnamese-asr-benchmark-eval [options]

# Run demo
vietnamese-asr-benchmark-demo

# Fetch HuggingFace datasets
vietnamese-asr-benchmark-fetch --fetch common_voice_vi vivos --max-samples 100

# Check setup
vietnamese-asr-benchmark-check
```

### Command-Line Options

```bash
vietnamese-asr-benchmark-eval --help
```

**Available options:**

- `--data-dir PATH`: Base directory containing datasets (default: `./data`)
- `--output-dir PATH`: Directory for output results (default: `./results`)
- `--models MODEL [MODEL ...]`: Models to evaluate (space-separated)
- `--datasets DATASET [DATASET ...]`: Datasets to evaluate (space-separated)
- `--train-ratio FLOAT`: Training set ratio (default: 0.7)
- `--val-ratio FLOAT`: Validation set ratio (default: 0.15)
- `--max-samples INT`: Maximum samples per dataset for quick testing
- `--list-models`: List all available models
- `--use-huggingface`: Use HuggingFace datasets
- `--hf-datasets DATASET [DATASET ...]`: HuggingFace datasets to use

### Using as Python Package

After installation, import the package in your Python code:

```python
from src import ASRMetrics, DatasetManager, ModelEvaluator, ASRVisualizer

# Calculate metrics
calculator = ASRMetrics()
metrics = calculator.calculate_all_metrics(
    reference="xin chào tôi là người việt nam",
    hypothesis="xin chào tôi là người việt"
)

# Load datasets
manager = DatasetManager(base_data_dir="./data")
datasets = manager.load_all_datasets()

# Evaluate models
evaluator = ModelEvaluator(models_to_evaluate=['phowhisper-small'])
evaluator.load_models()
models = evaluator.get_loaded_models()

# Create visualizations
visualizer = ASRVisualizer(output_dir="./plots")
visualizer.create_comprehensive_report(results_df)
```

### Jupyter Notebooks

Interactive examples are provided in the `notebooks/` directory:

```bash
# Start Jupyter
jupyter notebook notebooks/

# Available notebooks:
# - custom_analysis_example.ipynb: Custom analysis workflows
# - huggingface_integration_example.ipynb: HuggingFace datasets integration
```

Both notebooks are compatible with Google Colab and will automatically set up the environment.

## Module Documentation

### 1. Metrics Module (src.metrics)

Standalone module for calculating ASR metrics:

```python
from src.metrics import ASRMetrics, RTFTimer, format_metrics_report

calculator = ASRMetrics()

# Single utterance
metrics = calculator.calculate_all_metrics(
    reference="xin chào tôi là người việt nam",
    hypothesis="xin chào tôi là người việt",
    audio_duration=3.5,
    processing_time=0.5
)

# Batch processing
batch_metrics = calculator.calculate_batch_metrics(
    references=["text1", "text2"],
    hypotheses=["hyp1", "hyp2"]
)

# Print formatted report
print(format_metrics_report(metrics, "Evaluation Results"))
```

**Metrics Calculated:**
- WER (Word Error Rate)
- CER (Character Error Rate)
- MER (Match Error Rate)
- WIL (Word Information Lost)
- WIP (Word Information Preserved)
- SER (Sentence Error Rate)
- RTF (Real-Time Factor)

### 2. Dataset Loader (src.dataset_loader)

Handle multiple Vietnamese datasets:

```python
from src.dataset_loader import DatasetManager, AudioSample

manager = DatasetManager(base_data_dir="./data")

# Load all datasets
datasets = manager.load_all_datasets()

# Get statistics
stats = manager.get_dataset_statistics()

# Prepare train/val/test splits
splits = manager.prepare_train_test_splits(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

**Supported Datasets:**
- ViMD (Vietnamese Multidialectal)
- BUD500 (Vietnamese speech corpus)
- LSVSC (Large-scale Vietnamese Speech Corpus)
- VLSP 2020 (100h Vietnamese speech)
- VietMed (Medical Vietnamese speech)
- HuggingFace datasets (Common Voice, VIVOS, FOSD)

### 3. Model Evaluator (src.model_evaluator)

Load and evaluate ASR models:

```python
from src.model_evaluator import ModelEvaluator, ModelFactory

# List available models
available_models = ModelFactory.get_available_models()

# Load specific models
evaluator = ModelEvaluator(
    models_to_evaluate=['phowhisper-small', 'whisper-small']
)
evaluator.load_models()
models = evaluator.get_loaded_models()

# Transcribe audio
transcription = models['phowhisper-small'].transcribe("audio.wav")
```

**Supported Models:**
- PhoWhisper: tiny, base, small, medium, large
- OpenAI Whisper: small, medium, large-v3
- Wav2Vec2-XLSR: Vietnamese fine-tuned variants
- Custom models (easy to add)

### 4. Visualization (src.visualization)

Create comprehensive visualizations:

```python
from src.visualization import ASRVisualizer
import pandas as pd

visualizer = ASRVisualizer(output_dir="./plots")
results_df = pd.read_csv("results.csv")

# Create individual plots
visualizer.plot_metric_comparison(results_df, metric='wer')
visualizer.plot_all_metrics_heatmap(results_df)
visualizer.plot_model_performance_radar(results_df)

# Create all plots at once
visualizer.create_comprehensive_report(results_df)
```

**Generated Plots:**
- WER/CER/MER comparison bar charts
- Metrics heatmap (all metrics, all models/datasets)
- Performance radar chart (multi-dimensional comparison)
- RTF comparison (real-time factor analysis)
- Error breakdown (insertions/deletions/substitutions)
- Dataset statistics overview

## Customization

### Adding a New Dataset

1. Create a new loader class in `src/dataset_loader.py`:

```python
class MyDatasetLoader(DatasetLoader):
    def load_dataset(self) -> List[AudioSample]:
        samples = []
        # Your loading logic here
        return samples
```

2. Register it in `DatasetManager`:

```python
loaders = {
    'MyDataset': MyDatasetLoader,
    # ... existing loaders ...
}
```

### Adding a New Model

Add model configuration in `src/model_evaluator.py`:

```python
MODEL_CONFIGS = {
    'my-model': ModelConfig(
        name='My ASR Model',
        model_id='username/model-id',
        model_type='whisper'  # or 'wav2vec2'
    ),
    # ... existing configs ...
}
```

### Adding Custom Metrics

1. Add metric function in `src/metrics.py`:

```python
@staticmethod
def calculate_my_metric(reference: str, hypothesis: str) -> float:
    # Your metric calculation
    return score
```

2. Include in `calculate_all_metrics()` method

For more details, see [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md).

## Output Examples

### CSV Results

The main output CSV includes:

| Model | Dataset | WER | CER | MER | WIL | WIP | SER | RTF | ... |
|-------|---------|-----|-----|-----|-----|-----|-----|-----|-----|
| PhoWhisper-small | ViMD | 0.123 | 0.045 | 0.089 | 0.156 | 0.844 | 0.234 | 0.25 | ... |
| Whisper-small | ViMD | 0.145 | 0.052 | 0.098 | 0.178 | 0.822 | 0.267 | 0.18 | ... |

### Text Summary

```
================================================================================
VIETNAMESE ASR EVALUATION SUMMARY
================================================================================

OVERALL STATISTICS
--------------------------------------------------------------------------------
Models evaluated: 3
Datasets evaluated: 5
Total evaluations: 15

BEST PERFORMING MODELS (by WER)
--------------------------------------------------------------------------------
ViMD           : PhoWhisper-small              (WER: 0.1234)
BUD500         : Whisper-small                 (WER: 0.1456)
LSVSC          : PhoWhisper-small              (WER: 0.1123)
```

### Visualizations

All plots are saved as high-quality PNG files (300 DPI) in the output directory:

- `wer_comparison.png`: WER comparison across models and datasets
- `cer_comparison.png`: CER comparison across models and datasets
- `mer_comparison.png`: MER comparison across models and datasets
- `metrics_heatmap.png`: Color-coded matrix of all metrics
- `performance_radar.png`: Multi-dimensional performance comparison
- `rtf_comparison.png`: Real-time factor analysis
- `error_breakdown.png`: Distribution of insertion/deletion/substitution errors
- `dataset_statistics.png`: Overview of dataset characteristics

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Use `--max-samples` to limit evaluation size
2. Use smaller model variants (e.g., `phowhisper-small` instead of `large`)
3. Reduce batch size in model configuration
4. Use CPU-only mode if GPU memory is limited

### Model Loading Failures

If models fail to load:

1. Check internet connection (models download from HuggingFace)
2. Verify HuggingFace authentication if needed: `huggingface-cli login`
3. The system will fall back to mock transcription for demonstration
4. Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Dataset Loading Issues

If datasets don't load:

1. Verify directory structure matches expected format
2. Check metadata CSV format and encoding (UTF-8)
3. The system will generate synthetic samples for demonstration if data is missing
4. Check file permissions and paths

For more troubleshooting tips, see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [INSTALLATION.md](docs/INSTALLATION.md) - Complete installation guide
- [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) - Project organization details
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Command cheat sheet
- [PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) - Technical deep-dive
- [FILE_INDEX.md](docs/FILE_INDEX.md) - File navigation guide

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests: `pytest` (if available)
5. Format code: `black src/ scripts/` and `isort src/ scripts/`
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please ensure your code:
- Follows the existing code style
- Includes docstrings for functions and classes
- Does not include emojis (use text alternatives like [OK], [WARNING])
- Updates relevant documentation

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{vietnamese_asr_eval_2024,
  title = {Vietnamese ASR Evaluation Framework},
  year = {2024},
  author = {Vietnamese ASR Evaluation Team},
  url = {https://github.com/quangnt03/vietnamese-asr-benchmark},
  version = {1.0.0}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VinAI Research for PhoWhisper models
- OpenAI for Whisper models
- Meta AI (Facebook) for Wav2Vec2 models
- Contributors to the Vietnamese ASR datasets:
  - ViMD dataset creators
  - VLSP organizers
  - Common Voice contributors
  - HuggingFace community

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review the example notebooks in `notebooks/`

## Project Status

**Version**: 1.0.0
**Status**: Production Ready
**Last Updated**: October 2024

---

**Quick Links:**
- [Installation Guide](docs/INSTALLATION.md)
- [Quick Reference](docs/QUICK_REFERENCE.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Example Notebooks](notebooks/)
