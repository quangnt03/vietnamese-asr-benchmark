# Vietnamese ASR Evaluation - Quick Reference

Quick command reference for the Vietnamese ASR Evaluation Framework.

## Installation

### Method 1: Install as Package (Recommended)
```bash
git clone https://github.com/quangnt03/vietnamese-asr-benchmark.git
cd vietnamese-asr-benchmark
pip install -e .
```

### Method 2: Dependencies Only
```bash
git clone https://github.com/quangnt03/vietnamese-asr-benchmark.git
cd vietnamese-asr-benchmark
pip install -r requirements.txt
```

### Verify Installation
```bash
vietnamese-asr-benchmark-check                    # If installed
python scripts/check_setup.py   # Direct execution
```

## Quick Start Commands

### Fastest Start - Run Demo
```bash
vietnamese-asr-benchmark-demo                     # If installed
python scripts/demo.py          # Direct execution
```

### List Available Models
```bash
vietnamese-asr-benchmark-eval --list-models       # If installed
python scripts/main_evaluation.py --list-models
```

### Quick Test (10 samples)
```bash
vietnamese-asr-benchmark-eval --max-samples 10
python scripts/main_evaluation.py --max-samples 10
```

### Full Evaluation
```bash
vietnamese-asr-benchmark-eval --data-dir ./data --output-dir ./results
python scripts/main_evaluation.py --data-dir ./data --output-dir ./results
```

## Common Use Cases

### Evaluate Specific Models
```bash
vietnamese-asr-benchmark-eval --models phowhisper-small whisper-small
python scripts/main_evaluation.py --models phowhisper-small whisper-small
```

### Evaluate Specific Datasets
```bash
vietnamese-asr-benchmark-eval --datasets ViMD VLSP2020
python scripts/main_evaluation.py --datasets ViMD VLSP2020
```

### Combined: Specific Models + Datasets
```bash
vietnamese-asr-benchmark-eval \
    --models phowhisper-small whisper-small \
    --datasets ViMD VLSP2020 \
    --max-samples 50
```

### Custom Train/Val/Test Split
```bash
vietnamese-asr-benchmark-eval \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --data-dir ./data
```

## HuggingFace Datasets

### Fetch Datasets from HuggingFace
```bash
# Single dataset
vietnamese-asr-benchmark-fetch --fetch common_voice_vi --max-samples 100
python scripts/fetch_datasets.py --fetch common_voice_vi --max-samples 100

# Multiple datasets
vietnamese-asr-benchmark-fetch --fetch common_voice_vi vivos fosd --max-samples 100
python scripts/fetch_datasets.py --fetch common_voice_vi vivos fosd --max-samples 100

# Fetch all available datasets
vietnamese-asr-benchmark-fetch --fetch-all
python scripts/fetch_datasets.py --fetch-all
```

### List Available HuggingFace Datasets
```bash
vietnamese-asr-benchmark-fetch --list
python scripts/fetch_datasets.py --list
```

### Evaluate with HuggingFace Datasets
```bash
vietnamese-asr-benchmark-eval \
    --use-huggingface \
    --hf-datasets common_voice_vi vivos
```

## Command-Line Options

### Main Evaluation (vietnamese-asr-benchmark-eval)

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir PATH` | Base directory for datasets | `./data` |
| `--output-dir PATH` | Output directory for results | `./results` |
| `--models MODEL [...]` | Models to evaluate | All |
| `--datasets DATASET [...]` | Datasets to evaluate | All |
| `--train-ratio FLOAT` | Training set ratio | `0.7` |
| `--val-ratio FLOAT` | Validation set ratio | `0.15` |
| `--max-samples INT` | Max samples per dataset | Unlimited |
| `--list-models` | List available models | - |
| `--use-huggingface` | Use HuggingFace datasets | False |
| `--hf-datasets DATASET [...]` | HuggingFace datasets | None |

### Dataset Fetcher (vietnamese-asr-benchmark-fetch)

| Option | Description |
|--------|-------------|
| `--list` | List available HuggingFace datasets |
| `--fetch DATASET [...]` | Fetch specific datasets |
| `--fetch-all` | Fetch all available datasets |
| `--max-samples INT` | Max samples per dataset |
| `--output-dir PATH` | Output directory |

## Python Package Usage

### Import the Package
```python
from src import ASRMetrics, DatasetManager, ModelEvaluator, ASRVisualizer
```

### Calculate Metrics
```python
from src import ASRMetrics

calculator = ASRMetrics()
metrics = calculator.calculate_all_metrics(
    reference="xin chào tôi là người việt nam",
    hypothesis="xin chào tôi là người việt"
)
print(metrics)
```

### Load Datasets
```python
from src import DatasetManager

manager = DatasetManager(base_data_dir="./data")
datasets = manager.load_all_datasets()
stats = manager.get_dataset_statistics()
```

### Evaluate Models
```python
from src import ModelEvaluator

evaluator = ModelEvaluator(models_to_evaluate=['phowhisper-small'])
evaluator.load_models()
models = evaluator.get_loaded_models()
transcription = models['phowhisper-small'].transcribe("audio.wav")
```

### Create Visualizations
```python
from src import ASRVisualizer
import pandas as pd

visualizer = ASRVisualizer(output_dir="./plots")
results_df = pd.read_csv("results.csv")
visualizer.create_comprehensive_report(results_df)
```

## Jupyter Notebooks

### Start Jupyter
```bash
jupyter notebook notebooks/
```

### Available Notebooks
- `custom_analysis_example.ipynb` - Custom analysis workflows
- `huggingface_integration_example.ipynb` - HuggingFace integration

### Google Colab
Both notebooks are Colab-compatible and will auto-setup the environment.

## Available Models

| Model Key | Name | Parameters |
|-----------|------|------------|
| `phowhisper-tiny` | PhoWhisper-tiny | 39M |
| `phowhisper-base` | PhoWhisper-base | 74M |
| `phowhisper-small` | PhoWhisper-small | 244M |
| `phowhisper-medium` | PhoWhisper-medium | 769M |
| `phowhisper-large` | PhoWhisper-large | 1.5B |
| `whisper-small` | OpenAI Whisper-small | 244M |
| `whisper-medium` | OpenAI Whisper-medium | 769M |
| `whisper-large` | OpenAI Whisper-large-v3 | 1.5B |
| `wav2vec2-xlsr-vietnamese` | Wav2Vec2-XLSR-53 Vietnamese | 300M |
| `wav2vec2-base-vietnamese` | Wav2Vec2-Base Vietnamese | 95M |

## Supported Datasets

### Local Datasets
- **ViMD**: Vietnamese Multidialectal Dataset
- **BUD500**: Vietnamese speech corpus (500 speakers)
- **LSVSC**: Large-scale Vietnamese Speech Corpus
- **VLSP2020**: VLSP 2020 Vietnamese speech (100h)
- **VietMed**: Vietnamese Medical speech

### HuggingFace Datasets
- **common_voice_vi**: Mozilla Common Voice Vietnamese
- **vivos**: VIVOS Vietnamese speech corpus
- **fosd**: FOSD Vietnamese speech dataset

## Metrics Explained

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| **WER** | [0, ∞) | Lower | Word Error Rate |
| **CER** | [0, ∞) | Lower | Character Error Rate |
| **MER** | [0, 1] | Lower | Match Error Rate |
| **WIL** | [0, 1] | Lower | Word Information Lost |
| **WIP** | [0, 1] | Higher | Word Information Preserved |
| **SER** | [0, 1] | Lower | Sentence Error Rate |
| **RTF** | [0, ∞) | Lower | Real-Time Factor |

**RTF Interpretation:**
- RTF < 1.0: Faster than real-time
- RTF = 1.0: Real-time processing
- RTF > 1.0: Slower than real-time

## Output Files

### CSV Results
- `evaluation_results_YYYYMMDD_HHMMSS.csv` - Detailed metrics for all models/datasets
- `dataset_statistics.csv` - Dataset characteristics and statistics

### Text Summaries
- `evaluation_summary.txt` - Text summary of evaluation results
- `eda_report.json` - Exploratory data analysis report

### Visualizations (in `plots/`)
- `wer_comparison.png` - WER comparison across models
- `cer_comparison.png` - CER comparison across models
- `mer_comparison.png` - MER comparison across models
- `metrics_heatmap.png` - Heatmap of all metrics
- `performance_radar.png` - Radar chart of model performance
- `rtf_comparison.png` - Real-time factor comparison
- `error_breakdown.png` - Error type distribution
- `dataset_statistics.png` - Dataset overview

## Directory Structure

```
vietnamese-asr-benchmark/
├── src/                      # Core library
├── scripts/                  # Executable scripts
├── notebooks/                # Jupyter notebooks
├── docs/                     # Documentation
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── data/                     # Datasets (gitignored)
└── results/                  # Output (gitignored)
```

## Troubleshooting

### Out of Memory
```bash
# Use smaller samples
vietnamese-asr-benchmark-eval --max-samples 10

# Use smaller models
vietnamese-asr-benchmark-eval --models phowhisper-small
```

### Model Loading Issues
```bash
# Check setup
vietnamese-asr-benchmark-check

# Login to HuggingFace (if needed)
huggingface-cli login
```

### Dataset Issues
```bash
# Generate synthetic data for demo
vietnamese-asr-benchmark-demo
```

## Quick Tips

1. **Start with demo**: Run `vietnamese-asr-benchmark-demo` first to verify setup
2. **Test incrementally**: Use `--max-samples 10` for quick iteration
3. **Check available models**: Use `--list-models` before evaluation
4. **Use specific models**: Specify models explicitly to save time
5. **Monitor memory**: Start with small models, scale up as needed

## Documentation Links

- [README.md](../README.md) - Full documentation
- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Project organization
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Technical details
- [FILE_INDEX.md](FILE_INDEX.md) - File navigation

## One-Liner Examples

```bash
# Quick demo
vietnamese-asr-benchmark-demo

# Quick test
vietnamese-asr-benchmark-eval --max-samples 10

# Specific evaluation
vietnamese-asr-benchmark-eval --models phowhisper-small --datasets ViMD --max-samples 50

# Fetch HuggingFace datasets
vietnamese-asr-benchmark-fetch --fetch common_voice_vi --max-samples 100

# Check setup
vietnamese-asr-benchmark-check
```

---

**Version**: 1.0.0
**Last Updated**: October 2024

For complete documentation, see [README.md](../README.md)
