# Vietnamese ASR End-to-End Evaluation Pipeline

A comprehensive, modular evaluation framework for Vietnamese Automatic Speech Recognition (ASR) systems.

## 🎯 Features

- **Multi-Dataset Support**: [ViMD](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset), [Viet BUD500](https://huggingface.co/datasets/linhtran92/viet_bud500 ), [LSVSC](https://huggingface.co/datasets/doof-ferb/LSVSC), [VLSP 2020-100h](https://huggingface.co/datasets/doof-ferb/vlsp2020_vinai_100h), [VietMed](https://huggingface.co/datasets/leduckhai/VietMed)
- **Multi-Model Evaluation**: PhoWhisper, OpenAI Whisper, Wav2Vec2-XLS-R, Wav2Vn
- **Comprehensive Metrics**: WER, CER, MER, WIL, WIP, SER, RTF
- **Automated Workflow**: From data loading to visualization
- **Modular Design**: Reusable components for custom evaluations
- **Rich Visualizations**: Heatmaps, radar charts, comparative plots

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Detailed Usage](#detailed-usage)
- [Module Documentation](#module-documentation)
- [Customization](#customization)
- [Output Examples](#output-examples)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- 16GB+ RAM recommended

### Setup

1. Clone or download this repository:
```bash
cd vietnamese_asr_eval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch with CUDA support (if available):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

## 🎬 Quick Start

### Option 1: Run with Demo Data

The system includes synthetic data generation for demonstration:

```bash
python main_evaluation.py \
    --data-dir ./data \
    --output-dir ./results \
    --max-samples 10
```

### Option 2: Run with Your Data

Organize your data following this structure:

```
data/
├── vimd/
│   ├── audio/
│   │   └── province_name/
│   │       └── *.wav
│   └── transcripts/
│       └── metadata.csv
├── bud500/
├── lsvsc/
├── vlsp2020/
└── vietmed/
```

Then run:

```bash
python main_evaluation.py --data-dir ./data --output-dir ./results
```

### Option 3: Evaluate Specific Models and Datasets

```bash
python main_evaluation.py \
    --models phowhisper-small whisper-small wav2vec2-xlsr-vietnamese \
    --datasets ViMD VLSP2020 \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --output-dir ./results
```

## 📁 Directory Structure

```
vietnamese_asr_eval/
├── main_evaluation.py          # Main orchestration script
├── metrics.py                  # Standalone metrics module
├── dataset_loader.py           # Dataset loading and preprocessing
├── model_evaluator.py          # Model loading and inference
├── visualization.py            # Plotting and visualization
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Your datasets go here
│   ├── vimd/
│   ├── bud500/
│   ├── lsvsc/
│   ├── vlsp2020/
│   └── vietmed/
│
└── results/                    # Output directory (auto-created)
    ├── evaluation_results_*.csv
    ├── evaluation_summary.txt
    ├── dataset_statistics.csv
    ├── eda_report.json
    └── plots/
        ├── wer_comparison.png
        ├── cer_comparison.png
        ├── metrics_heatmap.png
        ├── performance_radar.png
        ├── rtf_comparison.png
        └── error_breakdown.png
```

## 📖 Detailed Usage

### Command Line Options

```bash
python main_evaluation.py --help
```

**Available options:**

- `--data-dir`: Base directory containing datasets (default: `./data`)
- `--output-dir`: Directory for output results (default: `./results`)
- `--models`: Models to evaluate (space-separated)
- `--datasets`: Datasets to evaluate (space-separated)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--max-samples`: Maximum samples per dataset for quick testing
- `--list-models`: List all available models

### Available Models

List all available models:

```bash
python main_evaluation.py --list-models
```

**Supported models:**
- `phowhisper-tiny`: PhoWhisper-tiny (39M params)
- `phowhisper-base`: PhoWhisper-base (74M params)
- `phowhisper-small`: PhoWhisper-small (244M params)
- `phowhisper-medium`: PhoWhisper-medium (769M params)
- `phowhisper-large`: PhoWhisper-large (1.5B params)
- `whisper-small`: OpenAI Whisper-small
- `whisper-medium`: OpenAI Whisper-medium
- `whisper-large`: OpenAI Whisper-large-v3
- `wav2vec2-xlsr-vietnamese`: Wav2Vec2-XLSR Vietnamese
- `wav2vec2-base-vietnamese`: Wav2Vec2-Base Vietnamese
- `wav2vn`: Wav2Vn (placeholder)

### Dataset Format

Each dataset should have the following structure:

**metadata.csv format:**
```csv
file_id,transcription,province,speaker_id
audio_001,xin chào tôi là người việt nam,hanoi,speaker_01
audio_002,hôm nay thời tiết đẹp,hcm,speaker_02
```

Audio files should be:
- Format: WAV (16kHz, mono recommended)
- Quality: Clear speech, minimal background noise
- Duration: 1-30 seconds per sample recommended

## 📚 Module Documentation

### 1. Metrics Module (`metrics.py`)

Standalone module for calculating ASR metrics:

```python
from metrics import ASRMetrics, RTFTimer

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
```

### 2. Dataset Loader (`dataset_loader.py`)

Handle multiple Vietnamese datasets:

```python
from dataset_loader import DatasetManager

manager = DatasetManager(base_data_dir="./data")
datasets = manager.load_all_datasets()
stats = manager.get_dataset_statistics()
splits = manager.prepare_train_test_splits()
```

### 3. Model Evaluator (`model_evaluator.py`)

Load and evaluate ASR models:

```python
from model_evaluator import ModelEvaluator, ModelFactory

evaluator = ModelEvaluator(
    models_to_evaluate=['phowhisper-small', 'whisper-small']
)
evaluator.load_models()
models = evaluator.get_loaded_models()

# Transcribe
transcription = models['phowhisper-small'].transcribe("audio.wav")
```

### 4. Visualization (`visualization.py`)

Create comprehensive visualizations:

```python
from visualization import ASRVisualizer
import pandas as pd

visualizer = ASRVisualizer(output_dir="./plots")
results_df = pd.read_csv("results.csv")

# Create all plots
visualizer.create_comprehensive_report(results_df)
```

## 🎨 Customization

### Adding a New Dataset

1. Create a new loader class in `dataset_loader.py`:

```python
class MyDatasetLoader(DatasetLoader):
    def load_dataset(self) -> List[AudioSample]:
        # Your loading logic here
        samples = []
        # ... load your data ...
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

1. Add model configuration in `model_evaluator.py`:

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

1. Add metric function in `metrics.py`:

```python
@staticmethod
def calculate_my_metric(reference: str, hypothesis: str) -> float:
    # Your metric calculation
    return score
```

2. Include in `calculate_all_metrics()` method

## 📊 Output Examples

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
...
```

### Visualizations

The pipeline generates multiple plots:
- **WER/CER/MER Comparison**: Bar charts comparing metrics across models and datasets
- **Metrics Heatmap**: Color-coded matrix of all metrics
- **Performance Radar**: Multi-dimensional performance comparison
- **RTF Comparison**: Real-time factor analysis
- **Error Breakdown**: Distribution of insertion/deletion/substitution errors
- **Dataset Statistics**: Overview of dataset characteristics

## 🔧 Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Reduce batch size (evaluation is done sequentially by default)
2. Use smaller model variants (e.g., `phowhisper-small` instead of `large`)
3. Use `--max-samples` to limit evaluation size
4. Enable CPU-only mode by not installing CUDA PyTorch

### Model Loading Failures

If models fail to load:

1. Check internet connection (models download from HuggingFace)
2. Verify HuggingFace authentication if needed
3. The system will fall back to mock transcription for demonstration

### Dataset Loading Issues

If datasets don't load:

1. Verify directory structure matches expected format
2. Check metadata CSV format and encoding (UTF-8)
3. The system will generate synthetic samples for demonstration if data is missing

## 📝 Citation

If you use this evaluation pipeline in your research, please cite:

```bibtex
@software{vietnamese_asr_eval_2024,
  title = {Vietnamese ASR End-to-End Evaluation Pipeline},
  year = {2024},
  author = {Your Name},
  url = {https://github.com/yourusername/vietnamese-asr-eval}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## 🙏 Acknowledgments

- VinAI Research for PhoWhisper models
- OpenAI for Whisper models
- Facebook AI for Wav2Vec2 models
- Contributors to the Vietnamese ASR datasets
