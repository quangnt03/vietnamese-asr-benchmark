# Vietnamese ASR Evaluation - Project Structure

## [DIR] Directory Layout

```
vietnamese-asr-benchmark/

 [FOLDER] src/              # Core library source code
    __init__.py             # Package initialization & exports
    metrics.py              # ASR metrics calculation (WER, CER, etc.)
    dataset_loader.py       # Dataset loading & management
    model_evaluator.py      # Model loading & evaluation
    visualization.py        # Results visualization

 [FOLDER] scripts/                  # Executable scripts
    main_evaluation.py      # Main evaluation pipeline
    demo.py                 # Quick demonstration script
    fetch_datasets.py       # HuggingFace dataset fetcher
    check_setup.py          # Setup verification script

 [FOLDER] notebooks/                # Jupyter notebooks
    custom_analysis_example.ipynb
    huggingface_integration_example.ipynb

 [FOLDER] docs/                     # Documentation
    INSTALLATION.md         # Installation guide
    PROJECT_SUMMARY.md      # Technical overview
    QUICK_REFERENCE.md      # Command cheat sheet
    FILE_INDEX.md           # File navigation guide
    PROJECT_STRUCTURE.md    # This file

 [FOLDER] configs/                  # Configuration files
    dataset_profile.json    # Dataset configurations

 [FOLDER] tests/                    # Unit tests
    __init__.py

 [FOLDER] data/                     # Data directory (gitignored)
    vimd/
    bud500/
    lsvsc/
    vlsp2020/
    vietmed/

 [FOLDER] results/                  # Output directory (gitignored)
    evaluation_results_*.csv
    evaluation_summary.txt
    plots/

 [FILE] setup.py                  # Package installation script
 [FILE] pyproject.toml            # Modern Python packaging config
 [FILE] MANIFEST.in               # Package manifest
 [FILE] requirements.txt          # Python dependencies
 [FILE] README.md                 # Project overview
 [FILE] .gitignore                # Git ignore rules
 [FILE] CLAUDE.md                 # AI assistant instructions
```

## [TARGET] Design Principles

### 1. **Separation of Concerns**
- **src/**: Reusable library code
- **scripts/**: Executable entry points
- **notebooks/**: Interactive examples
- **docs/**: Documentation
- **tests/**: Unit tests

### 2. **Standard Python Package Layout**
Following [Python Packaging User Guide](https://packaging.python.org/) best practices:
- `src/` layout for better isolation
- `pyproject.toml` for modern packaging
- `setup.py` for backward compatibility

### 3. **Importability**
The package can be imported directly after installation:
```python
from src.metrics import ASRMetrics
from src.dataset_loader import DatasetManager
from src.model_evaluator import ModelEvaluator
from src.visualization import ASRVisualizer
```

## [PACKAGE] Module Descriptions

### Core Modules (`src/`)

#### `metrics.py` (277 lines)
- **Purpose**: Calculate all ASR evaluation metrics
- **Key Classes**: `ASRMetrics`, `RTFTimer`
- **Metrics**: WER, CER, MER, WIL, WIP, SER, RTF
- **Standalone**: Can be used independently

#### `dataset_loader.py` (412 lines)
- **Purpose**: Load and manage Vietnamese datasets
- **Key Classes**: `DatasetManager`, `AudioSample`, various dataset loaders
- **Supports**: ViMD, BUD500, LSVSC, VLSP2020, VietMed, HuggingFace datasets
- **Features**: Auto train/val/test splitting, normalization, EDA

#### `model_evaluator.py` (380 lines)
- **Purpose**: Load and evaluate ASR models
- **Key Classes**: `ModelEvaluator`, `ModelFactory`, model implementations
- **Supports**: PhoWhisper, Whisper, Wav2Vec2-XLSR, custom models
- **Features**: Auto GPU detection, graceful fallback

#### `visualization.py` (445 lines)
- **Purpose**: Create comprehensive visualizations
- **Key Class**: `ASRVisualizer`
- **Plots**: WER/CER comparison, heatmaps, radar charts, RTF analysis
- **Output**: High-quality PNG plots (300 DPI)

### Executable Scripts (`scripts/`)

#### `main_evaluation.py` (470 lines)
- **Purpose**: End-to-end evaluation pipeline orchestrator
- **Usage**: `python scripts/main_evaluation.py [options]`
- **Features**: Complete automated workflow
- **Command**: `vietnamese-asr-benchmark-eval` (after installation)

#### `demo.py` (85 lines)
- **Purpose**: Quick demonstration with synthetic data
- **Usage**: `python scripts/demo.py`
- **Command**: `vietnamese-asr-benchmark-demo` (after installation)

#### `fetch_datasets.py` (~350 lines)
- **Purpose**: Fetch datasets from HuggingFace Hub
- **Usage**: `python scripts/fetch_datasets.py [options]`
- **Command**: `vietnamese-asr-benchmark-fetch` (after installation)

#### `check_setup.py` (175 lines)
- **Purpose**: Verify installation and dependencies
- **Usage**: `python scripts/check_setup.py`
- **Command**: `vietnamese-asr-benchmark-check` (after installation)

## [LAUNCH] Usage Patterns

### Pattern 1: As an Installed Package

```bash
# Install the package
pip install -e .

# Use command-line tools
vietnamese-asr-benchmark-demo
vietnamese-asr-benchmark-eval --models phowhisper-small --max-samples 10

# Import in Python
python
>>> from src.metrics import ASRMetrics
>>> calculator = ASRMetrics()
```

### Pattern 2: Direct Script Execution

```bash
# Run scripts directly (no installation needed)
python scripts/demo.py
python scripts/main_evaluation.py --max-samples 10

# Import modules directly (add to PYTHONPATH)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python
>>> from metrics import ASRMetrics
```

### Pattern 3: Jupyter Notebooks

```bash
# Local execution
jupyter notebook notebooks/custom_analysis_example.ipynb

# Google Colab
# Upload notebook to Colab - it auto-configures!
```

## [NOTE] Import Statements

```python
# Clean imports
from src.metrics import RTFTimer, format_metrics_report
from src.dataset_loader import AudioSample
```


## [CONFIG] Configuration Files

### `setup.py`
- Traditional Python package setup
- Defines package metadata and dependencies
- Registers console scripts (CLI commands)

### `pyproject.toml`
- Modern Python packaging standard (PEP 517/518)
- Tool configurations (black, isort, pytest, mypy)
- Preferred for new projects

### `MANIFEST.in`
- Specifies files to include/exclude in distribution
- Includes docs, notebooks, configs
- Excludes data, results, cache

### `requirements.txt`
- Core dependencies list
- Used by `pip install -r requirements.txt`
- Also read by `setup.py`

## [CHART] Data Flow

```

                    User Entry Points                        

  CLI Commands    Scripts    Notebooks    Python Import  

                                              
        
                          
                
                   src/    
                  (Core Library)   
                
                          
        
                                          
             
     metrics      dataset          model    
                   _loader        evaluator 
             
                                         
        
                         
                  
                  visualization 
                  
                         
                    
                     Results  
                      & Plots 
                    
```

## [INFO] Best Practices

### For Users
1. **Install the package**: Use `pip install -e .` for development
2. **Use CLI commands**: `vietnamese-asr-benchmark-demo`, `vietnamese-asr-benchmark-eval`, etc.
3. **Explore notebooks**: Start with `custom_analysis_example.ipynb`
4. **Read documentation**: Check `docs/` for guides

### For Developers
1. **Edit in `src/`**: All library code goes in `src/`
2. **Add scripts to `scripts/`**: Keep executables separate
3. **Write tests in `tests/`**: Mirror the `src/` structure
4. **Update `__init__.py`**: Export new public APIs
5. **Document**: Update relevant `.md` files

### For Contributors
1. **Follow structure**: Respect the directory organization
2. **Update imports**: Keep import statements consistent
3. **Test thoroughly**: Run `pytest` before committing
4. **Format code**: Use `black` and `isort`
5. **Update docs**: Keep documentation in sync


## [DOCS] Related Documentation

- [INSTALLATION.md](INSTALLATION.md) - How to install
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command cheat sheet
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Technical deep-dive
- [FILE_INDEX.md](FILE_INDEX.md) - File navigation
- [README.md](../README.md) - Project overview

## [BUILD] Future Enhancements

- [ ] Docker support for containerized deployment
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Comprehensive unit test coverage
- [ ] API documentation with Sphinx
- [ ] PyPI package publishing
- [ ] Benchmark suite for performance testing

---

**Version**: 1.0.0
**Last Updated**: 2025-10-31
**Status**: Production Ready [OK]
