"""
Shared utilities for Vietnamese ASR evaluation notebooks.
Handles environment detection (local vs Colab) and common setup.

Available Functions:
    - detect_environment(): Detect if running in Colab or local
    - setup_paths(): Setup project paths for both environments
    - load_config(): Load configuration files from configs/ directory
    - list_available_configs(): List all available config files
    - install_dependencies(): Install required packages
    - print_environment_info(): Display environment information
    - create_evaluation_summary(): Generate evaluation summary string
    - ReportGenerator: Class for generating evaluation reports
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import json
from datetime import datetime


def detect_environment() -> str:
    """
    Detect if running in Google Colab or local environment.

    Returns:
        str: 'colab' or 'local'
    """
    try:
        import google.colab
        return 'colab'
    except ImportError:
        return 'local'


def setup_paths() -> Dict[str, Path]:
    """
    Setup paths for local and Colab environments.

    Returns:
        Dict[str, Path]: Dictionary containing project_root, data_dir, output_dir, reports_dir
    """
    env = detect_environment()

    if env == 'colab':
        # Colab environment
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)

        # Adjust these paths according to your Google Drive structure
        project_root = Path('/content/drive/MyDrive/vietnamese_asr_benchmark')

        # If project doesn't exist in Drive, clone from GitHub
        if not project_root.exists():
            print("[INFO] Cloning repository to Google Drive...")
            import subprocess
            subprocess.run([
                'git', 'clone',
                'https://github.com/YOUR_USERNAME/vietnamese_asr_benchmark.git',
                str(project_root)
            ])
    else:
        # Local environment - find project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent

    paths = {
        'project_root': project_root,
        'data_dir': project_root / 'data',
        'output_dir': project_root / 'results',
        'reports_dir': project_root / 'docs' / 'reports',
        'plots_dir': project_root / 'results' / 'plots',
        'config_file': project_root / 'configs' / 'dataset_profile.json',
    }

    # Create directories if they don't exist
    for key, path in paths.items():
        if key != 'project_root' and not os.path.exists(path):
            path.mkdir(parents=True, exist_ok=True)

    # Add project root to Python path (NOT src directory)
    # This allows imports like: from src.dataset_loader import ...
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return paths


def load_config(config_name: str, config_dir: Path = None) -> Dict[str, Any]:
    """
    Load a configuration file from the configs directory.

    Args:
        config_name: Name of the config file (with or without .json extension)
        config_dir: Path to config directory. If None, uses default from setup_paths()

    Returns:
        Dict[str, Any]: Parsed configuration data

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON

    Example:
        >>> config = load_config('dataset_profile')
        >>> print(config.keys())
        >>> # Or with full filename:
        >>> config = load_config('dataset_profile.json')
    """
    # Get config directory
    if config_dir is None:
        paths = setup_paths()
        config_dir = paths['project_root'] / 'configs'
    else:
        config_dir = Path(config_dir)

    # Add .json extension if not present
    if not config_name.endswith('.json'):
        config_name = f"{config_name}.json"

    config_path = config_dir / config_name

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Available configs: {list(config_dir.glob('*.json'))}"
        )

    # Load and parse JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"[OK] Loaded config: {config_path.name}")
        return config
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in config file: {config_path}",
            e.doc, e.pos
        )


def list_available_configs(config_dir: Path = None) -> list:
    """
    List all available configuration files in the configs directory.

    Args:
        config_dir: Path to config directory. If None, uses default from setup_paths()

    Returns:
        list: List of available config file names (without .json extension)

    Example:
        >>> configs = list_available_configs()
        >>> print(configs)
        ['dataset_profile', 'model_configs', ...]
    """
    # Get config directory
    if config_dir is None:
        paths = setup_paths()
        config_dir = paths['project_root'] / 'configs'
    else:
        config_dir = Path(config_dir)

    # List all JSON files
    config_files = list(config_dir.glob('*.json'))
    config_names = [f.stem for f in config_files]

    print(f"[INFO] Found {len(config_names)} config file(s) in {config_dir}:")
    for name in config_names:
        print(f"  - {name}")

    return config_names


def install_dependencies(env: str = None):
    """
    Install required dependencies based on environment.

    Args:
        env: Environment type ('colab' or 'local'). Auto-detected if None.
    """
    if env is None:
        env = detect_environment()

    if env == 'colab':
        print("[INFO] Installing dependencies for Colab...")
        import subprocess

        # Install system dependencies
        subprocess.run(['apt-get', 'update', '-qq'], check=False)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'ffmpeg'], check=False)

        # Install Python packages
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            'transformers', 'torch', 'torchaudio', 'librosa',
            'jiwer', 'soundfile', 'datasets', 'accelerate'
        ], check=False)

        print("[OK] Dependencies installed successfully")
    else:
        print("[INFO] Local environment detected. Ensure requirements.txt is installed:")
        print("      pip install -r requirements.txt")


class ReportGenerator:
    """Generate systematic evaluation reports in Markdown format."""

    def __init__(self, reports_dir: Path):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_model_report(
        self,
        model_family: str,
        results: Dict[str, Any],
        output_filename: str = None
    ) -> Path:
        """
        Generate a comprehensive model evaluation report.

        Args:
            model_family: Model family name (e.g., 'PhoWhisper', 'Whisper')
            results: Dictionary containing evaluation results
            output_filename: Custom output filename (default: auto-generated)

        Returns:
            Path: Path to generated report file
        """
        if output_filename is None:
            output_filename = f"Báo_cáo_{model_family}_{self.timestamp}.md"

        report_path = self.reports_dir / output_filename

        # Extract data from results
        models = results.get('models', [])
        datasets = results.get('datasets', [])
        metrics_summary = results.get('metrics_summary', {})
        best_model = results.get('best_model', {})

        # Generate report content
        content = self._generate_report_content(
            model_family, models, datasets, metrics_summary, best_model, results
        )

        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[OK] Report generated: {report_path}")
        return report_path

    def _generate_report_content(
        self,
        model_family: str,
        models: list,
        datasets: list,
        metrics_summary: dict,
        best_model: dict,
        results: dict
    ) -> str:
        """Generate the Markdown content for the report."""

        lines = [
            f"# Báo cáo Đánh giá Mô hình {model_family}",
            "",
            f"**Ngày tạo**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## [INFO] Tổng quan",
            "",
            f"Báo cáo này trình bày kết quả đánh giá chi tiết các mô hình thuộc họ **{model_family}** "
            f"trên {len(datasets)} bộ dữ liệu tiếng Việt.",
            "",
            "### Các mô hình được đánh giá",
            "",
        ]

        for i, model in enumerate(models, 1):
            lines.append(f"{i}. `{model}`")

        lines.extend([
            "",
            "### Các bộ dữ liệu",
            "",
        ])

        for i, dataset in enumerate(datasets, 1):
            lines.append(f"{i}. **{dataset}**")

        lines.extend([
            "",
            "### Các chỉ số đánh giá",
            "",
            "| Chỉ số | Mô tả | Tốt hơn |",
            "|--------|-------|---------|",
            "| **WER** | Word Error Rate - Tỷ lệ lỗi từ | Thấp hơn |",
            "| **CER** | Character Error Rate - Tỷ lệ lỗi ký tự | Thấp hơn |",
            "| **MER** | Match Error Rate - Tỷ lệ lỗi khớp | Thấp hơn |",
            "| **WIL** | Word Information Lost - Mất thông tin từ | Thấp hơn |",
            "| **WIP** | Word Information Preserved - Bảo toàn thông tin từ | Cao hơn |",
            "| **SER** | Sentence Error Rate - Tỷ lệ lỗi câu | Thấp hơn |",
            "| **RTF** | Real-Time Factor - Hệ số thời gian thực | Thấp hơn |",
            "",
            "---",
            "",
            "## [CHART] Kết quả Tổng hợp",
            "",
        ])

        # Add metrics summary table
        if metrics_summary:
            lines.extend(self._generate_metrics_table(metrics_summary))

        lines.extend([
            "",
            "---",
            "",
            "## [TARGET] Mô hình Tốt nhất",
            "",
        ])

        if best_model:
            lines.append(f"**Mô hình**: `{best_model.get('model_name', 'N/A')}`")
            lines.append(f"**Dataset**: {best_model.get('dataset', 'N/A')}")
            lines.append(f"**WER**: {best_model.get('WER', 'N/A'):.4f}")
            lines.append(f"**CER**: {best_model.get('CER', 'N/A'):.4f}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## [LIST] Phân tích Chi tiết theo Dataset",
            "",
        ])

        # Add per-dataset analysis
        for dataset in datasets:
            lines.extend(self._generate_dataset_section(dataset, results))

        lines.extend([
            "---",
            "",
            "## [NOTE] Kết luận",
            "",
            "### Ưu điểm",
            "",
            "- [Điền ưu điểm dựa trên kết quả]",
            "",
            "### Nhược điểm",
            "",
            "- [Điền nhược điểm dựa trên kết quả]",
            "",
            "### Khuyến nghị",
            "",
            "- [Điền khuyến nghị sử dụng]",
            "",
            "---",
            "",
            f"**Tạo bởi**: Vietnamese ASR Evaluation Pipeline v1.0.0  ",
            f"**Timestamp**: {self.timestamp}",
        ])

        return "\n".join(lines)

    def _generate_metrics_table(self, metrics_summary: dict) -> list:
        """Generate a formatted metrics table."""
        lines = [
            "| Mô hình | Dataset | WER | CER | MER | WIL | WIP | SER | RTF |",
            "|---------|---------|-----|-----|-----|-----|-----|-----|-----|",
        ]

        for key, metrics in metrics_summary.items():
            model_name = metrics.get('model', 'N/A')
            dataset = metrics.get('dataset', 'N/A')
            wer = metrics.get('WER', 0)
            cer = metrics.get('CER', 0)
            mer = metrics.get('MER', 0)
            wil = metrics.get('WIL', 0)
            wip = metrics.get('WIP', 0)
            ser = metrics.get('SER', 0)
            rtf = metrics.get('RTF', 0)

            lines.append(
                f"| {model_name} | {dataset} | {wer:.4f} | {cer:.4f} | {mer:.4f} | "
                f"{wil:.4f} | {wip:.4f} | {ser:.4f} | {rtf:.4f} |"
            )

        return lines

    def _generate_dataset_section(self, dataset: str, results: dict) -> list:
        """Generate a section for a specific dataset."""
        lines = [
            f"### {dataset}",
            "",
            "[Phân tích chi tiết cho dataset này]",
            "",
        ]
        return lines

    def save_results_json(self, results: Dict[str, Any], filename: str = None) -> Path:
        """
        Save raw results to JSON file.

        Args:
            results: Results dictionary
            filename: Custom filename (default: auto-generated)

        Returns:
            Path: Path to saved JSON file
        """
        if filename is None:
            filename = f"results_{self.timestamp}.json"

        json_path = self.reports_dir / filename

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"[OK] Results saved to JSON: {json_path}")
        return json_path


def print_environment_info():
    """Print current environment information."""
    env = detect_environment()
    print(f"[INFO] Environment: {env}")
    print(f"[INFO] Python version: {sys.version.split()[0]}")
    print(f"[INFO] Python executable: {sys.executable}")

    # Check key packages
    packages = ['torch', 'transformers', 'librosa', 'jiwer']
    print("\n[INFO] Package versions:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  - {pkg}: {version}")
        except ImportError:
            print(f"  - {pkg}: [WARNING] Not installed")

    # Check CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n[INFO] CUDA available: {cuda_available}")
        if cuda_available:
            print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\n[WARNING] PyTorch not installed, cannot check CUDA")


def create_evaluation_summary(
    model_family: str,
    models_tested: list,
    datasets_tested: list,
    total_samples: int,
    total_time: float
) -> str:
    """
    Create a quick summary string for evaluation.

    Args:
        model_family: Model family name
        models_tested: List of model names tested
        datasets_tested: List of dataset names tested
        total_samples: Total number of samples processed
        total_time: Total time taken (seconds)

    Returns:
        str: Formatted summary string
    """
    summary = f"""
{'='*60}
{model_family} EVALUATION SUMMARY
{'='*60}
Models tested: {len(models_tested)}
Datasets tested: {len(datasets_tested)}
Total samples: {total_samples}
Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)
Average time per sample: {total_time/total_samples:.2f}s
{'='*60}
"""
    return summary
