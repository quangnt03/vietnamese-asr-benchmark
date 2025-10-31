#!/usr/bin/env python3
"""
Setup script for Vietnamese ASR Evaluation Framework

Installation:
    pip install -e .           # Development mode
    pip install .              # Standard installation
    pip install -e ".[dev]"    # With development dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'librosa>=0.10.0',
        'soundfile>=0.12.0',
        'jiwer>=3.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'tqdm>=4.65.0',
        'accelerate>=0.20.0',
        'huggingface_hub',
        'datasets',
        'kaggle',
    ]

# Development dependencies
dev_requirements = [
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
    'black>=22.0.0',
    'flake8>=4.0.0',
    'isort>=5.10.0',
    'mypy>=0.950',
    'jupyter>=1.0.0',
    'ipython>=8.0.0',
]

setup(
    name="vietnamese-asr-benchmark",
    version="1.0.0",
    author="Vietnamese ASR Evaluation Team",
    author_email="",
    description="A comprehensive framework for evaluating Vietnamese Automatic Speech Recognition models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quangnt03/vietnamese-asr-benchmark",
    project_urls={
        "Bug Tracker": "https://github.com/quangnt03/vietnamese-asr-benchmark/issues",
        "Documentation": "https://github.com/quangnt03/vietnamese-asr-benchmark#readme",
        "Source Code": "https://github.com/quangnt03/vietnamese-asr-benchmark",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "all": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "vietnamese-asr-benchmark-eval=scripts.main_evaluation:main",
            "vietnamese-asr-benchmark-demo=scripts.demo:run_demo",
            "vietnamese-asr-benchmark-fetch=scripts.fetch_datasets:main",
            "vietnamese-asr-benchmark-check=scripts.check_setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vietnamese-asr-benchmark": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "speech recognition",
        "ASR",
        "Vietnamese",
        "evaluation",
        "metrics",
        "WER",
        "CER",
        "transformers",
        "whisper",
        "wav2vec2",
    ],
)
