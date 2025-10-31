#!/usr/bin/env python3
"""
Setup Verification Script

This script checks if all dependencies are properly installed
and the system is ready for Vietnamese ASR evaluation.
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  [FAILED] Python {version.major}.{version.minor}.{version.micro}")
        print("  ! Python 3.8+ is required")
        return False


def check_package(package_name, import_name=None, display_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  [OK] {display_name:20s} ({version})")
        return True
    except ImportError:
        print(f"  [FAILED] {display_name:20s} (not installed)")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available (version {torch.version.cuda})")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ! CUDA not available (CPU-only mode)")
            return False
    except ImportError:
        print("  [FAILED] PyTorch not installed")
        return False


def check_custom_modules():
    """Check if custom modules are accessible."""
    print("\nChecking custom modules...")
    
    modules = {
        'metrics': 'Metrics Module',
        'dataset_loader': 'Dataset Loader',
        'model_evaluator': 'Model Evaluator',
        'visualization': 'Visualization',
        'main_evaluation': 'Main Evaluation'
    }
    
    all_ok = True
    for module_name, display_name in modules.items():
        try:
            importlib.import_module(module_name)
            print(f"  [OK] {display_name}")
        except ImportError as e:
            print(f"  [FAILED] {display_name} - {e}")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check if necessary directories exist or can be created."""
    print("\nChecking directories...")
    
    dirs = ['./data', './results', './cache']
    all_ok = True
    
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  [OK] {dir_path} exists")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  [OK] {dir_path} created")
            except Exception as e:
                print(f"  [FAILED] {dir_path} - cannot create: {e}")
                all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("="*70)
    print("Vietnamese ASR Evaluation Pipeline - Setup Verification")
    print("="*70 + "\n")
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    
    # Core packages
    print("\nChecking core dependencies...")
    checks.append(check_package('numpy'))
    checks.append(check_package('pandas'))
    checks.append(check_package('scipy'))
    
    # Deep learning
    print("\nChecking deep learning frameworks...")
    checks.append(check_package('torch', display_name='PyTorch'))
    checks.append(check_package('transformers'))
    
    # Audio processing
    print("\nChecking audio processing libraries...")
    checks.append(check_package('librosa'))
    checks.append(check_package('soundfile'))
    
    # Metrics
    print("\nChecking evaluation libraries...")
    checks.append(check_package('jiwer', display_name='JiWER'))
    
    # Visualization
    print("\nChecking visualization libraries...")
    checks.append(check_package('matplotlib'))
    checks.append(check_package('seaborn'))
    
    # Utilities
    print("\nChecking utility libraries...")
    checks.append(check_package('tqdm'))
    
    # CUDA
    cuda_ok = check_cuda()
    
    # Custom modules
    modules_ok = check_custom_modules()
    checks.append(modules_ok)
    
    # Directories
    dirs_ok = check_directories()
    checks.append(dirs_ok)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_checks = len(checks)
    passed_checks = sum(checks)
    
    print(f"\nChecks passed: {passed_checks}/{total_checks}")
    
    if all(checks):
        print("\n[OK] All checks passed! System is ready for evaluation.")
        print("\nQuick start:")
        print("  python demo.py                    # Run quick demo")
        print("  python main_evaluation.py --help  # See all options")
        return 0
    else:
        print("\n[WARNING] Some checks failed. Please install missing dependencies:")
        print("\n  pip install -r requirements.txt")
        
        if not cuda_ok:
            print("\nNote: CUDA is not available. The system will run in CPU mode,")
            print("which will be slower but still functional.")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
