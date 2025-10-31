#!/usr/bin/env python3
"""
Quick Demo Script for Vietnamese ASR Evaluation Pipeline

This script runs a demonstration of the evaluation pipeline using synthetic data.
Perfect for testing the system without needing real datasets.
"""

import sys
from pathlib import Path

# Add parent directory and src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main_evaluation import VietnameseASREvaluation


def run_demo():
    """Run a quick demonstration of the evaluation pipeline."""
    
    print("""

                                                                              
            Vietnamese ASR Evaluation Pipeline - Quick Demo                  
                                                                              
  This demo will:                                                             
  1. Generate synthetic Vietnamese datasets                                   
  2. Create train/validation/test splits                                      
  3. Load ASR models (or use mock transcription)                             
  4. Evaluate models with comprehensive metrics                               
  5. Generate CSV results and visualizations                                  
                                                                              
  Note: Real models will be used if available, otherwise mock                
        transcription will be used for demonstration purposes.                
                                                                              

    """)
    
    input("Press Enter to start the demo...")
    
    # Initialize evaluation with demo settings
    evaluation = VietnameseASREvaluation(
        data_dir="./demo_data",
        output_dir="./demo_results",
        models_to_evaluate=[
            'phowhisper-small',
            'whisper-small',
        ],
        datasets_to_evaluate=['ViMD', 'VLSP2020', 'VietMed']
    )
    
    # Run evaluation with limited samples for quick demo
    print("\nRunning evaluation with 10 samples per dataset for quick demo...\n")
    
    evaluation.run_complete_evaluation(
        train_ratio=0.7,
        val_ratio=0.15,
        max_samples_per_dataset=10  # Limit to 10 samples for quick demo
    )
    
    print("""

                                                                              
                            DEMO COMPLETE! [OK]                                  
                                                                              
  Results have been saved to: ./demo_results/                                 
                                                                              
  Check out:                                                                  
  - evaluation_results_*.csv : Detailed metrics                               
  - evaluation_summary.txt   : Human-readable summary                         
  - plots/                   : Visualization charts                           
                                                                              
  To run with your own data:                                                  
  python main_evaluation.py --data-dir /path/to/your/data                    
                                                                              

    """)


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n[FAILED] Demo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[FAILED] Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
