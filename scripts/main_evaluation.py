"""
Vietnamese ASR End-to-End Evaluation Pipeline

This is the main script that orchestrates the complete evaluation workflow:
1. Load datasets (ViMD, BUD500, LSVSC, VLSP2020, VietMed)
2. Perform EDA and preprocessing
3. Evaluate models (PhoWhisper, Wav2Vn, Whisper, Wav2Vec2-XLS-R)
4. Calculate metrics (WER, CER, MER, WIL, WIP, SER, RTF)
5. Generate CSV results and visualizations
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import custom modules
# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import ASRMetrics, RTFTimer, format_metrics_report
from src.dataset_loader import DatasetManager, AudioSample
from src.model_evaluator import ModelEvaluator, ModelFactory
from src.visualization import ASRVisualizer


class VietnameseASREvaluation:
    """
    Main evaluation pipeline for Vietnamese ASR models.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        output_dir: str = "./results",
        models_to_evaluate: list = None,
        datasets_to_evaluate: list = None,
        use_huggingface: bool = False,
        hf_datasets: list = None
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            data_dir: Base directory containing datasets
            output_dir: Directory for output results
            models_to_evaluate: List of model keys to evaluate
            datasets_to_evaluate: List of dataset names to evaluate
            use_huggingface: Whether to use HuggingFace datasets
            hf_datasets: List of HuggingFace dataset keys
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default models if not specified
        if models_to_evaluate is None:
            models_to_evaluate = [
                'phowhisper-small',
                'whisper-small',
                'wav2vec2-xlsr-vietnamese'
            ]
        
        # Default datasets if not specified
        if datasets_to_evaluate is None and not use_huggingface:
            datasets_to_evaluate = ['ViMD', 'BUD500', 'LSVSC', 'VLSP2020', 'VietMed']
        
        # Default HuggingFace datasets if not specified
        if use_huggingface and hf_datasets is None:
            hf_datasets = ['common_voice_vi', 'vivos']
        
        self.models_to_evaluate = models_to_evaluate
        self.datasets_to_evaluate = datasets_to_evaluate
        self.use_huggingface = use_huggingface
        self.hf_datasets = hf_datasets
        
        # Initialize components
        self.dataset_manager = DatasetManager(base_data_dir=str(self.data_dir))
        self.model_evaluator = ModelEvaluator(models_to_evaluate=models_to_evaluate)
        self.metrics_calculator = ASRMetrics()
        self.visualizer = ASRVisualizer(output_dir=str(self.output_dir / "plots"))
        
        # Storage for results
        self.datasets = {}
        self.splits = {}
        self.evaluation_results = []
        
    def step1_load_datasets(self):
        """Step 1: Load and prepare datasets."""
        print("\n" + "="*80)
        print("STEP 1: LOADING DATASETS")
        print("="*80 + "\n")
        
        # Load all datasets
        self.datasets = self.dataset_manager.load_all_datasets(
            use_huggingface=self.use_huggingface,
            hf_datasets=self.hf_datasets
        )
        
        # Get dataset statistics
        stats_df = self.dataset_manager.get_dataset_statistics()
        print("\nDataset Statistics:")
        print(stats_df.to_string(index=False))
        
        # Save statistics
        stats_path = self.output_dir / "dataset_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\n[OK] Dataset statistics saved to: {stats_path}")
        
        return stats_df
    
    def step2_eda_and_preprocessing(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Step 2: Exploratory Data Analysis and Preprocessing."""
        print("\n" + "="*80)
        print("STEP 2: EDA AND PREPROCESSING")
        print("="*80 + "\n")
        
        # Prepare train/val/test splits
        print("Creating train/validation/test splits...")
        print(f"Split ratios: Train={train_ratio}, Val={val_ratio}, Test={1-train_ratio-val_ratio}")
        print()
        
        self.splits = self.dataset_manager.prepare_train_test_splits(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=1-train_ratio-val_ratio
        )
        
        # Generate EDA report
        eda_report = self._generate_eda_report()
        
        eda_path = self.output_dir / "eda_report.json"
        with open(eda_path, 'w', encoding='utf-8') as f:
            json.dump(eda_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] EDA report saved to: {eda_path}")
    
    def _generate_eda_report(self) -> dict:
        """Generate exploratory data analysis report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        for dataset_name, split_data in self.splits.items():
            dataset_report = {
                'train_samples': len(split_data['train']),
                'val_samples': len(split_data['validation']),
                'test_samples': len(split_data['test']),
                'total_samples': sum(len(v) for v in split_data.values()),
                'dialects': {},
                'duration_stats': {}
            }
            
            # Analyze test set
            test_samples = split_data['test']
            
            # Count dialects
            if test_samples and test_samples[0].dialect:
                dialect_counts = {}
                for sample in test_samples:
                    dialect = sample.dialect or 'unknown'
                    dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
                dataset_report['dialects'] = dialect_counts
            
            # Duration statistics
            durations = [s.duration for s in test_samples]
            if durations:
                dataset_report['duration_stats'] = {
                    'mean': float(np.mean(durations)),
                    'std': float(np.std(durations)),
                    'min': float(np.min(durations)),
                    'max': float(np.max(durations)),
                    'total_hours': float(np.sum(durations) / 3600)
                }
            
            report['datasets'][dataset_name] = dataset_report
        
        return report
    
    def step3_load_models(self):
        """Step 3: Load ASR models."""
        print("\n" + "="*80)
        print("STEP 3: LOADING ASR MODELS")
        print("="*80 + "\n")
        
        self.model_evaluator.load_models()
    
    def step4_evaluate_models(self, max_samples_per_dataset: int = None):
        """
        Step 4: Evaluate all models on all datasets.
        
        Args:
            max_samples_per_dataset: Maximum samples to evaluate per dataset (for testing)
        """
        print("\n" + "="*80)
        print("STEP 4: EVALUATING MODELS")
        print("="*80 + "\n")
        
        models = self.model_evaluator.get_loaded_models()
        
        if not models:
            print("[WARNING] No models loaded. Skipping evaluation.")
            return
        
        total_evaluations = len(models) * len(self.splits)
        current_eval = 0
        
        for dataset_name, split_data in self.splits.items():
            # Use test set for evaluation
            test_samples = split_data['test']
            
            # Limit samples if specified
            if max_samples_per_dataset:
                test_samples = test_samples[:max_samples_per_dataset]
            
            print(f"\nEvaluating on {dataset_name} ({len(test_samples)} samples)...")
            
            for model_key, model in models.items():
                current_eval += 1
                print(f"\n[{current_eval}/{total_evaluations}] {model_key} on {dataset_name}")
                print("-" * 60)
                
                result = self._evaluate_single_model(
                    model=model,
                    model_name=model_key,
                    dataset_name=dataset_name,
                    samples=test_samples
                )
                
                self.evaluation_results.append(result)
                
                # Print summary
                print(f"  WER: {result['wer']:.4f}")
                print(f"  CER: {result['cer']:.4f}")
                print(f"  MER: {result['mer']:.4f}")
                if 'rtf_mean' in result:
                    print(f"  RTF: {result['rtf_mean']:.4f}")
    
    def _evaluate_single_model(
        self,
        model,
        model_name: str,
        dataset_name: str,
        samples: list
    ) -> dict:
        """
        Evaluate a single model on a dataset.
        
        Args:
            model: ASR model instance
            model_name: Name of the model
            dataset_name: Name of the dataset
            samples: List of AudioSample objects
            
        Returns:
            Dictionary with evaluation results
        """
        references = []
        hypotheses = []
        audio_durations = []
        processing_times = []
        
        # Transcribe all samples
        for sample in tqdm(samples, desc=f"Transcribing", leave=False):
            # Time the transcription
            with RTFTimer(sample.duration) as timer:
                try:
                    hypothesis = model.transcribe(sample.audio_path)
                except Exception as e:
                    print(f"Error transcribing {sample.audio_path}: {e}")
                    hypothesis = ""
            
            references.append(sample.transcription)
            hypotheses.append(hypothesis)
            audio_durations.append(sample.duration)
            processing_times.append(timer.get_processing_time())
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_batch_metrics(
            references=references,
            hypotheses=hypotheses,
            audio_durations=audio_durations,
            processing_times=processing_times
        )
        
        # Add metadata
        metrics['Model'] = model_name
        metrics['Dataset'] = dataset_name
        metrics['num_samples'] = len(samples)
        metrics['timestamp'] = datetime.now().isoformat()
        
        return metrics
    
    def step5_save_results(self):
        """Step 5: Save evaluation results to CSV."""
        print("\n" + "="*80)
        print("STEP 5: SAVING RESULTS")
        print("="*80 + "\n")
        
        if not self.evaluation_results:
            print("[WARNING] No evaluation results to save.")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.evaluation_results)
        
        # Reorder columns for better readability
        priority_cols = ['Model', 'Dataset', 'wer', 'cer', 'mer', 'wil', 'wip', 'ser']
        other_cols = [col for col in results_df.columns if col not in priority_cols]
        results_df = results_df[priority_cols + other_cols]
        
        # Save to CSV
        csv_path = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"[OK] Results saved to: {csv_path}")
        
        # Also save a summary
        summary = self._generate_summary(results_df)
        summary_path = self.output_dir / "evaluation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"[OK] Summary saved to: {summary_path}")
        
        return results_df
    
    def _generate_summary(self, results_df: pd.DataFrame) -> str:
        """Generate text summary of evaluation results."""
        summary = []
        summary.append("="*80)
        summary.append("VIETNAMESE ASR EVALUATION SUMMARY")
        summary.append("="*80)
        summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall statistics
        summary.append("OVERALL STATISTICS")
        summary.append("-"*80)
        summary.append(f"Models evaluated: {results_df['Model'].nunique()}")
        summary.append(f"Datasets evaluated: {results_df['Dataset'].nunique()}")
        summary.append(f"Total evaluations: {len(results_df)}\n")
        
        # Best performing models
        summary.append("BEST PERFORMING MODELS (by WER)")
        summary.append("-"*80)
        best_per_dataset = results_df.loc[results_df.groupby('Dataset')['wer'].idxmin()]
        for _, row in best_per_dataset.iterrows():
            summary.append(f"{row['Dataset']:15s}: {row['Model']:30s} (WER: {row['wer']:.4f})")
        
        summary.append("\n" + "="*80)
        
        return "\n".join(summary)
    
    def step6_create_visualizations(self, results_df: pd.DataFrame, dataset_stats_df: pd.DataFrame):
        """Step 6: Create visualization plots."""
        print("\n" + "="*80)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        self.visualizer.create_comprehensive_report(results_df, dataset_stats_df)
    
    def run_complete_evaluation(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples_per_dataset: int = None
    ):
        """
        Run the complete end-to-end evaluation pipeline.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            max_samples_per_dataset: Maximum samples per dataset (for testing)
        """
        start_time = time.time()
        
        print("\n" + "="*80)
        print("VIETNAMESE ASR END-TO-END EVALUATION PIPELINE")
        print("="*80)
        print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Models to evaluate: {', '.join(self.models_to_evaluate)}")
        
        if self.use_huggingface:
            print(f"Using HuggingFace datasets: {', '.join(self.hf_datasets or [])}")
        else:
            print(f"Datasets to evaluate: {', '.join(self.datasets_to_evaluate or [])}")
        
        try:
            # Step 1: Load datasets
            dataset_stats_df = self.step1_load_datasets()
            
            # Step 2: EDA and preprocessing
            self.step2_eda_and_preprocessing(train_ratio, val_ratio)
            
            # Step 3: Load models
            self.step3_load_models()
            
            # Step 4: Evaluate models
            self.step4_evaluate_models(max_samples_per_dataset)
            
            # Step 5: Save results
            results_df = self.step5_save_results()
            
            # Step 6: Create visualizations
            if results_df is not None:
                self.step6_create_visualizations(results_df, dataset_stats_df)
            
            # Final summary
            elapsed_time = time.time() - start_time
            print("\n" + "="*80)
            print("EVALUATION COMPLETE")
            print("="*80)
            print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Elapsed Time: {elapsed_time/60:.2f} minutes")
            print(f"\nAll results saved to: {self.output_dir}")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n[FAILED] Evaluation failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vietnamese ASR End-to-End Evaluation Pipeline"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Base directory containing datasets'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory for output results'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['phowhisper-small', 'whisper-small', 'wav2vec2-xlsr-vietnamese'],
        help='Models to evaluate'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['ViMD', 'BUD500', 'LSVSC', 'VLSP2020', 'VietMed'],
        help='Datasets to evaluate'
    )
    
    parser.add_argument(
        '--use-huggingface',
        action='store_true',
        help='Use HuggingFace datasets instead of local datasets'
    )
    
    parser.add_argument(
        '--hf-datasets',
        nargs='+',
        default=None,
        help='HuggingFace dataset keys (e.g., common_voice_vi vivos fleurs_vi)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per dataset (for testing)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\nAvailable models:")
        for model_key in ModelFactory.get_available_models():
            config = ModelFactory.MODEL_CONFIGS[model_key]
            print(f"  {model_key:30s} - {config.name} ({config.model_id})")
        return
    
    # Create and run evaluation
    evaluation = VietnameseASREvaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_to_evaluate=args.models,
        datasets_to_evaluate=args.datasets if not args.use_huggingface else None,
        use_huggingface=args.use_huggingface,
        hf_datasets=args.hf_datasets
    )
    
    evaluation.run_complete_evaluation(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_samples_per_dataset=args.max_samples
    )


if __name__ == "__main__":
    main()
