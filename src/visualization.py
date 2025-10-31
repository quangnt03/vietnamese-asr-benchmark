"""
Visualization Module for Vietnamese ASR Evaluation

This module provides comprehensive visualization functions for ASR evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class ASRVisualizer:
    """
    Visualizer for ASR evaluation results.
    """
    
    def __init__(self, output_dir: str = "./results/plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_metric_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = 'wer',
        output_filename: str = None
    ):
        """
        Plot comparison of a specific metric across models and datasets.
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to plot
            output_filename: Optional custom filename
        """
        if metric not in results_df.columns:
            print(f"Warning: Metric {metric} not found in results")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Pivot data for grouped bar chart
        pivot_df = results_df.pivot(index='Dataset', columns='Model', values=metric)
        
        # Create grouped bar chart
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(f'{metric.upper()} Comparison Across Models and Datasets', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()}', fontsize=14, fontweight='bold')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_filename is None:
            output_filename = f'{metric}_comparison.png'
        
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot: {self.output_dir / output_filename}")
        plt.close()
    
    def plot_all_metrics_heatmap(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = None,
        output_filename: str = 'metrics_heatmap.png'
    ):
        """
        Create a heatmap showing all metrics for all model-dataset combinations.
        
        Args:
            results_df: DataFrame with evaluation results
            metrics: List of metrics to include
            output_filename: Output filename
        """
        if metrics is None:
            metrics = ['wer', 'cer', 'mer', 'wil', 'ser']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            print("Warning: No valid metrics found for heatmap")
            return
        
        # Create model-dataset combination labels
        results_df['Model_Dataset'] = results_df['Model'] + '\n' + results_df['Dataset']
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(results_df) * 0.4)))
        
        # Prepare data for heatmap
        heatmap_data = results_df.set_index('Model_Dataset')[available_metrics]
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn_r',
            center=0.15,
            cbar_kws={'label': 'Error Rate'},
            ax=ax
        )
        
        ax.set_title('ASR Evaluation Metrics Heatmap', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Model - Dataset', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot: {self.output_dir / output_filename}")
        plt.close()
    
    def plot_model_performance_radar(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = None,
        output_filename: str = 'performance_radar.png'
    ):
        """
        Create radar charts comparing model performance across metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            metrics: List of metrics to include
            output_filename: Output filename
        """
        if metrics is None:
            metrics = ['wer', 'cer', 'mer', 'wil', 'ser']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if len(available_metrics) < 3:
            print("Warning: Need at least 3 metrics for radar chart")
            return
        
        # Average metrics across datasets for each model
        model_avg = results_df.groupby('Model')[available_metrics].mean()
        
        # Number of variables
        num_vars = len(available_metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_avg)))
        
        for idx, (model_name, values) in enumerate(model_avg.iterrows()):
            values_list = values.tolist()
            values_list += values_list[:1]  # Complete the circle
            
            ax.plot(angles, values_list, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values_list, alpha=0.15, color=colors[idx])
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in available_metrics], fontsize=12)
        
        ax.set_ylim(0, max(model_avg.max()) * 1.2)
        ax.set_title('Model Performance Comparison (Lower is Better)', 
                     fontsize=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot: {self.output_dir / output_filename}")
        plt.close()
    
    def plot_rtf_comparison(
        self,
        results_df: pd.DataFrame,
        output_filename: str = 'rtf_comparison.png'
    ):
        """
        Plot Real-Time Factor comparison.
        
        Args:
            results_df: DataFrame with evaluation results
            output_filename: Output filename
        """
        if 'rtf_mean' not in results_df.columns:
            print("Warning: RTF metric not found in results")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: RTF by Model
        model_rtf = results_df.groupby('Model')['rtf_mean'].mean().sort_values()
        model_rtf.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Real-time threshold')
        ax1.set_xlabel('Real-Time Factor', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Model', fontsize=12, fontweight='bold')
        ax1.set_title('Average RTF by Model', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RTF by Dataset
        dataset_rtf = results_df.groupby('Dataset')['rtf_mean'].mean().sort_values()
        dataset_rtf.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Real-time threshold')
        ax2.set_xlabel('Real-Time Factor', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Dataset', fontsize=12, fontweight='bold')
        ax2.set_title('Average RTF by Dataset', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot: {self.output_dir / output_filename}")
        plt.close()
    
    def plot_error_breakdown(
        self,
        results_df: pd.DataFrame,
        output_filename: str = 'error_breakdown.png'
    ):
        """
        Plot breakdown of error types (insertions, deletions, substitutions).
        
        Args:
            results_df: DataFrame with evaluation results
            output_filename: Output filename
        """
        error_cols = ['total_insertions', 'total_deletions', 'total_substitutions']
        
        if not all(col in results_df.columns for col in error_cols):
            print("Warning: Error breakdown columns not found in results")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Stacked bar chart by model
        model_errors = results_df.groupby('Model')[error_cols].sum()
        model_errors.plot(kind='bar', stacked=True, ax=axes[0], 
                         color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[0].set_title('Error Type Distribution by Model', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
        axes[0].legend(title='Error Type', labels=['Insertions', 'Deletions', 'Substitutions'])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Stacked bar chart by dataset
        dataset_errors = results_df.groupby('Dataset')[error_cols].sum()
        dataset_errors.plot(kind='bar', stacked=True, ax=axes[1],
                           color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[1].set_title('Error Type Distribution by Dataset', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dataset', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
        axes[1].legend(title='Error Type', labels=['Insertions', 'Deletions', 'Substitutions'])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot: {self.output_dir / output_filename}")
        plt.close()
    
    def plot_dataset_statistics(
        self,
        dataset_stats_df: pd.DataFrame,
        output_filename: str = 'dataset_statistics.png'
    ):
        """
        Plot dataset statistics.
        
        Args:
            dataset_stats_df: DataFrame with dataset statistics
            output_filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Number of samples
        axes[0, 0].bar(dataset_stats_df['Dataset'], dataset_stats_df['Num Samples'], 
                       color='steelblue')
        axes[0, 0].set_title('Number of Samples per Dataset', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Dataset', fontsize=10)
        axes[0, 0].set_ylabel('Number of Samples', fontsize=10)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Total duration
        axes[0, 1].bar(dataset_stats_df['Dataset'], dataset_stats_df['Total Duration (hours)'],
                       color='coral')
        axes[0, 1].set_title('Total Duration per Dataset', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Dataset', fontsize=10)
        axes[0, 1].set_ylabel('Duration (hours)', fontsize=10)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Average duration
        axes[1, 0].bar(dataset_stats_df['Dataset'], dataset_stats_df['Avg Duration (seconds)'],
                       color='lightgreen')
        axes[1, 0].set_title('Average Sample Duration', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Dataset', fontsize=10)
        axes[1, 0].set_ylabel('Duration (seconds)', fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Number of speakers and dialects
        x = np.arange(len(dataset_stats_df))
        width = 0.35
        axes[1, 1].bar(x - width/2, dataset_stats_df['Num Speakers'], width, 
                       label='Speakers', color='mediumpurple')
        axes[1, 1].bar(x + width/2, dataset_stats_df['Num Dialects'], width,
                       label='Dialects', color='gold')
        axes[1, 1].set_title('Speakers and Dialects per Dataset', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Dataset', fontsize=10)
        axes[1, 1].set_ylabel('Count', fontsize=10)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(dataset_stats_df['Dataset'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot: {self.output_dir / output_filename}")
        plt.close()
    
    def create_comprehensive_report(
        self,
        results_df: pd.DataFrame,
        dataset_stats_df: pd.DataFrame = None
    ):
        """
        Create all visualization plots for a comprehensive report.
        
        Args:
            results_df: DataFrame with evaluation results
            dataset_stats_df: Optional DataFrame with dataset statistics
        """
        print("\n" + "="*60)
        print("Creating Visualization Plots")
        print("="*60 + "\n")
        
        # Main metric comparisons
        for metric in ['wer', 'cer', 'mer', 'wil', 'ser']:
            if metric in results_df.columns:
                self.plot_metric_comparison(results_df, metric)
        
        # Heatmap
        self.plot_all_metrics_heatmap(results_df)
        
        # Radar chart
        self.plot_model_performance_radar(results_df)
        
        # RTF comparison
        if 'rtf_mean' in results_df.columns:
            self.plot_rtf_comparison(results_df)
        
        # Error breakdown
        self.plot_error_breakdown(results_df)
        
        # Dataset statistics
        if dataset_stats_df is not None:
            self.plot_dataset_statistics(dataset_stats_df)
        
        print(f"\n[OK] All plots saved to: {self.output_dir}\n")


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Vietnamese ASR Visualization Module - Example Usage\n")
    
    # Create synthetic results
    models = ['PhoWhisper-small', 'Whisper-small', 'Wav2Vec2-XLSR']
    datasets = ['ViMD', 'VLSP2020', 'VietMed']
    
    results_data = []
    for model in models:
        for dataset in datasets:
            results_data.append({
                'Model': model,
                'Dataset': dataset,
                'wer': np.random.uniform(0.05, 0.25),
                'cer': np.random.uniform(0.02, 0.15),
                'mer': np.random.uniform(0.04, 0.20),
                'wil': np.random.uniform(0.06, 0.30),
                'ser': np.random.uniform(0.10, 0.40),
                'rtf_mean': np.random.uniform(0.1, 0.5),
                'total_insertions': np.random.randint(10, 100),
                'total_deletions': np.random.randint(10, 100),
                'total_substitutions': np.random.randint(20, 150)
            })
    
    results_df = pd.DataFrame(results_data)
    
    visualizer = ASRVisualizer(output_dir="./example_plots")
    visualizer.create_comprehensive_report(results_df)
