"""
Vietnamese ASR Evaluation Metrics Module

This module provides standalone evaluation metrics for Automatic Speech Recognition systems.
All metrics are designed to be reusable across different evaluation scenarios.

Metrics included:
- WER: Word Error Rate
- CER: Character Error Rate
- MER: Match Error Rate
- WIL: Word Information Lost
- WIP: Word Information Preserved
- SER: Sentence Error Rate
- RTF: Real-Time Factor
"""

import time
from typing import List, Dict, Tuple
import numpy as np
from jiwer import wer, cer, mer, wil, wip
from jiwer import process_words, process_characters


class ASRMetrics:
    """
    Comprehensive metrics calculator for ASR evaluation.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate.
        
        Args:
            reference: Ground truth transcription
            hypothesis: ASR system output
            
        Returns:
            WER value (can be > 1.0)
        """
        return wer(reference, hypothesis)
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """
        Calculate Character Error Rate.
        
        Args:
            reference: Ground truth transcription
            hypothesis: ASR system output
            
        Returns:
            CER value (can be > 1.0)
        """
        return cer(reference, hypothesis)
    
    @staticmethod
    def calculate_mer(reference: str, hypothesis: str) -> float:
        """
        Calculate Match Error Rate.
        
        Args:
            reference: Ground truth transcription
            hypothesis: ASR system output
            
        Returns:
            MER value (bounded between 0 and 1)
        """
        return mer(reference, hypothesis)
    
    @staticmethod
    def calculate_wil(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Information Lost.
        
        Args:
            reference: Ground truth transcription
            hypothesis: ASR system output
            
        Returns:
            WIL value
        """
        return wil(reference, hypothesis)
    
    @staticmethod
    def calculate_wip(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Information Preserved.
        
        Args:
            reference: Ground truth transcription
            hypothesis: ASR system output
            
        Returns:
            WIP value
        """
        return wip(reference, hypothesis)
    
    @staticmethod
    def calculate_ser(references: List[str], hypotheses: List[str]) -> float:
        """
        Calculate Sentence Error Rate.
        Percentage of sentences with at least one error.
        
        Args:
            references: List of ground truth transcriptions
            hypotheses: List of ASR system outputs
            
        Returns:
            SER value (percentage of incorrect sentences)
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        incorrect_sentences = sum(
            1 for ref, hyp in zip(references, hypotheses) 
            if ref.strip() != hyp.strip()
        )
        return incorrect_sentences / len(references) if len(references) > 0 else 0.0
    
    @staticmethod
    def calculate_detailed_errors(reference: str, hypothesis: str) -> Dict[str, int]:
        """
        Calculate detailed error counts: insertions, deletions, substitutions, hits.
        
        Args:
            reference: Ground truth transcription
            hypothesis: ASR system output
            
        Returns:
            Dictionary with error counts
        """
        output = process_words(reference, hypothesis)
        
        return {
            'insertions': output.insertions,
            'deletions': output.deletions,
            'substitutions': output.substitutions,
            'hits': output.hits,
            'total_words': len(reference.split())
        }
    
    def calculate_all_metrics(
        self, 
        reference: str, 
        hypothesis: str,
        audio_duration: float = None,
        processing_time: float = None
    ) -> Dict[str, float]:
        """
        Calculate all available metrics for a single utterance.
        
        Args:
            reference: Ground truth transcription
            hypothesis: ASR system output
            audio_duration: Duration of audio in seconds (for RTF)
            processing_time: Time taken to process audio (for RTF)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'wer': self.calculate_wer(reference, hypothesis),
            'cer': self.calculate_cer(reference, hypothesis),
            'mer': self.calculate_mer(reference, hypothesis),
            'wil': self.calculate_wil(reference, hypothesis),
            'wip': self.calculate_wip(reference, hypothesis),
        }
        
        # Add detailed error counts
        errors = self.calculate_detailed_errors(reference, hypothesis)
        metrics.update(errors)
        
        # Calculate RTF if timing information provided
        if audio_duration is not None and processing_time is not None:
            metrics['rtf'] = processing_time / audio_duration if audio_duration > 0 else float('inf')
        
        return metrics
    
    def calculate_batch_metrics(
        self,
        references: List[str],
        hypotheses: List[str],
        audio_durations: List[float] = None,
        processing_times: List[float] = None
    ) -> Dict[str, float]:
        """
        Calculate aggregated metrics across multiple utterances.
        
        Args:
            references: List of ground truth transcriptions
            hypotheses: List of ASR system outputs
            audio_durations: List of audio durations (for RTF)
            processing_times: List of processing times (for RTF)
            
        Returns:
            Dictionary with aggregated metrics
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        # Calculate per-utterance metrics
        all_metrics = []
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            audio_dur = audio_durations[i] if audio_durations else None
            proc_time = processing_times[i] if processing_times else None
            metrics = self.calculate_all_metrics(ref, hyp, audio_dur, proc_time)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {
            'wer': np.mean([m['wer'] for m in all_metrics]),
            'cer': np.mean([m['cer'] for m in all_metrics]),
            'mer': np.mean([m['mer'] for m in all_metrics]),
            'wil': np.mean([m['wil'] for m in all_metrics]),
            'wip': np.mean([m['wip'] for m in all_metrics]),
            'ser': self.calculate_ser(references, hypotheses),
        }
        
        # Add standard deviations
        aggregated.update({
            'wer_std': np.std([m['wer'] for m in all_metrics]),
            'cer_std': np.std([m['cer'] for m in all_metrics]),
            'mer_std': np.std([m['mer'] for m in all_metrics]),
        })
        
        # Aggregate error counts
        total_insertions = sum(m['insertions'] for m in all_metrics)
        total_deletions = sum(m['deletions'] for m in all_metrics)
        total_substitutions = sum(m['substitutions'] for m in all_metrics)
        total_hits = sum(m['hits'] for m in all_metrics)
        total_words = sum(m['total_words'] for m in all_metrics)
        
        aggregated.update({
            'total_insertions': total_insertions,
            'total_deletions': total_deletions,
            'total_substitutions': total_substitutions,
            'total_hits': total_hits,
            'total_words': total_words,
        })
        
        # Calculate average RTF if timing information provided
        if processing_times and audio_durations:
            rtfs = [m.get('rtf') for m in all_metrics if 'rtf' in m and m['rtf'] != float('inf')]
            if rtfs:
                aggregated['rtf_mean'] = np.mean(rtfs)
                aggregated['rtf_std'] = np.std(rtfs)
                aggregated['total_audio_duration'] = sum(audio_durations)
                aggregated['total_processing_time'] = sum(processing_times)
        
        return aggregated


class RTFTimer:
    """
    Context manager for measuring Real-Time Factor.
    """
    
    def __init__(self, audio_duration: float):
        """
        Args:
            audio_duration: Duration of audio in seconds
        """
        self.audio_duration = audio_duration
        self.start_time = None
        self.processing_time = None
        self.rtf = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.processing_time = time.time() - self.start_time
        self.rtf = self.processing_time / self.audio_duration if self.audio_duration > 0 else float('inf')
    
    def get_rtf(self) -> float:
        """Get the calculated RTF value."""
        return self.rtf
    
    def get_processing_time(self) -> float:
        """Get the processing time in seconds."""
        return self.processing_time


def format_metrics_report(metrics: Dict[str, float], title: str = "ASR Metrics Report") -> str:
    """
    Format metrics dictionary into a readable report.
    
    Args:
        metrics: Dictionary of metric values
        title: Title for the report
        
    Returns:
        Formatted string report
    """
    report = f"\n{'='*60}\n{title:^60}\n{'='*60}\n\n"
    
    # Main metrics
    if 'wer' in metrics:
        report += f"Word Error Rate (WER):           {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)\n"
    if 'cer' in metrics:
        report += f"Character Error Rate (CER):      {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)\n"
    if 'mer' in metrics:
        report += f"Match Error Rate (MER):          {metrics['mer']:.4f} ({metrics['mer']*100:.2f}%)\n"
    if 'wil' in metrics:
        report += f"Word Information Lost (WIL):     {metrics['wil']:.4f} ({metrics['wil']*100:.2f}%)\n"
    if 'wip' in metrics:
        report += f"Word Information Preserved (WIP): {metrics['wip']:.4f} ({metrics['wip']*100:.2f}%)\n"
    if 'ser' in metrics:
        report += f"Sentence Error Rate (SER):       {metrics['ser']:.4f} ({metrics['ser']*100:.2f}%)\n"
    
    # RTF metrics
    if 'rtf_mean' in metrics:
        report += f"\nReal-Time Factor (RTF):          {metrics['rtf_mean']:.4f} ± {metrics.get('rtf_std', 0):.4f}\n"
        if 'total_audio_duration' in metrics:
            report += f"Total Audio Duration:            {metrics['total_audio_duration']:.2f}s\n"
            report += f"Total Processing Time:           {metrics['total_processing_time']:.2f}s\n"
    
    # Error details
    if 'total_insertions' in metrics:
        report += f"\n{'Error Breakdown':^60}\n{'-'*60}\n"
        report += f"Total Words:                     {metrics['total_words']}\n"
        report += f"Hits:                            {metrics['total_hits']}\n"
        report += f"Substitutions:                   {metrics['total_substitutions']}\n"
        report += f"Deletions:                       {metrics['total_deletions']}\n"
        report += f"Insertions:                      {metrics['total_insertions']}\n"
    
    report += f"\n{'='*60}\n"
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Vietnamese ASR Metrics Module - Example Usage\n")
    
    # Example Vietnamese text
    reference = "xin chào tôi là người việt nam"
    hypothesis = "xin chào tôi là người việt"
    
    calculator = ASRMetrics()
    
    # Calculate single metrics
    print(f"Reference:  {reference}")
    print(f"Hypothesis: {hypothesis}\n")
    
    metrics = calculator.calculate_all_metrics(reference, hypothesis)
    print(format_metrics_report(metrics, "Single Utterance Metrics"))
    
    # Batch example
    references = [
        "xin chào tôi là người việt nam",
        "hôm nay thời tiết đẹp",
        "tôi yêu tiếng việt"
    ]
    hypotheses = [
        "xin chào tôi là người việt",
        "hôm nay thời tiết đẹp quá",
        "tôi yêu tiếng việt"
    ]
    
    batch_metrics = calculator.calculate_batch_metrics(references, hypotheses)
    print(format_metrics_report(batch_metrics, "Batch Metrics (3 utterances)"))
