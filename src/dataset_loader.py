"""
Vietnamese ASR Dataset Loader Module

This module handles loading and preprocessing of various Vietnamese ASR datasets:
- ViMD: Vietnamese Multi-Dialect Dataset
- BUD500: Vietnamese Speech Dataset
- LSVSC: Large-Scale Vietnamese Speech Corpus
- VLSP 2020: Vietnamese Language and Speech Processing 2020
- VietMed: Vietnamese Medical Speech Dataset
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf
from dataclasses import dataclass
from tqdm import tqdm
import re

# HuggingFace datasets support
try:
    from datasets import load_dataset, load_from_disk, Audio
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class AudioSample:
    """Data class for audio samples."""
    audio_path: str
    transcription: str
    duration: float
    sample_rate: int
    dataset: str
    split: str  # 'train', 'validation', 'test'
    dialect: Optional[str] = None
    speaker_id: Optional[str] = None
    metadata: Optional[Dict] = None


class VietnameseTextNormalizer:
    """
    Text normalizer for Vietnamese transcriptions.
    Handles common normalization needs for ASR evaluation.
    """
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize Vietnamese text for ASR evaluation.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove punctuation (keeping Vietnamese diacritics)
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', '', text)
        
        # Remove extra whitespace again
        text = ' '.join(text.split())
        
        return text.strip()


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, data_dir: str, cache_dir: str = "./cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = VietnameseTextNormalizer()
        
    def load_dataset(self) -> List[AudioSample]:
        """Load the complete dataset."""
        raise NotImplementedError
    
    def train_test_split(
        self, 
        samples: List[AudioSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[List[AudioSample], List[AudioSample], List[AudioSample]]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            samples: List of audio samples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        np.random.seed(random_seed)
        indices = np.random.permutation(len(samples))
        
        train_end = int(len(samples) * train_ratio)
        val_end = train_end + int(len(samples) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        test_samples = [samples[i] for i in test_indices]
        
        # Update split labels
        for sample in train_samples:
            sample.split = 'train'
        for sample in val_samples:
            sample.split = 'validation'
        for sample in test_samples:
            sample.split = 'test'
        
        return train_samples, val_samples, test_samples
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds."""
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            print(f"Warning: Could not get duration for {audio_path}: {e}")
            return 0.0


class ViMDLoader(DatasetLoader):
    """Loader for Vietnamese Multi-Dialect (ViMD) dataset."""
    
    def load_dataset(self) -> Optional[List[AudioSample]]:
        """
        Load ViMD dataset.
        Expected structure:
        - data_dir/
          - audio/
            - province_name/
              - audio_files.wav
          - transcripts/
            - metadata.csv (with columns: file_id, transcription, province, speaker_id)
        """
        print("Loading ViMD dataset...")
        samples = []
        
        metadata_file = self.data_dir / "transcripts" / "metadata.csv"
        
        if not metadata_file.exists():
            print(f"Warning: ViMD metadata file not found at {metadata_file}")
            return None
        
        df = pd.read_csv(metadata_file)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading ViMD"):
            audio_path = self.data_dir / "audio" / row.get('province', 'unknown') / f"{row['file_id']}.wav"
            
            if not audio_path.exists():
                continue
            
            duration = self.get_audio_duration(str(audio_path))
            transcription = self.normalizer.normalize(row['transcription'])
            
            sample = AudioSample(
                audio_path=str(audio_path),
                transcription=transcription,
                duration=duration,
                sample_rate=16000,
                dataset="ViMD",
                split="unknown",
                dialect=row.get('province', 'unknown'),
                speaker_id=row.get('speaker_id', None),
                metadata={'province': row.get('province', 'unknown')}
            )
            samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from ViMD")
        return samples


class BUD500Loader(DatasetLoader):
    """Loader for BUD500 dataset."""

    def load_dataset(self) -> List[AudioSample]:
        """Load BUD500 dataset."""
        print("Loading BUD500 dataset...")

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"BUD500 data directory not found at {self.data_dir}\n"
                f"Please download the dataset or check the path."
            )

        # TODO: Implement actual BUD500 loading logic here
        raise NotImplementedError(
            f"BUD500 dataset loader not yet implemented.\n"
            f"Please implement the loading logic in BUD500Loader.load_dataset()"
        )


class LSVSCLoader(DatasetLoader):
    """Loader for Large-Scale Vietnamese Speech Corpus (LSVSC)."""

    def load_dataset(self) -> List[AudioSample]:
        """Load LSVSC dataset."""
        print("Loading LSVSC dataset...")

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"LSVSC data directory not found at {self.data_dir}\n"
                f"Please download the dataset or check the path."
            )

        # TODO: Implement actual LSVSC loading logic here
        raise NotImplementedError(
            f"LSVSC dataset loader not yet implemented.\n"
            f"Please implement the loading logic in LSVSCLoader.load_dataset()"
        )


class VLSP2020Loader(DatasetLoader):
    """Loader for VLSP 2020 dataset."""

    def load_dataset(self) -> List[AudioSample]:
        """
        Load VLSP 2020 dataset.
        Expected structure follows VLSP competition format.
        """
        print("Loading VLSP 2020 dataset...")

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"VLSP2020 data directory not found at {self.data_dir}\n"
                f"Please download the dataset or check the path."
            )

        # TODO: Implement actual VLSP2020 loading logic here
        raise NotImplementedError(
            f"VLSP2020 dataset loader not yet implemented.\n"
            f"Please implement the loading logic in VLSP2020Loader.load_dataset()"
        )


class VietMedLoader(DatasetLoader):
    """Loader for VietMed dataset."""

    def load_dataset(self) -> List[AudioSample]:
        """Load VietMed dataset."""
        print("Loading VietMed dataset...")

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"VietMed data directory not found at {self.data_dir}\n"
                f"Please download the dataset or check the path."
            )

        # TODO: Implement actual VietMed loading logic here
        raise NotImplementedError(
            f"VietMed dataset loader not yet implemented.\n"
            f"Please implement the loading logic in VietMedLoader.load_dataset()"
        )


class HuggingFaceDatasetLoader(DatasetLoader):
    """
    Loader for datasets from HuggingFace Hub.
    Supports Common Voice, VIVOS, FLEURS, FOSD, etc.
    """
    
    # Mapping of dataset configurations
    DATASET_CONFIGS = {
        'common_voice_vi': {
            'id': 'mozilla-foundation/common_voice_13_0',
            'config': 'vi',
            'audio_column': 'audio',
            'text_column': 'sentence',
            'splits': ['train', 'validation', 'test']
        },
        'vivos': {
            'id': 'vivos',
            'config': None,
            'audio_column': 'audio',
            'text_column': 'transcript',
            'splits': ['train', 'test']
        },
        'fleurs_vi': {
            'id': 'google/fleurs',
            'config': 'vi_vn',
            'audio_column': 'audio',
            'text_column': 'transcription',
            'splits': ['train', 'validation', 'test']
        },
        'fosd': {
            'id': 'doof-ferb/FOSD',
            'config': None,
            'audio_column': 'audio',
            'text_column': 'text',
            'splits': ['train']
        }
    }
    
    def __init__(
        self,
        dataset_key: str,
        data_dir: str,
        cache_dir: str = "./cache",
        from_disk: bool = False
    ):
        """
        Initialize HuggingFace dataset loader.
        
        Args:
            dataset_key: Key identifying the dataset (e.g., 'common_voice_vi')
            data_dir: Directory where dataset is saved
            cache_dir: Cache directory
            from_disk: Whether to load from disk (already downloaded)
        """
        super().__init__(data_dir, cache_dir)
        
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library required for HuggingFace datasets. "
                "Install with: pip install datasets"
            )
        
        if dataset_key not in self.DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_key}. "
                f"Available: {list(self.DATASET_CONFIGS.keys())}"
            )
        
        self.dataset_key = dataset_key
        self.config = self.DATASET_CONFIGS[dataset_key]
        self.from_disk = from_disk
        self.hf_dataset = None
    
    def load_dataset(self) -> List[AudioSample]:
        """Load dataset from HuggingFace."""
        print(f"Loading {self.dataset_key} from HuggingFace...")
        
        samples = []
        
        # Load from disk or download
        if self.from_disk and self.data_dir.exists():
            print(f"Loading from disk: {self.data_dir}")
            samples = self._load_from_disk()
        else:
            print(f"Downloading from HuggingFace Hub...")
            samples = self._download_and_load()
        
        print(f"Loaded {len(samples)} samples from {self.dataset_key}")
        return samples
    
    def _load_from_disk(self) -> List[AudioSample]:
        """Load dataset from disk."""
        samples = []
        
        # Check for metadata
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            splits = metadata.get('splits', ['train'])
        else:
            splits = self.config['splits']
        
        for split in splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                continue
            
            try:
                dataset = load_from_disk(str(split_dir))
                samples.extend(self._convert_to_samples(dataset, split))
            except Exception as e:
                print(f"Warning: Could not load {split} from disk: {e}")
        
        return samples
    
    def _download_and_load(self) -> List[AudioSample]:
        """Download dataset from HuggingFace Hub."""
        samples = []
        
        for split in self.config['splits']:
            try:
                # Load dataset
                if self.config['config']:
                    dataset = load_dataset(
                        self.config['id'],
                        self.config['config'],
                        split=split,
                        cache_dir=str(self.cache_dir)
                    )
                else:
                    dataset = load_dataset(
                        self.config['id'],
                        split=split,
                        cache_dir=str(self.cache_dir)
                    )
                
                # Cast audio to 16kHz
                if self.config['audio_column'] in dataset.column_names:
                    dataset = dataset.cast_column(
                        self.config['audio_column'],
                        Audio(sampling_rate=16000)
                    )
                
                samples.extend(self._convert_to_samples(dataset, split))
                
            except Exception as e:
                print(f"Warning: Could not download {split}: {e}")
        
        return samples
    
    def _convert_to_samples(self, dataset, split: str) -> List[AudioSample]:
        """Convert HuggingFace dataset to AudioSample objects."""
        samples = []
        
        audio_col = self.config['audio_column']
        text_col = self.config['text_column']
        
        for idx, item in enumerate(tqdm(dataset, desc=f"Converting {split}")):
            try:
                # Get audio information
                audio_data = item[audio_col]
                
                # Audio can be dict or array
                if isinstance(audio_data, dict):
                    audio_array = audio_data['array']
                    sample_rate = audio_data['sampling_rate']
                    duration = len(audio_array) / sample_rate
                    
                    # Create temporary audio path identifier
                    audio_path = f"{self.dataset_key}_{split}_{idx}"
                else:
                    # Fallback
                    audio_array = audio_data
                    sample_rate = 16000
                    duration = len(audio_array) / sample_rate if len(audio_array) > 0 else 0
                    audio_path = f"{self.dataset_key}_{split}_{idx}"
                
                # Get transcription
                transcription = item[text_col]
                if transcription:
                    transcription = self.normalizer.normalize(str(transcription))
                else:
                    continue  # Skip samples without transcription
                
                # Create sample
                sample = AudioSample(
                    audio_path=audio_path,
                    transcription=transcription,
                    duration=duration,
                    sample_rate=sample_rate,
                    dataset=self.dataset_key.upper(),
                    split=split,
                    metadata={
                        'hf_index': idx,
                        'hf_dataset': self.config['id'],
                        'audio_array': audio_array  # Store for later use
                    }
                )
                
                samples.append(sample)
                
            except Exception as e:
                print(f"Warning: Could not process sample {idx}: {e}")
                continue
        
        return samples


class DatasetManager:
    """
    Manager class to handle multiple datasets.
    Supports both HuggingFace datasets (default) and legacy local loaders.
    """

    def __init__(
        self,
        config_file: str,
        base_data_dir: str = Path(__file__).parent.parent / "data",
        use_huggingface: bool = True,
        cache_dir: str = None
    ):
        """
        Initialize dataset manager.

        Args:
            base_data_dir: Base directory containing all datasets
            config_file: Path to JSON file with HuggingFace dataset configurations
            use_huggingface: Whether to use HuggingFace datasets (default: True)
            cache_dir: Cache directory for HuggingFace downloads (default: base_data_dir/huggingface_cache)
        """
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.use_huggingface = use_huggingface and HF_AVAILABLE

        # Set cache directory
        if cache_dir is None:
            self.cache_dir = self.base_data_dir / "huggingface_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load HuggingFace dataset configurations from JSON
        self.config_file = Path(config_file)
        self.hf_configs = {}
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.hf_configs = json.load(f)
                print(f"[OK] Loaded {len(self.hf_configs)} dataset configs from {config_file}")
        elif use_huggingface:
            print(f"[WARNING] HuggingFace config file not found: {config_file}")
            print(f"[WARNING] Falling back to legacy local loaders")
            self.use_huggingface = False

        # Legacy local dataset loaders mapping (kept for backward compatibility)
        self.loaders = {
            'ViMD': ViMDLoader,
            'BUD500': BUD500Loader,
            'LSVSC': LSVSCLoader,
            'VLSP2020': VLSP2020Loader,
            'VietMed': VietMedLoader
        }

    def load_dataset(
        self,
        dataset_name: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        respect_predefined_splits: bool = True
    ) -> Dict[str, List[AudioSample]]:
        """
        Load a single dataset and split into train/val/test.

        Args:
            dataset_name: Name of dataset (e.g., 'ViMD', 'BUD500', etc.)
            train_ratio: Ratio for training set (ignored if respect_predefined_splits=True)
            val_ratio: Ratio for validation set (ignored if respect_predefined_splits=True)
            test_ratio: Ratio for test set (ignored if respect_predefined_splits=True)
            random_seed: Random seed for reproducibility
            respect_predefined_splits: If True, use predefined HuggingFace splits instead of re-splitting

        Returns:
            Dictionary with keys 'train', 'val', 'test' containing sample lists
        """
        # Normalize dataset name for lookup
        dataset_key = dataset_name.lower()

        # Debug: Check HuggingFace availability
        print(f"[DEBUG] Loading dataset: {dataset_key}")
        print(f"[DEBUG] use_huggingface: {self.use_huggingface}")
        print(f"[DEBUG] Available HF configs: {list(self.hf_configs.keys())}")
        print(f"[DEBUG] dataset_key in hf_configs: {dataset_key in self.hf_configs}")
        print(f"[DEBUG] loaders: {self.hf_configs}")
        # Try HuggingFace first if enabled
        if self.use_huggingface and dataset_key in self.hf_configs:
            print(f"[DEBUG] Using HuggingFace loader for {dataset_key}")
            return self._load_from_huggingface(
                dataset_key,
                respect_predefined_splits,
                train_ratio,
                val_ratio,
                test_ratio,
                random_seed
            )

        print(f"[DEBUG] Falling back to legacy local loader for {dataset_name}")

        # Fall back to legacy local loaders
        if dataset_name not in self.loaders:
            available_hf = list(self.hf_configs.keys()) if self.hf_configs else []
            available_local = list(self.loaders.keys())
            raise ValueError(
                f"Unknown dataset: {dataset_name}.\n"
                f"Available HuggingFace datasets: {available_hf}\n"
                f"Available local loaders: {available_local}"
            )

        # Construct data directory path
        data_dir = self.base_data_dir / dataset_key

        # Load dataset using legacy loader
        loader_class = self.loaders[dataset_name]
        loader = loader_class(str(data_dir))
        samples = loader.load_dataset()

        if not samples:
            print(f"[WARNING] No samples loaded for {dataset_name}")
            return {'train': [], 'val': [], 'test': []}

        # Split dataset
        train_samples, val_samples, test_samples = loader.train_test_split(
            samples, train_ratio, val_ratio, test_ratio, random_seed
        )

        # Return with 'val' key (not 'validation') to match notebook expectations
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }

    def _load_from_huggingface(
        self,
        dataset_key: str,
        respect_predefined_splits: bool,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int
    ) -> Dict[str, List[AudioSample]]:
        """Load dataset from HuggingFace Hub using config file."""
        config = self.hf_configs[dataset_key]
        print(f"Loading {dataset_key} from HuggingFace ({config['description']})...")

        # Check if dataset is already cached on disk
        dataset_cache_dir = self.cache_dir / dataset_key
        load_from_disk_cache = dataset_cache_dir.exists()

        samples_by_split = {}

        # Load each split
        for split in config['splits']:
            try:
                if load_from_disk_cache:
                    # Load from local cache
                    split_dir = dataset_cache_dir / split
                    if split_dir.exists():
                        print(f"  Loading {split} from disk cache...")
                        dataset = load_from_disk(str(split_dir))
                    else:
                        continue
                else:
                    # Download from HuggingFace Hub
                    print(f"  Downloading {split} from HuggingFace Hub...")
                    if config['config']:
                        dataset = load_dataset(
                            config['id'],
                            config['config'],
                            split=split,
                            cache_dir=str(self.cache_dir)
                        )
                    else:
                        dataset = load_dataset(
                            config['id'],
                            split=split,
                            cache_dir=str(self.cache_dir)
                        )

                    # Cast audio to 16kHz
                    if config['audio_column'] in dataset.column_names:
                        dataset = dataset.cast_column(
                            config['audio_column'],
                            Audio(sampling_rate=16000)
                        )

                # Convert to AudioSample objects
                samples = self._convert_hf_to_samples(dataset, config, split)
                samples_by_split[split] = samples
                print(f"  [OK] Loaded {len(samples)} samples from {split}")

            except Exception as e:
                print(f"  [WARNING] Failed to load {split}: {e}")
                samples_by_split[split] = []

        # Map splits to standard train/val/test format
        if respect_predefined_splits and samples_by_split:
            return self._map_splits_to_standard(samples_by_split, config['splits'])
        else:
            # Combine all samples and re-split
            all_samples = []
            for samples in samples_by_split.values():
                all_samples.extend(samples)

            if not all_samples:
                print(f"[WARNING] No samples loaded for {dataset_key}")
                return {'train': [], 'val': [], 'test': []}

            # Re-split using provided ratios
            loader = DatasetLoader(str(self.base_data_dir / dataset_key))
            train_samples, val_samples, test_samples = loader.train_test_split(
                all_samples, train_ratio, val_ratio, test_ratio, random_seed
            )

            return {
                'train': train_samples,
                'val': val_samples,
                'test': test_samples
            }

    def _convert_hf_to_samples(
        self,
        dataset,
        config: Dict,
        split_name: str
    ) -> List[AudioSample]:
        """Convert HuggingFace dataset to AudioSample objects."""
        samples = []

        for idx, item in enumerate(dataset):
            try:
                # Get audio data
                audio_data = item[config['audio_column']]
                if isinstance(audio_data, dict) and 'path' in audio_data:
                    audio_path = audio_data['path']
                elif isinstance(audio_data, dict) and 'array' in audio_data:
                    # Audio is in memory, save to temporary location
                    import soundfile as sf
                    import tempfile
                    temp_dir = self.cache_dir / "temp_audio"
                    temp_dir.mkdir(exist_ok=True)
                    audio_path = str(temp_dir / f"{split_name}_{idx}.wav")
                    sf.write(audio_path, audio_data['array'], audio_data['sampling_rate'])
                else:
                    audio_path = str(audio_data)

                # Get transcription
                transcription = item[config['text_column']]

                # Create AudioSample
                sample = AudioSample(
                    file_id=f"{split_name}_{idx}",
                    audio_path=audio_path,
                    transcription=transcription,
                    metadata={'split': split_name, 'source': 'huggingface'}
                )
                samples.append(sample)

            except Exception as e:
                print(f"  [WARNING] Failed to process sample {idx}: {e}")
                continue

        return samples

    def _map_splits_to_standard(
        self,
        samples_by_split: Dict[str, List[AudioSample]],
        available_splits: List[str]
    ) -> Dict[str, List[AudioSample]]:
        """Map HuggingFace splits to standard train/val/test format."""
        result = {'train': [], 'val': [], 'test': []}

        # Map split names
        for split_name, samples in samples_by_split.items():
            if split_name in ['train', 'training']:
                result['train'] = samples
            elif split_name in ['val', 'validation', 'dev', 'valid']:
                result['val'] = samples
            elif split_name in ['test', 'testing']:
                result['test'] = samples
            else:
                # Unknown split, add to train by default
                print(f"[WARNING] Unknown split '{split_name}', adding to train set")
                result['train'].extend(samples)

        return result

    def load_all_datasets(
        self,
        datasets_config: Dict[str, str] = None,
        use_huggingface: bool = False,
        hf_datasets: List[str] = None
    ) -> Dict[str, List[AudioSample]]:
        """
        Load all specified datasets.
        
        Args:
            datasets_config: Dictionary mapping dataset name to data directory
                           If None, uses default structure
            use_huggingface: Whether to use HuggingFace datasets
            hf_datasets: List of HuggingFace dataset keys to load
        
        Returns:
            Dictionary mapping dataset name to list of samples
        """
        # Load HuggingFace datasets if requested
        if use_huggingface and HF_AVAILABLE and hf_datasets:
            print("\n" + "="*70)
            print("Loading datasets from HuggingFace Hub")
            print("="*70 + "\n")
            
            for hf_key in hf_datasets:
                try:
                    # Check if already downloaded to disk
                    hf_cache_dir = self.base_data_dir / 'huggingface_cache' / hf_key
                    from_disk = hf_cache_dir.exists()
                    
                    loader = HuggingFaceDatasetLoader(
                        dataset_key=hf_key,
                        data_dir=str(hf_cache_dir),
                        from_disk=from_disk
                    )
                    self.datasets[hf_key] = loader.load_dataset()
                    print(f"[OK] {hf_key} loaded\n")
                except Exception as e:
                    print(f"[FAILED] Failed to load {hf_key}: {e}\n")
        
        # Load local datasets
        if datasets_config is None:
            datasets_config = {
                'ViMD': str(self.base_data_dir / 'vimd'),
                'BUD500': str(self.base_data_dir / 'bud500'),
                'LSVSC': str(self.base_data_dir / 'lsvsc'),
                'VLSP2020': str(self.base_data_dir / 'vlsp2020'),
                'VietMed': str(self.base_data_dir / 'vietmed')
            }
        
        loaders = {
            'ViMD': ViMDLoader,
            'BUD500': BUD500Loader,
            'LSVSC': LSVSCLoader,
            'VLSP2020': VLSP2020Loader,
            'VietMed': VietMedLoader
        }
        
        if not use_huggingface or not hf_datasets:
            print("\n" + "="*70)
            print("Loading local datasets")
            print("="*70 + "\n")
        
        for dataset_name, data_dir in datasets_config.items():
            if dataset_name in loaders:
                loader = loaders[dataset_name](data_dir)
                self.datasets[dataset_name] = loader.load_dataset()
            else:
                print(f"Warning: Unknown dataset {dataset_name}")
        
        return self.datasets
    
    def get_dataset_statistics(self) -> pd.DataFrame:
        """Get statistics for all loaded datasets."""
        stats = []
        
        for dataset_name, samples in self.datasets.items():
            total_duration = sum(s.duration for s in samples)
            
            stat = {
                'Dataset': dataset_name,
                'Num Samples': len(samples),
                'Total Duration (hours)': total_duration / 3600,
                'Avg Duration (seconds)': total_duration / len(samples) if samples else 0,
                'Num Speakers': len(set(s.speaker_id for s in samples if s.speaker_id)),
                'Num Dialects': len(set(s.dialect for s in samples if s.dialect))
            }
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def prepare_train_test_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Dict[str, List[AudioSample]]]:
        """
        Prepare train/val/test splits for all datasets.
        
        Returns:
            Dictionary with structure: {dataset_name: {'train': [...], 'val': [...], 'test': [...]}}
        """
        splits = {}
        
        for dataset_name, samples in self.datasets.items():
            loader = DatasetLoader(data_dir="dummy")
            train, val, test = loader.train_test_split(
                samples, train_ratio, val_ratio, test_ratio
            )
            splits[dataset_name] = {
                'train': train,
                'validation': val,
                'test': test
            }
            
            print(f"{dataset_name}: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return splits


if __name__ == "__main__":
    # Example usage
    print("Vietnamese ASR Dataset Loader - Example Usage\n")
    
    manager = DatasetManager(base_data_dir="./data")
    datasets = manager.load_all_datasets()
    
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(manager.get_dataset_statistics().to_string(index=False))
    
    print("\n" + "="*60)
    print("Preparing Train/Val/Test Splits:")
    print("="*60)
    splits = manager.prepare_train_test_splits()
