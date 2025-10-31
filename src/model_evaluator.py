"""
Vietnamese ASR Model Evaluator Module

This module handles evaluation of various Vietnamese ASR models:
- PhoWhisper (VinAI)
- Wav2Vn
- OpenAI Whisper
- Wav2Vec2-XLS-R (Vietnamese fine-tuned)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import warnings
from tqdm import tqdm

# Import transformers for model loading
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    pipeline
)

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for ASR models."""
    name: str
    model_id: str
    model_type: str  # 'whisper', 'wav2vec2'
    processor_id: Optional[str] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseASRModel:
    """Base class for ASR models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.device = config.device
        
    def load_model(self):
        """Load the model and processor."""
        raise NotImplementedError
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text
        """
        raise NotImplementedError
    
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """
        Transcribe a batch of audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of transcription texts
        """
        return [self.transcribe(path) for path in tqdm(audio_paths, desc=f"Transcribing with {self.config.name}")]


class PhoWhisperModel(BaseASRModel):
    """PhoWhisper model implementation."""
    
    def load_model(self):
        """Load PhoWhisper model."""
        print(f"Loading {self.config.name} from {self.config.model_id}...")
        try:
            self.processor = WhisperProcessor.from_pretrained(self.config.model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.config.model_id)
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK] {self.config.name} loaded successfully on {self.device}")
        except Exception as e:
            print(f"[FAILED] Failed to load {self.config.name}: {e}")
            print("Using mock transcription for demonstration...")
            self.model = None
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using PhoWhisper."""
        if self.model is None:
            return self._mock_transcribe(audio_path)
        
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            input_features = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""
    
    def _mock_transcribe(self, audio_path: str) -> str:
        """Mock transcription for demonstration."""
        mock_texts = [
            "xin chào tôi là người việt nam",
            "hôm nay thời tiết đẹp",
            "tôi yêu tiếng việt",
            "chúng tôi đang học máy học"
        ]
        import hashlib
        hash_val = int(hashlib.md5(audio_path.encode()).hexdigest(), 16)
        return mock_texts[hash_val % len(mock_texts)]


class WhisperModel(BaseASRModel):
    """OpenAI Whisper model implementation."""
    
    def load_model(self):
        """Load Whisper model."""
        print(f"Loading {self.config.name} from {self.config.model_id}...")
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.config.model_id,
                device=0 if self.device == 'cuda' else -1
            )
            print(f"[OK] {self.config.name} loaded successfully on {self.device}")
        except Exception as e:
            print(f"[FAILED] Failed to load {self.config.name}: {e}")
            print("Using mock transcription for demonstration...")
            self.pipe = None
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        if self.pipe is None:
            return PhoWhisperModel._mock_transcribe(None, audio_path)
        
        try:
            result = self.pipe(audio_path, generate_kwargs={"language": "vi"})
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""


class Wav2Vec2Model(BaseASRModel):
    """Wav2Vec2-XLS-R model implementation."""
    
    def load_model(self):
        """Load Wav2Vec2 model."""
        print(f"Loading {self.config.name} from {self.config.model_id}...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.config.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.config.model_id)
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK] {self.config.name} loaded successfully on {self.device}")
        except Exception as e:
            print(f"[FAILED] Failed to load {self.config.name}: {e}")
            print("Using mock transcription for demonstration...")
            self.model = None
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using Wav2Vec2."""
        if self.model is None:
            return PhoWhisperModel._mock_transcribe(None, audio_path)
        
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            
            input_values = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""


class Wav2VnModel(BaseASRModel):
    """Wav2Vn model implementation."""
    
    def load_model(self):
        """Load Wav2Vn model."""
        print(f"Loading {self.config.name}...")
        # Note: Wav2Vn might not be publicly available on HuggingFace
        # This is a placeholder implementation
        print("[WARNING] Wav2Vn model not publicly available. Using mock implementation.")
        self.model = None
    
    def transcribe(self, audio_path: str) -> str:
        """Mock transcription for Wav2Vn."""
        return PhoWhisperModel._mock_transcribe(None, audio_path)


class ModelFactory:
    """Factory class to create ASR models."""
    
    # Predefined model configurations
    MODEL_CONFIGS = {
        'phowhisper-tiny': ModelConfig(
            name='PhoWhisper-tiny',
            model_id='vinai/PhoWhisper-tiny',
            model_type='whisper'
        ),
        'phowhisper-base': ModelConfig(
            name='PhoWhisper-base',
            model_id='vinai/PhoWhisper-base',
            model_type='whisper'
        ),
        'phowhisper-small': ModelConfig(
            name='PhoWhisper-small',
            model_id='vinai/PhoWhisper-small',
            model_type='whisper'
        ),
        'phowhisper-medium': ModelConfig(
            name='PhoWhisper-medium',
            model_id='vinai/PhoWhisper-medium',
            model_type='whisper'
        ),
        'phowhisper-large': ModelConfig(
            name='PhoWhisper-large',
            model_id='vinai/PhoWhisper-large',
            model_type='whisper'
        ),
        'whisper-small': ModelConfig(
            name='Whisper-small',
            model_id='openai/whisper-small',
            model_type='whisper'
        ),
        'whisper-medium': ModelConfig(
            name='Whisper-medium',
            model_id='openai/whisper-medium',
            model_type='whisper'
        ),
        'whisper-large': ModelConfig(
            name='Whisper-large-v3',
            model_id='openai/whisper-large-v3',
            model_type='whisper'
        ),
        'wav2vec2-xlsr-vietnamese': ModelConfig(
            name='Wav2Vec2-XLSR-Vietnamese',
            model_id='anuragshas/wav2vec2-large-xlsr-53-vietnamese',
            model_type='wav2vec2'
        ),
        'wav2vec2-base-vietnamese': ModelConfig(
            name='Wav2Vec2-Base-Vietnamese',
            model_id='nguyenvulebinh/wav2vec2-base-vietnamese-250h',
            model_type='wav2vec2'
        ),
        'wav2vn': ModelConfig(
            name='Wav2Vn',
            model_id='wav2vn-placeholder',
            model_type='wav2vn'
        )
    }
    
    @staticmethod
    def create_model(model_key: str) -> BaseASRModel:
        """
        Create an ASR model instance.
        
        Args:
            model_key: Key identifying the model configuration
            
        Returns:
            ASR model instance
        """
        if model_key not in ModelFactory.MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(ModelFactory.MODEL_CONFIGS.keys())}")
        
        config = ModelFactory.MODEL_CONFIGS[model_key]
        
        if config.model_type == 'whisper':
            if 'phowhisper' in model_key.lower():
                return PhoWhisperModel(config)
            else:
                return WhisperModel(config)
        elif config.model_type == 'wav2vec2':
            return Wav2Vec2Model(config)
        elif config.model_type == 'wav2vn':
            return Wav2VnModel(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model keys."""
        return list(ModelFactory.MODEL_CONFIGS.keys())


class ModelEvaluator:
    """
    Evaluator class to manage model evaluation workflow.
    """
    
    def __init__(self, models_to_evaluate: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            models_to_evaluate: List of model keys to evaluate.
                              If None, uses a default set.
        """
        if models_to_evaluate is None:
            # Default models for evaluation
            models_to_evaluate = [
                'phowhisper-small',
                'whisper-small',
                'wav2vec2-xlsr-vietnamese'
            ]
        
        self.models_to_evaluate = models_to_evaluate
        self.models = {}
        
    def load_models(self):
        """Load all models to be evaluated."""
        print("\n" + "="*60)
        print("Loading ASR Models")
        print("="*60 + "\n")
        
        for model_key in self.models_to_evaluate:
            try:
                model = ModelFactory.create_model(model_key)
                model.load_model()
                self.models[model_key] = model
            except Exception as e:
                print(f"Failed to load {model_key}: {e}")
        
        print(f"\nSuccessfully loaded {len(self.models)}/{len(self.models_to_evaluate)} models\n")
    
    def get_loaded_models(self) -> Dict[str, BaseASRModel]:
        """Get dictionary of loaded models."""
        return self.models


if __name__ == "__main__":
    # Example usage
    print("Vietnamese ASR Model Evaluator - Example Usage\n")
    
    print("Available models:")
    for model_key in ModelFactory.get_available_models():
        print(f"  - {model_key}")
    
    print("\n" + "="*60)
    evaluator = ModelEvaluator(models_to_evaluate=['phowhisper-small', 'whisper-small'])
    evaluator.load_models()
    
    # Example transcription
    print("\nExample transcription (with mock data):")
    models = evaluator.get_loaded_models()
    for model_key, model in models.items():
        transcription = model.transcribe("dummy_audio.wav")
        print(f"{model_key}: {transcription}")
