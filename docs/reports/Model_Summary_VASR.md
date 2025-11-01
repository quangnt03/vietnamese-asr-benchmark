## 1. **PhoWhisper** (VinAI Research, 2024)

PhoWhisper is a Vietnamese-optimized ASR model fine-tuned from OpenAI's Whisper on an 844-hour dataset encompassing diverse Vietnamese accents [arXiv](https://arxiv.org/abs/2406.02555)[GitHub](https://github.com/VinAIResearch/PhoWhisper). It comes in five versions (tiny, base, small, medium, large) and demonstrates state-of-the-art performance on benchmark Vietnamese ASR datasets [GitHub](https://github.com/VinAIResearch/PhoWhisper).

**Key strengths:**

- Specifically designed for Vietnamese dialect variations
- Open-source and available on Hugging Face
- Multiple size options for different deployment scenarios

## 2. **Wav2Vn** (2024)

Wav2Vn leverages Mixture of Experts (MoE) to effectively handle diverse phonetic variations across different Vietnamese accents, demonstrating significant performance improvements in word error rate and robustness to accent variability [IEEE Xplore](https://ieeexplore.ieee.org/document/10717691/).

**Key strengths:**

- Explicitly designed for multi-accent Vietnamese ASR
- Uses MoE architecture for handling dialect diversity
- Scalable unsupervised pre-training approach

## 3. **Whisper (Multilingual)** (OpenAI)

Whisper is OpenAI's advanced multilingual ASR system using self-supervised learning on 680,000 hours of speech data, employing transformer models with attention and multi-task learning [arXiv](https://ar5iv.labs.arxiv.org/html/2410.03458). While not Vietnamese-specific, it's a strong baseline that is renowned for its SOTA performance in multilingual tasks and exceptional fine-tuning capabilities [arXiv](https://arxiv.org/html/2510.22295).

**Key strengths:**

- Strong zero-shot multilingual performance
- Large pre-training data
- Good foundation for fine-tuning on Vietnamese dialects

## 4. **Wav2Vec2-XLS-R (Vietnamese fine-tuned versions)**

XLSR learns cross-lingual speech representations by pretraining a single model from raw waveforms of speech in multiple languages, with the resulting model fine-tuned on labeled data [Hugging Face](https://huggingface.co/facebook/wav2vec2-large-xlsr-53). Several Vietnamese-specific fine-tuned versions are available on Hugging Face.

**Key strengths:**

- Cross-lingual pre-training benefits Vietnamese
- Multiple Vietnamese fine-tuned checkpoints available
- Efficient architecture suitable for deployment