# Fine-tuning ASR models for Singlish-to-English

This repository contains code and instructions for fine-tuning ASR models for **Singlish-to-English transcription and translation**, using Hugging Face Transformers and PyTorch.

We provide fine-tuned models:
- [`ivabojic/whisper-small-sing2eng-transcribe`](https://huggingface.co/ivabojic/whisper-small-sing2eng-transcribe)
- [`ivabojic/whisper-medium-sing2eng-transcribe`](https://huggingface.co/ivabojic/whisper-medium-sing2eng-transcribe)
- [`ivabojic/whisper-small-sing2eng-translate`](https://huggingface.co/ivabojic/whisper-small-sing2eng-translate)
- [`ivabojic/whisper-medium-sing2eng-translate`](https://huggingface.co/ivabojic/whisper-medium-sing2eng-translate)


## Overview

Fine-tuning is done using PyTorch's DistributedDataParallel (DDP) for multi-GPU support. The training pipeline supports:

- Mixed-precision training (via `autocast`)
- Gradient accumulation
- Periodic evaluation with WER/BLEU metrics
- Learning rate scheduling with `ReduceLROnPlateau`
- Automatic checkpointing of the best model


## Evaluation metrics

Depending on the task, the following metrics are used during validation:

- **Transcription**:  
  - `WER` – Word Error Rate  
  - `WER_ortho` – Orthographic normalization  
  - `WER_remove` – Disfluency removal

- **Translation**:  
  - `BLEU` – Bilingual Evaluation Understudy  
  - `ROUGE` – Recall-Oriented Understudy for Gisting Evaluation  
  - `BERTScore` – Semantic similarity using BERT embeddings

---


## Fine-tuning examples

### Fine-tuning `openai/whisper-medium` for transcription (1 GPU)

To fine-tune `openai/whisper-medium` on a single GPU for 3 epochs with batch size 8:

```bash
CUDA_VISIBLE_DEVICES=0 python fine_tuning.py \
  --model 1 \              # Model index: 1 = openai/whisper-medium  
  --world_size 1 \         # Single GPU  
  --batch 8 \              # Batch size  
  --epoch 3                # Number of training epochs
```

### Fine-tuning `openai/whisper-small` for translation (7 GPUs)

To fine-tune `openai/whisper-small for speech-to-text translation using 7 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python fine_tuning.py \
  --model 0 \              # Model index: 0 = openai/whisper-small  
  --world_size 7 \         # Number of GPUs  
  --batch 16 \             # Batch size per GPU  
  --task translate         # Translation task
```
