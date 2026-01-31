# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastVLM LoRA Fine-tuning with MLX - A Python tool for fine-tuning Apple's FastVLM (0.5B vision-language model) using LoRA adapters on Apple Silicon Macs. Designed for memory-efficient training on consumer hardware (8GB+ RAM).

## Commands

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# Basic training with CSV dataset (outputs mlx-vlm format for Python inference)
python train.py --dataset ./examples/sample_dataset --output ./output --epochs 3

# Training with JSONL dataset
python train.py --dataset ./examples/sample_dataset.jsonl --output ./output --epochs 3

# Advanced training with quantization
python train.py --dataset /path/to/data --output ./output --lora-rank 64 --epochs 5 --learning-rate 2e-4 --quantize

# Keep adapter files for potential re-merging
python train.py --dataset /path/to/data --output ./output --keep-adapter

# Training for MetaDone/mlx-swift apps (iOS/macOS)
python train.py --dataset /path/to/data --output ./output --target mlx-swift --quantize
```

### Output Targets
- `--target mlx-vlm` (default): Python inference with mlx_vlm library
- `--target mlx-swift`: MetaDone and other mlx-swift based iOS/macOS apps

### Inference
```bash
python inference.py --model ./output --image test.jpg --prompt "Describe this image"
```

## Architecture

### Core Files

- **train.py** (~1300 lines) - Main training script containing:
  - Runtime patches for mlx_vlm library bugs (lines 25-249)
  - LoRALinear class implementing Low-Rank Adaptation (lines 276-398)
  - Dataset loading for CSV and JSONL formats (lines 405-467)
  - Training loop with gradient filtering (lines 886-1202)
  - Model finalization with weight merging and quantization (lines 494-883)

- **inference.py** - Simple wrapper around mlx_vlm's load/generate API

### Data Flow

1. Load FastVLM base model from HuggingFace
2. Apply LoRA adapters to 48 linear layers (7 per transformer layer × 24 layers)
3. Train only LoRA parameters while freezing base model
4. Merge LoRA weights into base model
5. Optionally quantize to 4-bit
6. Output standalone model directory with all assets

### Dataset Formats

**CSV Format** (directory with images/ folder and captions.csv):
```
dataset/
├── images/
│   └── *.jpg
└── captions.csv  # columns: filename, caption
```

**JSONL Format** (single file):
```json
{"image": "path/to/image.jpg", "caption": "description text"}
{"image": "path/to/image.jpg", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

### Key Technical Details

- **LoRA Target Layers**: Attention projections (q, k, v, o) and MLP projections (gate, up, down)
- **Batch Size**: Default 1 to manage Apple Silicon memory
- **Runtime Patches**: train.py patches mlx_vlm bugs at import time (axis parameter fix, Qwen2Model signature fix)
- **Weight Merging**: W' = W + (lora_b.T @ lora_a.T) * scale, where scale = lora_alpha / lora_rank

### Output Formats

**mlx-vlm (default)**: For Python inference with mlx_vlm library
- Weight keys: `mm_projector.X`, `vision_tower.vision_model.X`, `language_model.model.X`
- lm_head weights: tied with embed_tokens (not saved separately)
- Config: `tie_word_embeddings: true`

**mlx-swift**: For MetaDone and iOS/macOS apps using mlx-swift
- Weight keys: `multi_modal_projector.linear_X`, `language_model.model.X`, `language_model.lm_head.X`
- vision_tower: skipped (uses CoreML fastvithd.mlpackage)
- Config: `tie_word_embeddings: false`, `processor_class: "LlavaProcessor"`
- Requires fastvithd.mlpackage copied from MetaDone bundle

### Output Directory Structure
```
output/
├── model.safetensors          # Merged model weights
├── model.safetensors.index.json  # Weight index (mlx-swift only)
├── config.json                # FastVLM configuration
├── tokenizer.json             # Tokenizer data
├── tokenizer_config.json
├── special_tokens_map.json
├── generation_config.json
├── preprocessor_config.json
├── processor_config.json
├── fastvithd.mlpackage/       # CoreML vision encoder (mlx-swift only)
├── adapter.safetensors        # (with --keep-adapter) LoRA weights before merge
└── adapter_config.json        # (with --keep-adapter) LoRA config
```

## Dependencies

Target: macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+, 8GB+ RAM

Key packages:
- mlx>=0.21.0 (requires this version for tree_map)
- mlx-vlm>=0.3.9 (has known bugs patched by train.py)
- transformers>=4.40.0
- huggingface-hub>=0.23.0
