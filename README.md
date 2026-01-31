# FastVLM LoRA Fine-tuning with MLX

Fine-tune Apple's [FastVLM](https://huggingface.co/apple/FastVLM-0.5B) vision-language model using LoRA adapters on Apple Silicon.

## Features

- Native MLX implementation for Apple Silicon (M1/M2/M3/M4)
- LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
- Automatic mlx_vlm compatibility patches
- Support for CSV and JSONL dataset formats
- Optional 4-bit quantization for smaller model size
- **Dual output formats**: Python (mlx-vlm) or iOS/macOS apps (mlx-swift)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~8GB RAM minimum (16GB+ recommended for larger datasets)

## Installation

```bash
# Clone the repository
git clone https://github.com/AdrienMusic/FastVLM-LoRA-MLX.git
cd FastVLM-LoRA-MLX

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Format

### Option 1: CSV Format

Create a folder with the following structure:
```
my_dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── captions.csv
```

The `captions.csv` file should have the following columns:
```csv
filename,caption
image001.jpg,"A detailed description of the first image"
image002.jpg,"A detailed description of the second image"
```

### Option 2: JSONL Format

Create a `.jsonl` file with one JSON object per line:
```jsonl
{"image": "/path/to/image1.jpg", "caption": "Description of image 1"}
{"image": "/path/to/image2.jpg", "caption": "Description of image 2"}
```

Or with conversations format:
```jsonl
{"image": "/path/to/image1.jpg", "conversations": [{"from": "human", "value": "Describe this image"}, {"from": "gpt", "value": "Description of image 1"}]}
```

## Usage

### Basic Training (Python/mlx-vlm)

```bash
python train.py \
    --dataset /path/to/your/dataset \
    --output ./output \
    --epochs 3
```

### Training for iOS/macOS Apps (mlx-swift)

```bash
python train.py \
    --dataset /path/to/your/dataset \
    --output ./output \
    --target mlx-swift \
    --quantize \
    --epochs 3
```

### Advanced Options

```bash
python train.py \
    --dataset /path/to/your/dataset \
    --output ./output \
    --model apple/FastVLM-0.5B \
    --epochs 5 \
    --batch-size 1 \
    --learning-rate 2e-4 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --max-samples 1000 \
    --target mlx-vlm \
    --quantize
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset`, `-d` | Required | Path to dataset folder or JSONL file |
| `--output`, `-o` | Required | Output directory for the trained model |
| `--model`, `-m` | `apple/FastVLM-0.5B` | Base FastVLM model |
| `--epochs`, `-e` | `1` | Number of training epochs |
| `--batch-size`, `-b` | `1` | Batch size (keep at 1 for memory efficiency) |
| `--learning-rate`, `--lr` | `1e-4` | Learning rate |
| `--lora-rank`, `-r` | `8` | LoRA rank (higher = more parameters) |
| `--lora-alpha`, `-a` | `16` | LoRA alpha scaling factor |
| `--max-samples` | None | Limit number of training samples |
| `--target` | `mlx-vlm` | Output format: `mlx-vlm` (Python) or `mlx-swift` (iOS/macOS) |
| `--quantize` | False | Quantize output to 4-bit |
| `--keep-adapter` | False | Keep adapter files after merging |

### Output Targets

| Target | Use Case | Model Size |
|--------|----------|------------|
| `mlx-vlm` | Python inference with mlx_vlm library | ~1.2 GB (16-bit) / ~340 MB (4-bit) |
| `mlx-swift` | iOS/macOS apps using mlx-swift | ~1.2 GB (16-bit) / ~340 MB (4-bit) |

### LoRA Rank Guidelines

| Rank | Parameters | Use Case |
|------|------------|----------|
| 8 | ~4M | Quick experiments, small datasets |
| 32 | ~17M | General fine-tuning |
| 64 | ~35M | Better quality, larger datasets |
| 128 | ~70M | Maximum quality |

## Inference

### With inference.py

```bash
python inference.py \
    --model ./output \
    --image test.jpg \
    --prompt "Describe this image in detail."
```

### With mlx-vlm directly

```python
from mlx_vlm import load, generate
from mlx_vlm.utils import load_image

model, processor = load("./output", trust_remote_code=True)
image = load_image("test_image.jpg")

output = generate(
    model,
    processor,
    "<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n",
    image,
    max_tokens=256
)
print(output)
```

## Output Structure

```
output/
├── model.safetensors          # Merged model weights
├── model.safetensors.index.json
├── config.json                # Model configuration
├── tokenizer.json             # Tokenizer
├── tokenizer_config.json
├── preprocessor_config.json
├── processor_config.json
├── special_tokens_map.json
├── fastvithd.mlpackage/       # Vision encoder (mlx-swift only)
└── ...
```

## Training Tips

1. **Start small**: Test with `--max-samples 50` first to verify everything works
2. **Monitor loss**: Loss should decrease between epochs. If not, try a lower learning rate
3. **Memory**: If you run out of memory, reduce `--lora-rank` or ensure batch-size is 1
4. **Quality captions**: The quality of your captions directly impacts the fine-tuned model
5. **Quantization**: Use `--quantize` for deployment to reduce model size by ~70%

## Troubleshooting

### "module 'mlx.core' has no attribute 'tree_map'"
Make sure you have mlx >= 0.21.0: `pip install --upgrade mlx`

### Memory errors
- Reduce LoRA rank: `--lora-rank 8`
- Use batch size 1: `--batch-size 1`
- Close other applications

### Loss not decreasing
- Try a lower learning rate: `--learning-rate 5e-5`
- Check your dataset for quality issues
- Ensure captions are descriptive and accurate

## License

MIT License

## Acknowledgments

- [Apple MLX](https://github.com/ml-explore/mlx) - Machine learning framework for Apple Silicon
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models for MLX
- [FastVLM](https://huggingface.co/apple/FastVLM-0.5B) - Apple's efficient vision-language model
