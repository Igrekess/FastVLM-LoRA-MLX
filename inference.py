#!/usr/bin/env python3
"""
inference.py - Run inference with a fine-tuned FastVLM model

Usage:
    python inference.py --model ./output --image test.jpg --prompt "Describe this image"
"""

import argparse
import os
from pathlib import Path

# Apply mlx_vlm patches before importing mlx_vlm
from train import apply_mlx_vlm_patches
if not os.environ.get('MLX_VLM_PATCHED'):
    apply_mlx_vlm_patches()
    os.environ['MLX_VLM_PATCHED'] = '1'


def main():
    parser = argparse.ArgumentParser(description="FastVLM Inference")
    parser.add_argument("--model", "-m", required=True, help="Path to fine-tuned model")
    parser.add_argument("--image", "-i", required=True, help="Path to image")
    parser.add_argument("--prompt", "-p", default="Describe this image in detail.", help="Prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    # Check paths
    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        print(f"Error: Model path not found: {model_path}")
        return

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"Loading model from: {model_path}")

    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_image
    from mlx_vlm.prompt_utils import apply_chat_template

    # Load model
    model, processor = load(str(model_path), trust_remote_code=True)

    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(str(image_path))

    # Format prompt with chat template
    formatted_prompt = apply_chat_template(
        processor, model.config, args.prompt, num_images=1
    )

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    output = generate(
        model,
        processor,
        formatted_prompt,
        image,
        max_tokens=args.max_tokens,
        temp=args.temperature,
        verbose=False,
    )

    print(output.text if hasattr(output, 'text') else output)


if __name__ == "__main__":
    main()
