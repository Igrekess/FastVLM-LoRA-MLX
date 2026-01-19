#!/usr/bin/env python3
"""
inference.py - Run inference with a fine-tuned FastVLM model

Usage:
    python inference.py --model ./output --image test.jpg --prompt "Describe this image"
"""

import argparse
from pathlib import Path


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

    # Load model
    model, processor = load(str(model_path))

    # Load image
    print(f"Loading image: {image_path}")
    image = load_image(str(image_path))

    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    output = generate(
        model,
        processor,
        image,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temperature,
        verbose=False,
    )

    print(output)


if __name__ == "__main__":
    main()
