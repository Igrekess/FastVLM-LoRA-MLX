#!/usr/bin/env python3
"""
train_fastvlm_mlx.py - Native MLX Training Script for FastVLM

Fine-tunes FastVLM using pure MLX with LoRA adapters.
Designed for Apple Silicon (M1/M2/M3).

This script includes automatic runtime patching for mlx_vlm compatibility,
fixing known bugs without modifying the user's mlx_vlm installation.

Usage:
    python train_fastvlm_mlx.py --dataset /path/to/dataset --output ./output
"""

import argparse
import json
import math
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# MLX-VLM Runtime Patches
# ============================================================
# These patches fix known bugs in mlx_vlm without modifying
# the user's installation. They are applied at runtime via
# monkey-patching and survive mlx_vlm updates.
#
# Known bugs patched:
# 1. fastvlm.py: `dim=0` should be `axis=0` (MLX API)
# 2. language.py: Qwen2Model signature changed in mlx-lm
# ============================================================

def apply_mlx_vlm_patches():
    """
    Apply runtime patches to mlx_vlm for FastVLM compatibility.

    This function patches known bugs in mlx_vlm without modifying
    the installed package. Patches are applied via monkey-patching
    and are safe to apply multiple times (idempotent).

    Returns:
        dict: Information about applied patches
    """
    import importlib
    import mlx.core as mx

    patches_applied = {
        "fastvlm_axis_fix": False,
        "language_qwen2_fix": False,
        "version": None,
    }

    try:
        import mlx_vlm
        patches_applied["version"] = getattr(mlx_vlm, "__version__", "unknown")
        vlm_path = Path(mlx_vlm.__file__).parent
        print(f"[Patch] mlx_vlm version: {patches_applied['version']}")
        print(f"[Patch] mlx_vlm path: {vlm_path}")
    except ImportError:
        print("[Patch] ERROR: mlx_vlm not installed")
        return patches_applied

    # ----------------------------------------------------------------
    # Patch 1: Fix dim=0 -> axis=0 in fastvlm.py
    # ----------------------------------------------------------------
    # MLX uses `axis` parameter for concatenation, not `dim`
    # This bug causes: TypeError: concatenate() got an unexpected keyword argument 'dim'
    # ----------------------------------------------------------------
    try:
        fastvlm_path = vlm_path / "models" / "fastvlm" / "fastvlm.py"
        if fastvlm_path.exists():
            content = fastvlm_path.read_text()

            # Check if bug exists (dim=0 instead of axis=0)
            if "dim=0" in content:
                print("[Patch] Found 'dim=0' bug in fastvlm.py - applying axis fix...")

                # Monkey-patch the Model class's prepare_inputs_for_multimodal
                from mlx_vlm.models.fastvlm import fastvlm as fastvlm_module

                original_prepare = fastvlm_module.Model.prepare_inputs_for_multimodal

                def patched_prepare_inputs_for_multimodal(self, image_features, input_ids, mask):
                    """Patched version with axis=0 fix."""
                    if mask is not None:
                        input_ids = [
                            cur_input_ids[
                                (start := mx.argmax(cur_mask).item()) : start
                                + cur_mask.sum().item()
                            ]
                            for cur_input_ids, cur_mask in zip(input_ids, mask)
                        ]

                    import numpy as np
                    new_input_embeds = []
                    cur_image_idx = 0
                    for batch_idx, cur_input_ids in enumerate(input_ids):
                        num_images = (cur_input_ids == self.config.image_token_index).sum()
                        if num_images == 0:
                            cur_image_features = image_features[cur_image_idx]
                            cur_input_embeds_1 = self.language_model.model.embed_tokens(
                                cur_input_ids
                            )
                            cur_input_embeds = mx.concatenate(
                                [cur_input_embeds_1, cur_image_features[0:0]], axis=0  # FIXED
                            )
                            new_input_embeds.append(cur_input_embeds)
                            cur_image_idx += 1
                            continue

                        image_token_indices = (
                            [-1]
                            + np.where(np.array(cur_input_ids == self.config.image_token_index))[
                                0
                            ].tolist()
                            + [cur_input_ids.shape[0]]
                        )
                        cur_input_ids_noim = []
                        for i in range(len(image_token_indices) - 1):
                            cur_input_ids_noim.append(
                                cur_input_ids[
                                    image_token_indices[i] + 1 : image_token_indices[i + 1]
                                ]
                            )
                        split_sizes = image_token_indices[1:]
                        cur_input_embeds = self.language_model.model.embed_tokens(
                            mx.concatenate(cur_input_ids_noim)
                        )
                        cur_input_embeds_no_im = mx.split(cur_input_embeds, split_sizes)

                        cur_new_input_embeds = []
                        for i in range(num_images.item() + 1):
                            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                            if i < num_images:
                                cur_image_features = image_features[cur_image_idx]
                                cur_image_idx += 1
                                cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds = mx.concatenate(cur_new_input_embeds)

                        new_input_embeds.append(cur_new_input_embeds)

                    if self.config.tokenizer_model_max_length is not None:
                        new_input_embeds = [
                            x[: self.config.tokenizer_model_max_length] for x in new_input_embeds
                        ]

                    max_len = max(x.shape[0] for x in new_input_embeds)
                    new_input_embeds_padded = []
                    for i, cur_new_embed in enumerate(new_input_embeds):
                        cur_len = cur_new_embed.shape[0]
                        padded = cur_new_embed
                        if max_len > cur_len:
                            if self.config.tokenizer_padding_side == "left":
                                padded = mx.concatenate(
                                    (
                                        mx.zeros(
                                            (max_len - cur_len, cur_new_embed.shape[1]),
                                            dtype=cur_new_embed.dtype,
                                        ),
                                        cur_new_embed,
                                    ),
                                    axis=0,  # FIXED
                                )
                            else:
                                padded = mx.concatenate(
                                    (
                                        cur_new_embed,
                                        mx.zeros(
                                            (max_len - cur_len, cur_new_embed.shape[1]),
                                            dtype=cur_new_embed.dtype,
                                        ),
                                    ),
                                    axis=0,  # FIXED
                                )
                        new_input_embeds_padded.append(padded)
                    new_input_embeds = mx.stack(new_input_embeds_padded)
                    return new_input_embeds

                fastvlm_module.Model.prepare_inputs_for_multimodal = patched_prepare_inputs_for_multimodal
                patches_applied["fastvlm_axis_fix"] = True
                print("[Patch] Applied fastvlm axis=0 fix")
            else:
                print("[Patch] fastvlm.py already uses axis=0 - no patch needed")
                patches_applied["fastvlm_axis_fix"] = "not_needed"

    except Exception as e:
        print(f"[Patch] Warning: Could not apply fastvlm axis fix: {e}")

    # ----------------------------------------------------------------
    # Patch 2: Fix Qwen2Model signature in language.py
    # ----------------------------------------------------------------
    # mlx-lm changed Qwen2Model.__call__ signature (removed mask param)
    # Different mlx_vlm versions have different bugs:
    # - Old: self.model(inputs, mask=mask, cache=cache, inputs_embeds=inputs_embeds)
    # - v0.3.9: self.model(inputs, None, cache, inputs_embeds) - passes 4 args but Qwen2 expects 3
    # Fixed: self.model(inputs, cache, inputs_embeds)
    # ----------------------------------------------------------------
    try:
        language_path = vlm_path / "models" / "fastvlm" / "language.py"
        if language_path.exists():
            content = language_path.read_text()

            # Check for various bug patterns
            needs_patch = False
            if "self.model(inputs, mask" in content or "mask=mask" in content:
                print("[Patch] Found Qwen2Model mask keyword bug in language.py")
                needs_patch = True
            elif "self.model(inputs, None, cache" in content:
                # mlx_vlm 0.3.9 bug: passes None as second arg (was mask), but Qwen2 doesn't expect it
                print("[Patch] Found Qwen2Model extra None argument bug in language.py (v0.3.9)")
                needs_patch = True

            if needs_patch:
                print("[Patch] Applying Qwen2Model fix...")

                from mlx_vlm.models.fastvlm import language as language_module
                from mlx_vlm.models.base import LanguageModelOutput

                def patched_language_model_call(
                    self,
                    inputs,
                    mask=None,
                    cache=None,
                    inputs_embeds=None,
                ):
                    """Patched version compatible with newer mlx-lm Qwen2Model."""
                    # New mlx-lm API: Qwen2Model(inputs, cache, input_embeddings)
                    # Note: mask is ignored as Qwen2Model handles it internally
                    out = self.model(inputs, cache, inputs_embeds)
                    out = self.model.embed_tokens.as_linear(out)
                    return LanguageModelOutput(out)

                language_module.LanguageModel.__call__ = patched_language_model_call
                patches_applied["language_qwen2_fix"] = True
                print("[Patch] Applied language.py Qwen2Model fix")
            else:
                print("[Patch] language.py already compatible - no patch needed")
                patches_applied["language_qwen2_fix"] = "not_needed"

    except Exception as e:
        print(f"[Patch] Warning: Could not apply language Qwen2Model fix: {e}")

    # Summary
    print(f"[Patch] Summary: {patches_applied}")
    return patches_applied


# ============================================================
# Apply patches before importing mlx_vlm components
# ============================================================
if not os.environ.get('MLX_VLM_PATCHED'):
    print("\n" + "="*60)
    print("Applying mlx_vlm compatibility patches...")
    print("="*60)
    _patch_result = apply_mlx_vlm_patches()
    os.environ['MLX_VLM_PATCHED'] = '1'
    print("="*60 + "\n")


import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map
import numpy as np
from PIL import Image


# ============================================================
# LoRA Implementation for MLX
# ============================================================

class LoRALinear(nn.Module):
    """Linear layer with LoRA (Low-Rank Adaptation)."""

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        """Create a LoRALinear from an existing Linear layer."""
        output_dims, input_dims = linear.weight.shape
        lora = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        # Copy original weights (frozen)
        lora.linear = linear
        return lora

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = dropout

        # Original linear layer (will be set from_linear or created)
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        # LoRA matrices A and B
        # A: input_dims -> rank (initialized with small random values)
        # B: rank -> output_dims (initialized to zero)
        self.lora_a = mx.random.normal((input_dims, rank)) * 0.01
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array, training: bool = True) -> mx.array:
        # Original output (frozen weights)
        y = self.linear(x)

        # LoRA adaptation with optional dropout
        # x @ A @ B * scale
        lora_input = x
        if self.dropout > 0 and training:
            # Apply dropout mask
            mask = mx.random.bernoulli(1 - self.dropout, x.shape)
            lora_input = (x * mask) / (1 - self.dropout)

        lora_out = (lora_input @ self.lora_a) @ self.lora_b * self.scale

        return y + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = None,
) -> Tuple[nn.Module, List[str]]:
    """
    Apply LoRA to specified linear layers in the model.

    Args:
        model: The model to modify
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        target_modules: List of module name patterns to target

    Returns:
        Modified model and list of modified layer names
    """
    if target_modules is None:
        # Default: target attention and MLP projections in language model
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    modified_layers = []

    def replace_linear_with_lora(parent: nn.Module, name: str, module: nn.Module, path: str):
        """Recursively replace Linear layers with LoRALinear."""
        if isinstance(module, nn.Linear):
            # Check if this layer should have LoRA
            should_apply = any(target in path for target in target_modules)
            if should_apply:
                lora_layer = LoRALinear.from_linear(module, rank=rank, alpha=alpha)
                setattr(parent, name, lora_layer)
                modified_layers.append(path)
                return

        # Recurse into children
        for child_name, child in module.children().items():
            child_path = f"{path}.{child_name}" if path else child_name
            replace_linear_with_lora(module, child_name, child, child_path)

    # Start recursion from model root
    for name, child in model.children().items():
        replace_linear_with_lora(model, name, child, name)

    return model, modified_layers


def get_lora_parameters(model: nn.Module) -> Dict[str, mx.array]:
    """Get only the LoRA parameters (trainable)."""
    lora_params = {}

    def collect_lora_params(module: nn.Module, prefix: str = ""):
        if isinstance(module, LoRALinear):
            lora_params[f"{prefix}.lora_a"] = module.lora_a
            lora_params[f"{prefix}.lora_b"] = module.lora_b
        else:
            for name, child in module.children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                collect_lora_params(child, child_prefix)

    collect_lora_params(model)
    return lora_params


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(dataset_path: str, max_samples: int = None) -> List[Dict]:
    """Load dataset from various formats."""
    path = Path(dataset_path)
    data = []

    # CSV folder format (MetaDone format: images/ + captions.csv)
    if path.is_dir():
        csv_path = path / "captions.csv"
        images_dir = path / "images"

        if csv_path.exists():
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get('filename', row.get('image_path', ''))
                    caption = row.get('caption', '')

                    if images_dir.exists():
                        image_path = images_dir / filename
                    else:
                        image_path = path / filename

                    if image_path.exists() and caption:
                        data.append({
                            "image": str(image_path),
                            "caption": caption
                        })
        else:
            # Look for JSONL file in directory (MetaDone exported format)
            jsonl_files = list(path.glob("*.jsonl"))
            if jsonl_files:
                jsonl_path = jsonl_files[0]  # Use first JSONL found
                print(f"Found JSONL file: {jsonl_path}")
                return load_dataset(str(jsonl_path), max_samples)

    # JSONL format (direct file path)
    elif path.suffix == '.jsonl':
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "image" in item:
                        if "conversations" in item:
                            # Extract caption from conversations
                            for conv in item["conversations"]:
                                if conv.get("from") in ["gpt", "assistant"]:
                                    caption = conv.get("value", "")
                                    break
                            else:
                                caption = ""
                        else:
                            caption = item.get("caption", "")

                        data.append({
                            "image": item["image"],
                            "caption": caption
                        })

    if max_samples:
        data = data[:max_samples]

    return data


# ============================================================
# Training Functions
# ============================================================

def load_fastvlm_model(model_path: str):
    """Load FastVLM model using mlx_vlm.load for proper multimodal support."""
    from mlx_vlm import load

    # The Apple model doesn't have preprocessor_config.json, so we use mlx-community version
    # which has all required processor files for proper multimodal handling
    mlx_community_model = "mlx-community/FastVLM-0.5B-bf16"

    # Use mlx_vlm.load which returns the full processor with proper image handling
    # We load from mlx-community to get the proper processor with image handling
    model, processor = load(mlx_community_model, trust_remote_code=True)

    # Extract tokenizer for compatibility with existing code
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # Return processor as image_processor for backward compatibility
    # The processor contains both tokenizer and image_processor
    return model, tokenizer, processor


def finalize_model(output_dir: str, base_model_name: str, adapter_weights: dict, lora_rank: int = 8, lora_alpha: int = 16, quantize: bool = False, keep_adapter: bool = False, hf_token: str = None, target: str = "mlx-vlm"):
    """
    Finalize the trained model for inference.

    This function:
    1. Downloads/loads the base model
    2. Converts weights to target format (mlx-vlm or mlx-swift)
    3. Merges LoRA adapter weights into the base model
    4. Optionally quantizes to 4-bit (if quantize=True)
    5. Saves the model weights
    6. Copies required config files and assets

    Args:
        target: "mlx-vlm" for Python mlx_vlm inference (default, standalone)
                "mlx-swift" for MetaDone/mlx-swift inference (iOS/macOS apps)
    """
    import shutil
    import re
    from huggingface_hub import snapshot_download

    print("[PROGRESS] Step 1/6: Loading base model weights...", flush=True)

    # Use token from argument or environment variable
    effective_token = hf_token or os.environ.get('HF_TOKEN')
    if effective_token:
        print("  Using Hugging Face authentication token")

    # Download base model if needed
    try:
        model_path = snapshot_download(base_model_name, token=effective_token)
        print(f"  Base model path: {model_path}")
    except Exception as e:
        print(f"  Warning: Could not download base model: {e}")
        print("  Attempting to use cached model...")
        model_path = None

    # Find the model.safetensors file
    base_model_file = None
    if model_path:
        possible_paths = [
            os.path.join(model_path, "model.safetensors"),
            os.path.join(model_path, "pytorch_model.bin"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                base_model_file = p
                break

    if not base_model_file:
        print("  ERROR: Could not find base model weights")
        print("  Skipping model finalization - manual conversion required")
        return

    print(f"  Loading weights from: {base_model_file}")
    base_weights = mx.load(base_model_file)
    print(f"  Loaded {len(base_weights)} tensors")

    print("[PROGRESS] Step 2/5: Converting to MLX format...", flush=True)

    def convert_key(key):
        """Convert PyTorch key to target format.

        mlx-vlm (Python) expects:
        - mm_projector keys: mm_projector.X (keep as-is, remove model. prefix)
        - vision_tower keys: vision_tower.vision_model.X
        - language model keys: language_model.model.X
        - lm_head keys: skip (tied with embed_tokens)

        mlx-swift (MetaDone) expects:
        - mm_projector keys: multi_modal_projector.linear_X.weight/bias
        - vision_tower keys: skip (using CoreML fastvithd.mlpackage)
        - language model keys: language_model.model.X
        - lm_head keys: language_model.lm_head.X
        """
        if target == "mlx-swift":
            # ============ MLX-SWIFT FORMAT ============
            # mm_projector keys -> multi_modal_projector.linear_X format
            if "mm_projector" in key:
                match = re.match(r"(?:model\.)?mm_projector\.(\d+)\.(.*)", key)
                if match:
                    idx, rest = match.groups()
                    return f"multi_modal_projector.linear_{idx}.{rest}"
                return key

            # vision_tower keys - SKIP (mlx-swift uses CoreML fastvithd.mlpackage)
            if "vision_tower" in key:
                return None

            # lm_head keys - convert to language_model.lm_head.X
            if key.startswith("lm_head."):
                return "language_model." + key

            # Language model keys
            if key.startswith("model."):
                return "language_model." + key

            return key
        else:
            # ============ MLX-VLM FORMAT (default) ============
            # mm_projector keys - remove model. prefix
            if "mm_projector" in key:
                if key.startswith("model."):
                    return key[6:]  # Remove "model." prefix -> mm_projector.X
                return key

            # vision_tower keys - convert to mlx_vlm format
            if "vision_tower" in key:
                if key.startswith("model.vision_tower.vision_tower.model."):
                    new_key = "vision_tower.vision_model." + key[38:]
                    new_key = new_key.replace("patch_embed.", "patch_embed.blocks.")
                    return new_key
                elif key.startswith("model.vision_tower."):
                    return key[6:]  # Remove "model." prefix
                return key

            # lm_head keys - skip (mlx_vlm ties lm_head with embed_tokens)
            if key.startswith("lm_head."):
                return None

            # Language model keys
            if key.startswith("model."):
                return "language_model." + key

            return key

    converted_weights = {}
    skipped = 0
    for key, tensor in base_weights.items():
        new_key = convert_key(key)
        if new_key is not None:
            converted_weights[new_key] = tensor
        else:
            skipped += 1

    print(f"  Converted {len(converted_weights)} tensors")

    print("[PROGRESS] Step 3/5: Merging LoRA adapter...", flush=True)

    def get_base_key_from_lora(lora_key):
        """Convert LoRA key to base model key.

        LoRA keys are like: layers[0].self_attn.q_proj.lora_a
        Base keys should be: language_model.model.layers.0.self_attn.q_proj.weight
        (matching the converted mlx_vlm format)
        """
        match = re.match(r"(.+)\.(lora_[ab])", lora_key)
        if match:
            path, lora_type = match.groups()
            # layers[0].mlp.down_proj -> layers.0.mlp.down_proj
            path = re.sub(r"layers\[(\d+)\]", r"layers.\1", path)
            base_key = f"language_model.model.{path}.weight"
            return base_key, lora_type
        return None, None

    # Group LoRA pairs
    lora_pairs = {}
    for key, tensor in adapter_weights.items():
        base_key, lora_type = get_base_key_from_lora(key)
        if base_key:
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            # Convert MLX array to numpy if needed, then back to MLX
            if hasattr(tensor, 'tolist'):
                lora_pairs[base_key][lora_type] = tensor
            else:
                lora_pairs[base_key][lora_type] = mx.array(tensor)

    # Merge LoRA: W' = W + (lora_b.T @ lora_a.T) * scale
    merged_count = 0
    scale = lora_alpha / lora_rank
    print(f"  Using LoRA scale: {scale} (alpha={lora_alpha}, rank={lora_rank})")
    for base_key, lora in lora_pairs.items():
        if 'lora_a' in lora and 'lora_b' in lora and base_key in converted_weights:
            lora_a = lora['lora_a']
            lora_b = lora['lora_b']
            original = converted_weights[base_key]

            delta = mx.matmul(lora_b.T, lora_a.T) * scale

            if original.shape == delta.shape:
                converted_weights[base_key] = original + delta
                merged_count += 1

    print(f"  Merged {merged_count} LoRA layers")

    # Determine total steps based on quantization
    total_steps = 6 if quantize else 5

    if quantize:
        print(f"[PROGRESS] Step 4/{total_steps}: Quantizing model to 4-bit...", flush=True)

        # Use mx.quantize() for proper mlx-swift compatibility
        # This produces uint32 packed format that mlx-swift expects
        group_size = 64
        bits = 4

        def should_quantize(key, tensor):
            """Determine if a tensor should be quantized."""
            if not key.endswith('.weight'):
                return False
            if 'norm' in key.lower() or 'layernorm' in key.lower():
                return False
            # Note: embed_tokens IS quantized in the bundled model, so we quantize it too
            if tensor.ndim < 2:
                return False
            if tensor.size < 256:
                return False
            # Check if dimensions are compatible with group_size
            if tensor.shape[-1] % group_size != 0:
                return False
            return True

        final_weights = {}
        quantized_count = 0
        skipped_count = 0

        for key, tensor in converted_weights.items():
            if should_quantize(key, tensor):
                # Use mx.quantize on individual tensor - produces uint32 packed format
                packed, scales, biases = mx.quantize(tensor, group_size=group_size, bits=bits)
                base_key = key[:-7]  # Remove '.weight'
                final_weights[f"{base_key}.weight"] = packed
                # Convert scales and biases to float16 for mlx-swift compatibility
                final_weights[f"{base_key}.scales"] = scales.astype(mx.float16)
                final_weights[f"{base_key}.biases"] = biases.astype(mx.float16)
                quantized_count += 1
            else:
                final_weights[key] = tensor
                skipped_count += 1

        print(f"  Quantized {quantized_count} layers (mx.quantize uint32 format)")
        print(f"  Kept {skipped_count} layers in full precision")
        save_step = 5
    else:
        print(f"[PROGRESS] Step 4/{total_steps}: Keeping 16-bit weights (no quantization)...", flush=True)
        final_weights = converted_weights
        print(f"  Keeping {len(final_weights)} tensors in 16-bit precision")
        save_step = 4

    print(f"[PROGRESS] Step {save_step}/{total_steps}: Saving model...", flush=True)

    # Convert any bfloat16 to float16 for mlx-swift compatibility
    bfloat_count = 0
    for k, v in final_weights.items():
        if v.dtype == mx.bfloat16:
            final_weights[k] = v.astype(mx.float16)
            bfloat_count += 1
    if bfloat_count > 0:
        print(f"  Converted {bfloat_count} tensors from bfloat16 to float16")

    output_model_file = os.path.join(output_dir, "model.safetensors")
    mx.save_safetensors(output_model_file, final_weights)

    size_mb = os.path.getsize(output_model_file) / (1024 * 1024)
    print(f"  Saved model: {size_mb:.1f} MB {'(4-bit)' if quantize else '(16-bit)'}")

    # Create model.safetensors.index.json (required by mlx-swift)
    total_size = sum(w.nbytes for w in final_weights.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: "model.safetensors" for k in final_weights.keys()}
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=4)
    print(f"  Created model index")

    # Remove adapter files (unless --keep-adapter was specified)
    if not keep_adapter:
        adapter_file = os.path.join(output_dir, "adapter.safetensors")
        adapter_config = os.path.join(output_dir, "adapter_config.json")
        if os.path.exists(adapter_file):
            os.remove(adapter_file)
        if os.path.exists(adapter_config):
            os.remove(adapter_config)
    else:
        print(f"  Keeping adapter files (adapter.safetensors, adapter_config.json)")

    config_step = save_step + 1
    print(f"[PROGRESS] Step {config_step}/{total_steps}: Creating config and copying assets...", flush=True)

    import glob as glob_module
    config_file = os.path.join(output_dir, "config.json")

    if target == "mlx-swift":
        # ================================================================
        # MLX-SWIFT TARGET: Use MetaDone bundled model configs
        # ================================================================
        print(f"  Target: mlx-swift (MetaDone)")

        metadone_bundled_paths = [
            # Xcode DerivedData (development)
            os.path.expanduser("~/Library/Developer/Xcode/DerivedData/MetaDone-*/Build/Products/Debug/MetaDone.app/Contents/Resources/Models/fastvlm-0.5b-captions"),
            os.path.expanduser("~/Library/Developer/Xcode/DerivedData/MetaDone-*/Build/Products/Release/MetaDone.app/Contents/Resources/Models/fastvlm-0.5b-captions"),
            # Installed app
            "/Applications/MetaDone.app/Contents/Resources/Models/fastvlm-0.5b-captions",
            os.path.expanduser("~/Applications/MetaDone.app/Contents/Resources/Models/fastvlm-0.5b-captions"),
        ]

        # Also check relative to script location (when running from MetaDone bundle)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        metadone_bundled_paths.extend([
            os.path.join(script_dir, "..", "Models", "fastvlm-0.5b-captions"),
            os.path.join(script_dir, "Models", "fastvlm-0.5b-captions"),
            os.path.join(script_dir, "Resources", "Models", "fastvlm-0.5b-captions"),
        ])

        bundled_model_dir = None
        for pattern in metadone_bundled_paths:
            matches = glob_module.glob(pattern)
            for match in matches:
                if os.path.exists(os.path.join(match, "config.json")):
                    bundled_model_dir = match
                    print(f"  Found MetaDone bundled model: {match}")
                    break
            if bundled_model_dir:
                break

        if bundled_model_dir:
            # Copy all config files from MetaDone bundled model
            config_files_to_copy = [
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "preprocessor_config.json", "processor_config.json",
                "special_tokens_map.json", "generation_config.json"
            ]
            for cf in config_files_to_copy:
                src = os.path.join(bundled_model_dir, cf)
                dst = os.path.join(output_dir, cf)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            print(f"  Copied config files from MetaDone bundled model")

            # Update config.json with quantization info if needed
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            if quantize:
                config_data["quantization"] = {"group_size": 64, "bits": 4}
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=4)
                print(f"  Added quantization info to config.json")

            # Copy fastvithd.mlpackage (vision encoder for mlx-swift)
            mlpackage_src = os.path.join(bundled_model_dir, "fastvithd.mlpackage")
            mlpackage_dst = os.path.join(output_dir, "fastvithd.mlpackage")
            if os.path.exists(mlpackage_src) and not os.path.exists(mlpackage_dst):
                shutil.copytree(mlpackage_src, mlpackage_dst)
                print(f"  Copied fastvithd.mlpackage")
        else:
            # Fallback: Create mlx-swift compatible config files
            print("  MetaDone bundled model not found, creating mlx-swift config files...")
            _create_mlx_swift_configs(output_dir, model_path, quantize)

    else:
        # ================================================================
        # MLX-VLM TARGET (default): Use HuggingFace model configs
        # ================================================================
        print(f"  Target: mlx-vlm (Python)")

        # Try to find mlx-community model in HuggingFace cache
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        mlx_community_patterns = [
            os.path.join(hf_cache, "models--mlx-community--FastVLM-0.5B-bf16", "snapshots", "*"),
        ]

        mlx_community_dir = None
        for pattern in mlx_community_patterns:
            matches = glob_module.glob(pattern)
            for match in matches:
                if os.path.exists(os.path.join(match, "config.json")):
                    mlx_community_dir = match
                    print(f"  Found mlx-community model: {match}")
                    break
            if mlx_community_dir:
                break

        if mlx_community_dir:
            # Copy config files from mlx-community model
            config_files_to_copy = [
                "config.json", "tokenizer.json",
                "preprocessor_config.json", "processor_config.json",
                "special_tokens_map.json", "generation_config.json",
                # Python files needed for AutoProcessor
                "processing_fastvlm.py", "llava_qwen.py"
            ]
            for cf in config_files_to_copy:
                src = os.path.join(mlx_community_dir, cf)
                dst = os.path.join(output_dir, cf)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            print(f"  Copied config files from mlx-community model")

            # Copy tokenizer_config.json from base model (has chat_template)
            if model_path:
                src = os.path.join(model_path, "tokenizer_config.json")
                dst = os.path.join(output_dir, "tokenizer_config.json")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"  Copied tokenizer_config.json from base model (with chat_template)")

            # Update config.json with quantization info if needed
            if quantize:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                config_data["quantization"] = {"group_size": 64, "bits": 4}
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=4)
                print(f"  Added quantization info to config.json")
        else:
            # Fallback: copy from base model and create missing files
            print("  mlx-community model not found, using base model configs...")
            _create_mlx_vlm_configs(output_dir, model_path, quantize)


def _create_mlx_swift_configs(output_dir: str, model_path: str, quantize: bool):
    """Create mlx-swift compatible config files (fallback)."""
    import json
    import shutil

    config_file = os.path.join(output_dir, "config.json")

    # Create config.json with correct settings for mlx-swift
    metadone_config = {
        "architectures": ["LlavaQwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "freeze_mm_mlp_adapter": False,
        "hidden_act": "silu",
        "hidden_size": 896,
        "image_aspect_ratio": "pad",
        "image_grid_pinpoints": None,
        "image_token_index": 151646,
        "initializer_range": 0.02,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "max_window_layers": 24,
        "mm_hidden_size": 3072,
        "mm_patch_merge_type": "flat",
        "mm_projector_lr": None,
        "mm_projector_type": "mlp2x_gelu",
        "mm_use_im_patch_token": False,
        "mm_use_im_start_end": False,
        "mm_vision_select_feature": "patch",
        "mm_vision_select_layer": -2,
        "mm_vision_tower": "mobileclip_l_1024",
        "model_type": "llava_qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "tie_word_embeddings": False,
        "tokenizer_model_max_length": 8192,
        "tokenizer_padding_side": "right",
        "torch_dtype": "bfloat16",
        "tune_mm_mlp_adapter": False,
        "unfreeze_mm_vision_tower": True,
        "use_cache": True,
        "use_mm_proj": True,
        "use_sliding_window": False,
        "vision_config": {},
        "vocab_size": 151936
    }
    if quantize:
        metadone_config["quantization"] = {"group_size": 64, "bits": 4}
    with open(config_file, "w") as f:
        json.dump(metadone_config, f, indent=4)

    # Create processor_config.json (LlavaProcessor for mlx-swift)
    processor_config = {
        "image_token": "<image>",
        "num_additional_image_tokens": 0,
        "patch_size": 64,
        "processor_class": "LlavaProcessor",
        "vision_feature_select_strategy": None
    }
    with open(os.path.join(output_dir, "processor_config.json"), "w") as f:
        json.dump(processor_config, f, indent=2)

    # Create preprocessor_config.json
    preprocessor_config = {
        "crop_size": {"height": 1024, "width": 1024},
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.0, 0.0, 0.0],
        "image_processor_type": "CLIPImageProcessor",
        "image_std": [1.0, 1.0, 1.0],
        "processor_class": "LlavaProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": 1024}
    }
    with open(os.path.join(output_dir, "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor_config, f, indent=2)

    # Copy tokenizer files from base model
    if model_path:
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "added_tokens.json", "generation_config.json"
        ]
        for tf in tokenizer_files:
            src = os.path.join(model_path, tf)
            dst = os.path.join(output_dir, tf)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Generate tokenizer.json if not present
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path) and model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            print("  Generated tokenizer.json")
        except Exception as e:
            print(f"  WARNING: Failed to generate tokenizer.json: {e}")

    print("  WARNING: fastvithd.mlpackage not found - please copy manually from MetaDone.app")


def _create_mlx_vlm_configs(output_dir: str, model_path: str, quantize: bool):
    """Create mlx-vlm compatible config files (fallback)."""
    import json
    import shutil

    config_file = os.path.join(output_dir, "config.json")

    # Create config.json with mlx-vlm compatible settings
    mlx_vlm_config = {
        "architectures": ["LlavaQwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 896,
        "image_token_index": 151646,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "mm_hidden_size": 3072,
        "mm_projector_type": "mlp2x_gelu",
        "mm_vision_select_feature": "patch",
        "mm_vision_select_layer": -2,
        "mm_vision_tower": "mobileclip_l_1024",
        "model_type": "llava_qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,  # mlx-vlm uses tied embeddings
        "use_cache": True,
        "vocab_size": 151936
    }
    if quantize:
        mlx_vlm_config["quantization"] = {"group_size": 64, "bits": 4}
    with open(config_file, "w") as f:
        json.dump(mlx_vlm_config, f, indent=4)

    # Copy tokenizer and processor files from base model
    if model_path:
        files_to_copy = [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "added_tokens.json", "generation_config.json",
            "preprocessor_config.json", "processor_config.json"
        ]
        for f in files_to_copy:
            src = os.path.join(model_path, f)
            dst = os.path.join(output_dir, f)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Generate tokenizer.json if not present
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path) and model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            print("  Generated tokenizer.json")
        except Exception as e:
            print(f"  WARNING: Failed to generate tokenizer.json: {e}")

    print("\nModel finalization complete!")


def train(
    dataset_path: str,
    output_dir: str,
    model_name: str = "apple/FastVLM-0.5B",
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    max_samples: int = None,
    quantize: bool = False,
    keep_adapter: bool = False,
    hf_token: str = None,
    dropout: float = 0.0,
    warmup_ratio: float = 0.0,
    target: str = "mlx-vlm",
):
    """Main training function.

    Args:
        target: Output format - "mlx-vlm" for Python inference (default),
                "mlx-swift" for MetaDone/iOS apps
    """

    print(f"\n{'='*60}")
    print("FastVLM MLX Native Training")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    if dropout > 0:
        print(f"LoRA dropout: {dropout}")
    if warmup_ratio > 0:
        print(f"Warmup ratio: {warmup_ratio} (cosine scheduler)")
    print(f"Quantize: {'Yes (4-bit)' if quantize else 'No (16-bit)'}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")

    # Use local model path if available
    local_model = Path.home() / ".cache/huggingface/hub/models--apple--FastVLM-0.5B/snapshots"
    if local_model.exists():
        snapshots = list(local_model.glob("*"))
        if snapshots:
            model_path = str(snapshots[0])
            print(f"Using local model: {model_path}")
    else:
        model_path = model_name

    model, tokenizer, image_processor = load_fastvlm_model(model_path)

    print(f"Model type: {type(model).__name__}")

    # Apply LoRA to language model
    print("\nApplying LoRA to language model...")

    modified_count = 0
    lora_layers = {}

    # FastVLM structure: language_model.model.layers[i].self_attn.{q,k,v,o}_proj
    #                    language_model.model.layers[i].mlp.{gate,up,down}_proj
    lm_model = model.language_model.model
    layers = lm_model.layers

    print(f"  Found {len(layers)} transformer layers")

    for i, layer in enumerate(layers):
        # Self attention projections
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if hasattr(layer.self_attn, proj_name):
                orig_linear = getattr(layer.self_attn, proj_name)
                if isinstance(orig_linear, nn.Linear):
                    lora_layer = LoRALinear.from_linear(orig_linear, rank=lora_rank, alpha=lora_alpha, dropout=dropout)
                    setattr(layer.self_attn, proj_name, lora_layer)
                    path = f"layers[{i}].self_attn.{proj_name}"
                    lora_layers[path] = lora_layer
                    modified_count += 1

        # MLP projections
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, proj_name):
                orig_linear = getattr(layer.mlp, proj_name)
                if isinstance(orig_linear, nn.Linear):
                    lora_layer = LoRALinear.from_linear(orig_linear, rank=lora_rank, alpha=lora_alpha, dropout=dropout)
                    setattr(layer.mlp, proj_name, lora_layer)
                    path = f"layers[{i}].mlp.{proj_name}"
                    lora_layers[path] = lora_layer
                    modified_count += 1

    print(f"Modified {modified_count} layers with LoRA")

    # Count trainable parameters
    total_lora_params = 0
    for name, layer in lora_layers.items():
        total_lora_params += layer.lora_a.size + layer.lora_b.size
    print(f"Trainable LoRA parameters: {total_lora_params:,}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(dataset_path, max_samples)
    print(f"Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print("ERROR: No samples found in dataset!")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # Setup training - only LoRA parameters will be updated
    # ================================================================
    # We don't freeze the model. Instead, we filter gradients to only
    # update LoRA parameters (lora_a and lora_b). This is simpler and
    # more reliable than freeze/unfreeze which doesn't work well with
    # plain mx.array attributes.
    print("\nLoRA training mode: only lora_a and lora_b will be updated")

    # ================================================================
    # Setup learning rate scheduler (cosine with warmup)
    # ================================================================
    total_steps = epochs * math.ceil(len(dataset) / batch_size)
    warmup_steps = int(total_steps * warmup_ratio)

    def get_lr(step: int) -> float:
        """Get learning rate for given step with warmup + cosine decay."""
        if warmup_ratio <= 0:
            return learning_rate

        if step < warmup_steps:
            # Linear warmup
            return learning_rate * (step + 1) / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    # Setup optimizer for LoRA parameters only
    print("Setting up optimizer...")
    if warmup_ratio > 0:
        print(f"  Using cosine scheduler: {warmup_steps} warmup steps, {total_steps} total steps")
    optimizer = optim.Adam(learning_rate=learning_rate)

    # ================================================================
    # Helper function to zero out non-LoRA gradients
    # ================================================================
    def zero_non_lora_grads(grads):
        """
        Zero out gradients for non-LoRA parameters while preserving structure.
        This ensures only lora_a and lora_b are updated by the optimizer.
        """
        if isinstance(grads, dict):
            result = {}
            for k, v in grads.items():
                if k in ("lora_a", "lora_b"):
                    # Keep LoRA gradients as-is
                    result[k] = v
                elif isinstance(v, dict):
                    # Recurse into nested dicts
                    result[k] = zero_non_lora_grads(v)
                elif isinstance(v, mx.array):
                    # Zero out non-LoRA gradients
                    result[k] = mx.zeros_like(v)
                else:
                    result[k] = v
            return result
        return grads

    # ================================================================
    # Define loss function for nn.value_and_grad()
    # ================================================================
    # Get image token index from model config
    image_token_index = getattr(model.config, 'image_token_index', -200)
    # FastVLM expands each image token to this many visual embeddings
    num_image_tokens = 256  # FastVLM uses 256 visual tokens per image

    def loss_fn(model, input_ids, pixel_values, mask):
        """
        Compute cross-entropy loss for a single sample.
        This function handles image token expansion - the model expands each
        image token (-200) into multiple visual embeddings (256 tokens).

        Strategy: Build expanded labels that match logits shape by inserting
        ignore tokens (-100) at visual embedding positions.
        """
        # Forward pass
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=mask,
        )

        # Extract logits
        logits = output.logits if hasattr(output, 'logits') else output

        # Handle image token expansion
        # Input: (1, input_len) with image_token_index at some position
        # Output: (1, output_len, vocab_size) where output_len = input_len - 1 + num_image_tokens
        input_len = input_ids.shape[1]
        output_len = logits.shape[1]

        # Find position of image token in input_ids
        input_ids_flat = input_ids[0]  # Shape: (input_len,)

        # Build expanded labels using concatenation (MLX-friendly)
        # Structure: [tokens_before_image, ignore_tokens_for_image, tokens_after_image]

        image_pos = None
        for i in range(input_len):
            if input_ids_flat[i].item() == image_token_index:
                image_pos = i
                break

        if image_pos is not None:
            # Tokens before image (for next-token prediction, shift by 1)
            before_image = input_ids_flat[1:image_pos+1] if image_pos > 0 else mx.array([], dtype=mx.int32)

            # Ignore tokens for visual embeddings (-100)
            ignore_visual = mx.full((num_image_tokens,), -100, dtype=mx.int32)

            # Tokens after image
            after_image = input_ids_flat[image_pos+2:] if image_pos + 1 < input_len else mx.array([], dtype=mx.int32)

            # Concatenate to form labels
            parts = []
            if before_image.size > 0:
                parts.append(before_image)
            parts.append(ignore_visual)
            if after_image.size > 0:
                parts.append(after_image)

            if len(parts) > 1:
                labels = mx.concatenate(parts)
            else:
                labels = parts[0]

            # Pad or trim to match output_len - 1 (for shifted prediction)
            target_len = output_len - 1
            if labels.size < target_len:
                padding = mx.full((target_len - labels.size,), -100, dtype=mx.int32)
                labels = mx.concatenate([labels, padding])
            elif labels.size > target_len:
                labels = labels[:target_len]
        else:
            # No image token, standard next-token prediction
            labels = input_ids_flat[1:]
            target_len = output_len - 1
            if labels.size < target_len:
                padding = mx.full((target_len - labels.size,), -100, dtype=mx.int32)
                labels = mx.concatenate([labels, padding])

        # Shift logits for next-token prediction
        shift_logits = logits[0, :-1, :]  # (output_len-1, vocab_size)

        # Create mask for valid (non-ignored) positions
        valid_mask = (labels >= 0).astype(mx.float32)
        num_valid = mx.sum(valid_mask)

        # Replace -100 with 0 for safe indexing
        safe_labels = mx.maximum(labels, 0)

        # Compute log softmax for numerical stability
        log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)

        # Gather log probs for target tokens
        gathered = mx.take_along_axis(log_probs, safe_labels[:, None], axis=1)[:, 0]

        # Apply mask and compute mean loss (only on valid tokens)
        masked_loss = -gathered * valid_mask
        loss = mx.sum(masked_loss) / mx.maximum(num_valid, mx.array(1.0))

        return loss

    # Create the value_and_grad function
    # This computes both the loss and gradients in a single pass
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # ================================================================
    # Training loop with proper gradient computation
    # ================================================================
    print(f"\nTraining for {epochs} epochs...")
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    global_step = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        total_loss = 0.0
        num_samples = 0

        for step in range(steps_per_epoch):
            # Get batch
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch = dataset[start_idx:end_idx]

            batch_loss = mx.array(0.0)
            batch_grads = None
            valid_samples = 0

            for item in batch:
                try:
                    # Get image path and caption
                    image_path = item["image"]
                    caption = item["caption"]

                    # Create prompt with image token (FastVLM/Qwen2 format)
                    # The <image> token will be replaced by image_token_index (-200) by prepare_inputs
                    prompt = f"<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n{caption}<|im_end|>"

                    # Use mlx_vlm's prepare_inputs to properly handle image + text fusion
                    # This correctly inserts the image token at the right position
                    from mlx_vlm.generate import prepare_inputs

                    inputs = prepare_inputs(
                        image_processor,  # This is actually the full processor now
                        images=[image_path],
                        prompts=prompt,
                        image_token_index=model.config.image_token_index,
                    )

                    input_ids = inputs["input_ids"]  # Shape: (1, seq_len) with image token inserted
                    pixel_values = inputs["pixel_values"]  # Shape: (1, C, H, W)
                    mask = inputs.get("attention_mask", mx.ones_like(input_ids))

                    # ============================================
                    # COMPUTE LOSS AND GRADIENTS
                    # ============================================
                    loss, grads = loss_and_grad_fn(model, input_ids, pixel_values, mask)

                    # Accumulate batch loss
                    batch_loss = batch_loss + loss
                    valid_samples += 1

                    # Accumulate gradients
                    if batch_grads is None:
                        batch_grads = grads
                    else:
                        # Add gradients element-wise
                        batch_grads = tree_map(lambda a, b: a + b, batch_grads, grads)

                except Exception as e:
                    print(f"Error processing sample: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Average gradients over batch and apply update
            if valid_samples > 0 and batch_grads is not None:
                # Average the gradients
                batch_grads = tree_map(lambda g: g / valid_samples, batch_grads)
                batch_loss = batch_loss / valid_samples

                # Force MLX computation to check for NaN
                mx_eval = mx.eval  # MLX's eval for lazy computation, not Python's eval
                mx_eval(batch_loss)
                loss_value = batch_loss.item()

                # Skip update if loss is NaN or Inf
                if not (math.isnan(loss_value) or math.isinf(loss_value)):
                    # ============================================
                    # ZERO OUT NON-LoRA GRADIENTS
                    # ============================================
                    lora_grads = zero_non_lora_grads(batch_grads)

                    # ============================================
                    # GRADIENT CLIPPING for stability
                    # ============================================
                    def clip_grad(g):
                        if isinstance(g, mx.array):
                            return mx.clip(g, -1.0, 1.0)
                        return g
                    lora_grads = tree_map(clip_grad, lora_grads)

                    # ============================================
                    # UPDATE LEARNING RATE (scheduler)
                    # ============================================
                    if warmup_ratio > 0:
                        current_lr = get_lr(global_step)
                        optimizer.learning_rate = current_lr

                    # ============================================
                    # UPDATE PARAMETERS WITH OPTIMIZER
                    # ============================================
                    optimizer.update(model, lora_grads)
                    global_step += 1

                    # Force MLX computation
                    state = [model.parameters(), optimizer.state]
                    mx_eval(state)

                    total_loss += loss_value
                    num_samples += 1
                else:
                    print(f"  Warning: Skipping step due to NaN/Inf loss")

                if step % 10 == 0:
                    # Format expected by Swift: "Epoch X/Y, Step X/Y, Loss: X.XX"
                    print(f"Epoch {epoch + 1}/{epochs}, Step {step + 1}/{steps_per_epoch}, Loss: {loss_value:.4f}", flush=True)

        avg_loss = total_loss / max(num_samples, 1)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    # Save LoRA adapter
    print("\nSaving LoRA adapter...")

    # Collect final LoRA weights
    adapter_weights = {}
    for name, layer in lora_layers.items():
        adapter_weights[f"{name}.lora_a"] = layer.lora_a
        adapter_weights[f"{name}.lora_b"] = layer.lora_b

    adapter_path = os.path.join(output_dir, "adapter.safetensors")
    mx.save_safetensors(adapter_path, adapter_weights)

    # Save config
    config = {
        "base_model": model_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "training_samples": len(dataset),
        "modified_layers": list(lora_layers.keys()),
    }

    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nAdapter saved to: {output_dir}")

    # Finalize model (convert to target format and merge LoRA)
    print(f"\n{'='*60}")
    print(f"Finalizing model for {target}...")
    print(f"{'='*60}")
    finalize_model(output_dir, model_name, adapter_weights, lora_rank=lora_rank, lora_alpha=lora_alpha, quantize=quantize, keep_adapter=keep_adapter, hf_token=hf_token, target=target)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Model ready at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="FastVLM Native MLX Training with LoRA",
    )

    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to dataset (CSV folder or JSONL)"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for LoRA adapter"
    )

    parser.add_argument(
        "--model", "-m",
        default="apple/FastVLM-0.5B",
        help="Base FastVLM model"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=1,
        help="Number of epochs"
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size"
    )

    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )

    parser.add_argument(
        "--lora-rank", "-r",
        type=int,
        default=8,
        help="LoRA rank"
    )

    parser.add_argument(
        "--lora-alpha", "-a",
        type=int,
        default=16,
        help="LoRA alpha"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use"
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Quantize the final model to 4-bit"
    )

    parser.add_argument(
        "--keep-adapter",
        action="store_true",
        default=False,
        help="Keep adapter.safetensors and adapter_config.json after merging (for re-merging later)"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token for authentication"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate (0.0-1.0, default: 0.0)"
    )

    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Warmup ratio for cosine scheduler (0.0-1.0, default: 0.0 = no warmup)"
    )

    parser.add_argument(
        "--target",
        type=str,
        choices=["mlx-vlm", "mlx-swift"],
        default="mlx-vlm",
        help="Output format: 'mlx-vlm' for Python inference (default), 'mlx-swift' for MetaDone/iOS apps"
    )

    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_samples=args.max_samples,
        quantize=args.quantize,
        keep_adapter=args.keep_adapter,
        hf_token=args.hf_token,
        dropout=args.dropout,
        warmup_ratio=args.warmup_ratio,
        target=args.target,
    )


if __name__ == "__main__":
    main()
