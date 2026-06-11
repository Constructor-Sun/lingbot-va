#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.modules.utils import load_text_encoder, load_tokenizer  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/data1/liu/exp/robocasa/datasets/training_no_base/atomic")
DEFAULT_CHECKPOINT_ROOT = Path("/data1/liu/exp/checkpoints/lingbot-va-base")
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_TEXT_LENGTH = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the empty text embedding used for classifier-free guidance "
            "in LingBot-VA post-training."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root where empty_emb.pt will be written by default.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Checkpoint root containing tokenizer/ and text_encoder/.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults to <dataset-root>/empty_emb.pt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Torch device for text encoder inference.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for text encoder inference.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=DEFAULT_MAX_TEXT_LENGTH,
        help="Tokenizer max sequence length for the exported embedding.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Text to encode. Keep the default empty string for CFG empty embedding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def torch_dtype_from_name(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@torch.inference_mode()
def encode_text(text_encoder, tokenizer, device, dtype, text: str, max_text_length: int):
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_text_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_len = int(mask.gt(0).sum(dim=1).item())

    prompt_embeds = text_encoder(input_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = prompt_embeds[0, :seq_len]
    if seq_len < max_text_length:
        padding = prompt_embeds.new_zeros((max_text_length - seq_len, prompt_embeds.shape[1]))
        prompt_embeds = torch.cat([prompt_embeds, padding], dim=0)

    return prompt_embeds.to(torch.bfloat16).cpu()


def main():
    args = parse_args()
    output_path = args.output_path or (args.dataset_root / "empty_emb.pt")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refuse to overwrite existing file: {output_path}")

    checkpoint_root = args.checkpoint_root.resolve()
    tokenizer = load_tokenizer(checkpoint_root / "tokenizer")
    text_encoder = load_text_encoder(
        checkpoint_root / "text_encoder",
        torch_dtype=torch_dtype_from_name(args.dtype),
        torch_device=torch.device(args.device),
    )
    text_encoder.eval()

    empty_emb = encode_text(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=torch.device(args.device),
        dtype=torch_dtype_from_name(args.dtype),
        text=args.text,
        max_text_length=args.max_text_length,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(empty_emb, output_path)
    print(f"Saved empty embedding to {output_path}")
    print(f"shape={tuple(empty_emb.shape)} dtype={empty_emb.dtype}")


if __name__ == "__main__":
    main()
