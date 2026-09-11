"""Export a native MiniMind checkpoint as a model owned by Dzeck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


ROOT_DIR = Path(__file__).resolve().parents[1]


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    hidden_size: int,
    layers: int,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        max_position_embeddings=32768,
        use_moe=False,
    )
    model = MiniMindForCausalLM(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.float().eval()
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ROOT_DIR / "model")
    tokenizer.save_pretrained(output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    manifest = {
        "model_id": "dzeck-mini-id",
        "display_name": "Dzeck Mini ID",
        "status": "trained",
        "training_stage": checkpoint_path.stem,
        "base_model": "MiniMind native architecture",
        "parameter_count": parameter_count,
        "language": "id",
    }
    (output_dir / "dzeck-model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dzeck checkpoint tersimpan di {output_dir}")
    print(f"Parameter: {parameter_count:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.hidden_size, args.layers)


if __name__ == "__main__":
    main()