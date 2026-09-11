---
name: MiniMind training branches
description: The distinction between native MiniMind training and the larger Qwen foundation used for Dzeck inference.
---

The repository has two separate model paths. The scripts under `trainer/` build
the native MiniMind architecture from its local tokenizer and PyTorch checkpoints.
The `dzeck-large-id` directory is a Transformers-format Qwen2.5 3B foundation for
inference and is not directly compatible with the native `train_pretrain.py`,
`train_full_sft.py`, or `train_lora.py` argument flow.

**Why:** The native trainer constructs `MiniMindConfig` and loads `out/*.pth`,
while the large foundation is a Qwen model with a different architecture and
vocabulary. Treating them as one pipeline would either fail to load or train the
wrong model.

**How to apply:** Explain the native MiniMind path separately from a future
Qwen-compatible LoRA/SFT path. For current knowledge, prefer retrieval over
retraining facts into either model; use fine-tuning mainly for behavior, style,
and Indonesian instruction following.