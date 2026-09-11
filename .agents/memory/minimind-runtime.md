---
name: MiniMind runtime setup
description: Environment-specific setup needed to run MiniMind inference and model downloads.
---

The repository is an inference/training project rather than a conventional application with a built-in test suite. Inference needs both the Python dependencies and a downloaded model or trained checkpoint.

**Why:** The Replit package environment can expose more than one Python interpreter. The dependency directory is attached to the project Python 3.11 interpreter, while the `modelscope` wrapper may resolve to Python 3.12 and fail on compiled NumPy extensions. Generic PyPI PyTorch wheels also pull large CUDA packages and PEP 668 blocks direct writes unless the installer opts in.

**How to apply:** Run project commands with `python3.11`. Install PyTorch from the CPU wheel index with the PEP 668 opt-in, and use `python3.11 -m modelscope.cli.cli` for downloads. The Indonesian default uses the multilingual Qwen checkpoint at `./qwen2.5-0.5b-instruct`; the original Chinese-trained MiniMind checkpoint remains available at `./minimind-3` for comparison. A clean clone has no unit-test files; `compileall` is the available syntax smoke check unless tests are added. The Chinese-trained checkpoint cannot reliably produce natural Indonesian from prompting alone; use the multilingual checkpoint or Indonesian fine-tuning when language quality matters.