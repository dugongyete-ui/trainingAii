---
name: Dzeck model provenance
description: The naming rule for distinguishing the Dzeck project model from its third-party foundation checkpoint.
---

The project may expose a stable public model ID such as `dzeck-large-id`, but that
does not mean the local weights were trained by Dzeck. A Qwen foundation checkpoint
must remain visibly identified as a foundation model until a Dzeck training result
has been exported and marked as a trained checkpoint.

**Why:** The existing Indonesian-labeled folder was a direct Qwen2.5 checkpoint,
which caused confusion about whether it was the user's own model.

**How to apply:** Keep public API/UI naming, source model, parameter size, and
training status as separate fields. Treat exported training manifests or native
MiniMind checkpoints as Dzeck-owned; label plain Qwen directories as foundation
models even when the public API uses the Dzeck name.