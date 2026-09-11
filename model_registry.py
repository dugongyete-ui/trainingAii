"""Shared model identity and discovery for the Dzeck command-line tools.

The repository can contain both a foundation checkpoint and checkpoints produced
by the Dzeck training pipeline.  Keeping this information in one place prevents
the CLI, WebUI, and API from presenting a Qwen foundation model as if it had
already been trained by this project.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DZECK_MODEL_ID = os.environ.get("DZECK_MODEL_ID", "dzeck-large-id")
DZECK_DISPLAY_NAME = os.environ.get("DZECK_DISPLAY_NAME", "Dzeck Large ID")
DZECK_BASE_MODEL = os.environ.get(
    "DZECK_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"
)
DZECK_MODEL_DIR = Path(
    os.environ.get("DZECK_MODEL_DIR", str(ROOT_DIR / "dzeck-large-id"))
)
LEGACY_MODEL_DIR = ROOT_DIR / "dzeck-small-id"


@dataclass(frozen=True)
class ModelIdentity:
    """Human- and API-facing information about a local checkpoint."""

    path: Path
    model_id: str
    display_name: str
    parameter_count: int | None
    source_model: str | None
    is_trained_dzeck: bool
    status: str

    @property
    def parameter_label(self) -> str:
        if self.parameter_count is None:
            return "jumlah parameter belum diketahui"
        count = self.parameter_count
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}B parameter"
        return f"{count / 1_000_000:.0f}M parameter"

    @property
    def source_label(self) -> str:
        if self.is_trained_dzeck:
            return "checkpoint hasil training Dzeck"
        if self.source_model:
            return f"fondasi {self.source_model}"
        return "sumber model belum terdeteksi"

    def summary(self) -> str:
        return f"{self.display_name} — {self.parameter_label}; {self.source_label}"


def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "config.json").is_file():
        return True
    return any(
        item.is_file()
        and item.suffix in {".safetensors", ".bin", ".pt", ".pth"}
        for item in path.iterdir()
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_config(path: Path) -> dict[str, Any]:
    return _read_json(path / "config.json")


def _read_manifest(path: Path) -> dict[str, Any]:
    for name in ("dzeck-model.json", "model_identity.json"):
        manifest = _read_json(path / name)
        if manifest:
            return manifest
    return {}


def _source_from_readme(path: Path) -> str | None:
    readme = path / "README.md"
    if not readme.is_file():
        return None
    try:
        text = readme.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"^base_model:\s*([^\n]+)", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    match = re.search(r"Qwen2\.5[-\w.]*\d+(?:\.\d+)?B[-\w.]*", text)
    return match.group(0) if match else None


def _parameter_count(path: Path, config: dict[str, Any]) -> int | None:
    manifest = _read_manifest(path)
    explicit = manifest.get("parameter_count")
    if isinstance(explicit, int) and explicit > 0:
        return explicit

    # A safetensors/bin checkpoint is normally stored in fp16/bf16.  This is
    # an estimate for display only; a manifest or a loaded model is authoritative.
    weight_bytes = sum(
        item.stat().st_size
        for item in path.iterdir()
        if item.is_file() and item.name.endswith((".safetensors", ".bin"))
    )
    if weight_bytes:
        dtype_bytes = 2 if config.get("torch_dtype") in {"float16", "bfloat16"} else 4
        return int(weight_bytes / dtype_bytes)

    hidden_size = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    vocab_size = config.get("vocab_size")
    if all(isinstance(value, int) for value in (hidden_size, layers, vocab_size)):
        return int(vocab_size * hidden_size + layers * hidden_size * hidden_size * 12)
    return None


def _qwen_source(path: Path, config: dict[str, Any]) -> str | None:
    source = _source_from_readme(path)
    if source:
        return source
    if path.resolve() == DZECK_MODEL_DIR.resolve():
        return DZECK_BASE_MODEL
    hidden_size = config.get("hidden_size")
    known_sizes = {
        896: "Qwen/Qwen2.5-0.5B-Instruct",
        1536: "Qwen/Qwen2.5-1.5B-Instruct",
        2048: "Qwen/Qwen2.5-3B-Instruct",
        3584: "Qwen/Qwen2.5-7B-Instruct",
    }
    return known_sizes.get(hidden_size, "Qwen foundation checkpoint")


def _is_trained_dzeck(path: Path, config: dict[str, Any], manifest: dict[str, Any]) -> bool:
    if manifest.get("status") in {"trained", "fine-tuned", "dzeck"}:
        return True
    if config.get("model_type") == "minimind":
        return True
    return False


def get_model_identity(path: str | os.PathLike[str]) -> ModelIdentity:
    model_path = Path(path).expanduser().resolve()
    config = _read_config(model_path)
    manifest = _read_manifest(model_path)
    is_qwen = config.get("model_type") == "qwen2" or any(
        "Qwen" in str(architecture) for architecture in config.get("architectures", [])
    )
    is_trained = _is_trained_dzeck(model_path, config, manifest)
    source_model = (
        manifest.get("base_model")
        if isinstance(manifest.get("base_model"), str)
        else (_qwen_source(model_path, config) if is_qwen else None)
    )

    if is_trained:
        model_id = manifest.get("model_id", DZECK_MODEL_ID)
        display_name = manifest.get("display_name", DZECK_DISPLAY_NAME)
        status = "trained"
    elif model_path.resolve() == DZECK_MODEL_DIR.resolve():
        model_id = DZECK_MODEL_ID
        display_name = DZECK_DISPLAY_NAME
        status = "foundation"
    elif model_path.resolve() == LEGACY_MODEL_DIR.resolve():
        model_id = "qwen2.5-0.5b-instruct"
        display_name = "Qwen2.5 0.5B (legacy)"
        status = "legacy-foundation"
    else:
        model_id = model_path.name
        display_name = model_path.name
        status = "trained" if is_trained else "foundation"

    return ModelIdentity(
        path=model_path,
        model_id=str(model_id),
        display_name=str(display_name),
        parameter_count=_parameter_count(model_path, config),
        source_model=source_model,
        is_trained_dzeck=is_trained,
        status=status,
    )


def default_model_path() -> Path:
    """Return the preferred local model, with a clear legacy fallback."""

    requested = os.environ.get("DZECK_MODEL_PATH")
    candidates = [Path(requested).expanduser() if requested else DZECK_MODEL_DIR]
    candidates.extend((LEGACY_MODEL_DIR, ROOT_DIR / "qwen2.5-0.5b-instruct", ROOT_DIR / "model"))
    for candidate in candidates:
        if _is_model_dir(candidate):
            return candidate.resolve()
    return DZECK_MODEL_DIR.resolve()


def discover_model_paths() -> list[Path]:
    """Find local Transformers/PyTorch model directories for the WebUI."""

    paths: list[Path] = []
    for base_dir in (ROOT_DIR, ROOT_DIR / "scripts"):
        if not base_dir.is_dir():
            continue
        for child in sorted(base_dir.iterdir(), key=lambda item: item.name, reverse=True):
            if child.name.startswith((".", "_")) or child in paths:
                continue
            if _is_model_dir(child):
                paths.append(child.resolve())
    preferred = default_model_path()
    if _is_model_dir(preferred) and preferred not in paths:
        paths.insert(0, preferred)
    return paths