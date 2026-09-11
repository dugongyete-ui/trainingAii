"""Background CPU training helpers used by the Dzeck WebUI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "out"
RUN_FILE = OUT_DIR / ".training-run.json"
LOG_FILE = OUT_DIR / "training.log"
PRETRAIN_DATA = ROOT_DIR / "dataset" / "cpu_pretrain_subset.jsonl"
SFT_DATA = ROOT_DIR / "dataset" / "dzeck_id_sft.jsonl"


def _read_run() -> dict[str, Any]:
    try:
        return json.loads(RUN_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_run(data: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _tail_log(limit: int = 5000) -> str:
    try:
        return LOG_FILE.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def get_status() -> dict[str, Any]:
    run = _read_run()
    pid = run.get("pid")
    active = isinstance(pid, int) and _pid_alive(pid)
    if run and not active:
        run["active"] = False
        _write_run(run)
    return {
        "active": active,
        "stage": run.get("stage", ""),
        "pid": pid if active else None,
        "hidden_size": run.get("hidden_size"),
        "layers": run.get("layers"),
        "started_at": run.get("started_at", ""),
        "log": _tail_log(),
        "run": run,
    }


def start_training(
    stage: str,
    *,
    epochs: int,
    hidden_size: int,
    layers: int,
    batch_size: int,
    max_seq_len: int,
) -> dict[str, Any]:
    status = get_status()
    if status["active"]:
        raise RuntimeError("Training masih berjalan.")
    if stage not in {"pretrain", "sft"}:
        raise ValueError("Tahap training tidak dikenal.")
    if not PRETRAIN_DATA.is_file() or not SFT_DATA.is_file():
        raise FileNotFoundError("Dataset training belum tersedia.")
    if stage == "sft":
        pretrain_checkpoint = OUT_DIR / f"pretrain_{hidden_size}.pth"
        if not pretrain_checkpoint.is_file():
            raise FileNotFoundError(
                "Checkpoint pretraining belum ada. Jalankan pretraining terlebih dahulu."
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")
    script = "train_pretrain.py" if stage == "pretrain" else "train_full_sft.py"
    data_path = PRETRAIN_DATA if stage == "pretrain" else SFT_DATA
    command = [
        sys.executable,
        script,
        "--save_dir",
        str(OUT_DIR),
        "--save_weight",
        "pretrain" if stage == "pretrain" else "full_sft",
        "--data_path",
        str(data_path),
        "--from_weight",
        "none" if stage == "pretrain" else "pretrain",
        "--from_resume",
        "1",
        "--device",
        "cpu",
        "--dtype",
        "bfloat16",
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--hidden_size",
        str(hidden_size),
        "--num_hidden_layers",
        str(layers),
        "--max_seq_len",
        str(max_seq_len),
        "--num_workers",
        "0",
        "--accumulation_steps",
        "1",
        "--log_interval",
        "5",
        "--save_interval",
        "25",
        "--use_compile",
        "0",
    ]
    environment = os.environ.copy()
    environment["DZECK_DEVICE"] = "cpu"
    environment["OMP_NUM_THREADS"] = environment.get("DZECK_CPU_THREADS", "4")
    environment["MKL_NUM_THREADS"] = environment["OMP_NUM_THREADS"]
    with LOG_FILE.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR / "trainer"),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    _write_run(
        {
            "pid": process.pid,
            "active": True,
            "stage": stage,
            "hidden_size": hidden_size,
            "layers": layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "command": command,
        }
    )
    return get_status()


def stop_training() -> dict[str, Any]:
    status = get_status()
    pid = status.get("pid")
    if not status["active"] or not isinstance(pid, int):
        return status
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    run = _read_run()
    run["active"] = False
    _write_run(run)
    return get_status()


def checkpoint_path(stage: str, hidden_size: int) -> Path:
    weight = "pretrain" if stage == "pretrain" else "full_sft"
    return OUT_DIR / f"{weight}_{hidden_size}.pth"


def export_checkpoint(stage: str, hidden_size: int, layers: int) -> tuple[bool, str]:
    checkpoint = checkpoint_path(stage, hidden_size)
    if not checkpoint.is_file():
        return False, f"Checkpoint belum ditemukan: {checkpoint.name}"
    output_dir = ROOT_DIR / "dzeck-model"
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "export_dzeck.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_dir),
        "--hidden-size",
        str(hidden_size),
        "--layers",
        str(layers),
    ]
    result = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode != 0:
        return False, output or "Ekspor checkpoint gagal."
    return True, output or f"Checkpoint Dzeck diekspor ke {output_dir}."