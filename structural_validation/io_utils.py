"""Atomic output helpers shared by structural-validation CLIs."""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from omegaconf import OmegaConf


def normalize_json_value(value):
    """Recursively convert supported metadata values to JSON-safe builtins."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return value
    if isinstance(value, str):
        return str(value)
    if type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Metadata contains a non-finite floating-point value.")
        return value
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Metadata mapping keys must be strings.")
            normalized[key] = normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    raise TypeError(
        f"Metadata value of type {type(value).__name__} is not JSON serializable."
    )


def serialize_json_value(payload):
    """Normalize and serialize metadata using standard finite JSON values."""
    normalized = normalize_json_value(payload)
    return json.dumps(normalized, indent=2, allow_nan=False) + "\n"


@contextmanager
def output_set_transaction(output_dir, filenames):
    """Stage and rollback-safely commit a complete named output set."""
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    names = tuple(filenames)
    if not names or len(set(names)) != len(names):
        raise ValueError("Output transaction filenames must be nonempty and unique.")
    if any(Path(name).name != name for name in names):
        raise ValueError("Output transaction filenames must be plain basenames.")

    stage_dir = Path(tempfile.mkdtemp(
        dir=str(destination_dir), prefix=".apo2mol-output-stage-"
    ))
    backup_dir = None
    staged = {name: stage_dir / name for name in names}
    try:
        yield staged
        missing = [name for name, path in staged.items() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Output transaction is missing staged files: {missing}")

        backup_dir = Path(tempfile.mkdtemp(
            dir=str(destination_dir), prefix=".apo2mol-output-backup-"
        ))
        backed_up = []
        committed = []
        try:
            for name in names:
                destination = destination_dir / name
                if destination.exists():
                    os.replace(destination, backup_dir / name)
                    backed_up.append(name)
            for name in names:
                destination = destination_dir / name
                os.replace(staged[name], destination)
                committed.append(name)
        except BaseException:
            for name in reversed(committed):
                destination = destination_dir / name
                if destination.exists():
                    destination.unlink()
            for name in reversed(backed_up):
                backup = backup_dir / name
                if backup.exists():
                    os.replace(backup, destination_dir / name)
            raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)


@contextmanager
def atomic_output_path(path):
    """Yield a same-directory temporary path and atomically replace on success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    handle.close()
    temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(frame, path, index=False):
    """Write a pandas-compatible frame without exposing a partial destination."""
    with atomic_output_path(path) as temporary:
        frame.to_csv(temporary, index=index)


def atomic_write_json(payload, path):
    """Write formatted JSON without exposing a partial destination."""
    with atomic_output_path(path) as temporary:
        temporary.write_text(
            serialize_json_value(payload),
            encoding="utf-8",
        )
