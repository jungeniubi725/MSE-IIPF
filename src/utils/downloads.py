"""Streaming downloads, integrity checks and constrained archive extraction."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def fetch_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "MERC-source-release/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def download(url, target, checksum=None, expected_size=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    algorithm, expected = checksum.split(":", 1) if checksum else ("sha256", None)

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, algorithm).hexdigest()

    if target.exists():
        if expected_size is not None and target.stat().st_size != expected_size:
            raise ValueError(f"Existing file has wrong size: {target}; remove it and retry")
        if expected and digest(target) != expected:
            raise ValueError(f"Existing file checksum mismatch: {target}; remove it and retry")
        return {"url": url, "file": target.name, "algorithm": algorithm, "digest": digest(target), "cached": True}
    partial = target.with_name(target.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MERC-source-release/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError(f"Incomplete download: {target.name}")
            actual = digest(partial)
            if expected and actual != expected:
                raise ValueError(f"Checksum mismatch: {target.name}")
            partial.replace(target)
            return {"url": url, "file": target.name, "algorithm": algorithm, "digest": actual, "cached": False}
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def extract_ravdess(archive_path, destination):
    """Extract only actor media; exclude redundant modality-02 video and metadata."""
    destination = Path(destination).resolve()
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        selected = []
        for member in archive.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            if ".." in parts or member.filename.startswith(("/", "\\")):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.is_dir():
                continue
            name = parts[-1]
            match = re.fullmatch(r"(01|03)-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-(\d{2})\.(mp4|wav)", name)
            if not match:
                continue
            target = (destination / f"Actor_{match.group(2)}" / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Extraction escaped destination")
            selected.append((member, target))
        for member, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise ValueError(f"Existing media size mismatch: {target}")
                continue
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def parse_sdk_splits(source):
    """Read published literal lists without executing downloaded Python code."""
    aliases = {"standard_train_fold": "train", "standard_valid_fold": "val", "standard_test_fold": "test"}
    splits = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in aliases:
                    values = ast.literal_eval(node.value)
                    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                        raise ValueError("Malformed SDK split list")
                    splits[aliases[target.id]] = values
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Missing SDK split lists")
    if any(set(splits[a]) & set(splits[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Overlapping official splits")
    return splits

