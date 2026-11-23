"""Create an allowlisted source ZIP, excluding data/weights/results regardless of .gitignore."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT_FILES = {"README.md", "README_zh-CN.md", "pyproject.toml", "requirements.txt", "requirements-extract.txt",
              ".gitignore", ".gitattributes", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"}
TREES = {"src", "scripts", "experiments", "configs", "tests", "docs", "licenses", ".github"}
EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json"}
# Top-level datasets are already excluded by TREES. Keep src/data: it is code.
SKIP = {"__pycache__", ".pytest_cache", ".git", ".venv", "runs", "cache", "weights", "outputs"}


def selected_files(root):
    root = Path(root).resolve()
    selected = []
    for name in sorted(ROOT_FILES):
        if (root / name).is_file():
            selected.append(root / name)
    for directory in sorted(TREES):
        tree = root / directory
        if not tree.exists():
            continue
        if tree.is_symlink() or getattr(tree, "is_junction", lambda: False)():
            raise ValueError(f"Source tree cannot be a link: {tree}")
        for path in tree.rglob("*"):
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise ValueError(f"Source entry cannot be a link: {path}")
            if any(part in SKIP for part in path.relative_to(tree).parts):
                continue
            if path.is_file() and path.suffix in EXTENSIONS:
                if not path.resolve().is_relative_to(root) or path.stat().st_size > 2_000_000:
                    raise ValueError(f"Unexpected release source: {path}")
                path.read_text(encoding="utf-8")  # binary payloads are never accepted
                selected.append(path)
    if not (root / "README.md").is_file():
        raise ValueError("README.md is required")
    return sorted(set(selected), key=lambda p: p.relative_to(root).as_posix())


def build_archive(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = selected_files(root)
    if output in paths:
        raise ValueError("Archive path collides with source")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            info = zipfile.ZipInfo("MERC-GitHub/" + relative, date_time=(2026, 11, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
            manifest.append(f"{hashlib.sha256(payload).hexdigest()}  {relative}")
        archive.writestr("MERC-GitHub/SHA256SUMS.txt", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity verification failed")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {"files": len(paths) + 1, "bytes": output.stat().st_size, "sha256": digest}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, default=Path("dist/MERC-GitHub-source.zip"))
    args = p.parse_args()
    print(build_archive(args.root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
