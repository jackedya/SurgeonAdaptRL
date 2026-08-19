from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXCLUDED_DIRECTORIES = {".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__", "verification"}
EXCLUDED_FILES = {"integrity_manifest.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return path.name not in EXCLUDED_FILES and not any(part in EXCLUDED_DIRECTORIES for part in relative.parts)


def build(root: Path) -> dict[str, object]:
    files = []
    code_lines = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and included(candidate, root)):
        relative = path.relative_to(root).as_posix()
        lines = count_lines(path)
        if path.suffix in {".py", ".sh"}:
            code_lines += lines
        files.append({"path": relative, "lines": lines, "sha256": sha256(path), "bytes": path.stat().st_size})
    return {
        "schema_version": 1,
        "project_status": "PARTIALLY_VERIFIED",
        "code_lines": code_lines,
        "file_count": len(files),
        "files": files,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "integrity_manifest.json"
    destination.write_text(json.dumps(build(root), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
