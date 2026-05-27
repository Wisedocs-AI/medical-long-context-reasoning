from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTS = {".txt", ".md"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Per-case file mapping each prompt to a difficulty label. It is an evaluation
# annotation only: the label is recorded in outputs/metadata but is NEVER added
# to the text sent to the model (see runner._compose_prompt).
DIFFICULTY_FILENAME = "difficulties.csv"


@dataclass
class Case:
    uuid: str
    root: Path
    prompts: list[Path]
    summaries: list[Path]
    images: list[Path]


def _list_sorted(d: Path, exts: set[str]) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts)


def load_case(repo_root: Path, uuid: str) -> Case:
    root = repo_root / "cases" / uuid
    return Case(
        uuid=uuid,
        root=root,
        prompts=_list_sorted(root / "prompts", TEXT_EXTS),
        summaries=_list_sorted(root / "summaries", TEXT_EXTS),
        images=_list_sorted(root / "images", IMG_EXTS),
    )


def _prompt_key(name: str) -> str:
    """Normalize a prompt reference to its bare stem for difficulty lookup.

    Accepts a filename (``q01.txt``), a stem (``q01``), or a full path; returns
    the lowercased stem (``q01``)."""
    return Path(name).stem.strip().lower()


def load_prompt_difficulties(case_root: Path) -> dict[str, str]:
    """Map prompt stem -> difficulty label for a case.

    Reads ``cases/<case>/difficulties.csv`` (header: ``prompt,difficulty``). The
    ``prompt`` column may be a filename (``q01.txt``) or a stem (``q01``).
    Returns ``{}`` when the file is absent or unreadable.

    Difficulty labels are evaluation annotations only and are never sent to the
    model; they exist so results can be sliced/labeled by difficulty.
    """
    path = case_root / DIFFICULTY_FILENAME
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = (row.get("prompt") or "").strip()
                difficulty = (row.get("difficulty") or "").strip()
                if prompt:
                    out[_prompt_key(prompt)] = difficulty
    except Exception:
        return {}
    return out


def list_filler_text(repo_root: Path, *, filler_subdir: str = "empty") -> list[Path]:
    return _list_sorted(repo_root / "filler_files" / filler_subdir / "ocr", TEXT_EXTS)


def list_filler_images(repo_root: Path, *, filler_subdir: str = "empty") -> list[Path]:
    return _list_sorted(repo_root / "filler_files" / filler_subdir / "images", IMG_EXTS)


def filler_form(path: Path) -> str:
    """Form prefix of a filler file.

    Handles two naming conventions:
      - underscore-separated: `form-1_003.jpg` -> `form-1`
      - variant-style:        `cms-10106-v1-001.jpg` -> `cms-10106`
    """
    stem = path.stem
    # Strip variant+page suffix like -v1-001, -v2-003, -v12-012
    stripped = re.sub(r"-v\d+-\d+$", "", stem)
    if stripped != stem:
        return stripped
    # Fallback: split on underscore (original convention)
    return stem.split("_", 1)[0]


def list_filler_forms(repo_root: Path, *, filler_subdir: str = "empty") -> set[str]:
    """All distinct filler form prefixes found across text and image filler pools."""
    files = list_filler_text(repo_root, filler_subdir=filler_subdir) + list_filler_images(repo_root, filler_subdir=filler_subdir)
    return {filler_form(p) for p in files}


def filter_filler_by_forms(paths: list[Path], allowed_forms: list[str]) -> list[Path]:
    """Keep only filler files whose form prefix is in `allowed_forms`.

    An empty `allowed_forms` means "no restriction" and returns `paths` unchanged.
    """
    if not allowed_forms:
        return paths
    allowed = set(allowed_forms)
    return [p for p in paths if filler_form(p) in allowed]
