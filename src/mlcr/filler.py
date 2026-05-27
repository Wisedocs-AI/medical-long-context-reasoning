from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BuiltContext:
    text: str | None
    images: list[Path]
    filler_text_files: list[Path] = field(default_factory=list)
    filler_image_files: list[Path] = field(default_factory=list)
    insertion_positions: list[int] = field(default_factory=list)
    real_page_count: int = 0
    filler_page_count: int = 0


def make_rng(global_seed: int, row_key: str) -> random.Random:
    """Deterministic per-row RNG derived from (global_seed, row_key)."""
    h = hashlib.blake2b(f"{global_seed}:{row_key}".encode(), digest_size=16).digest()
    return random.Random(int.from_bytes(h, "big"))


def build_context(
    *,
    case_summaries: list[Path],
    case_images: list[Path],
    filler_text_files: list[Path],
    filler_image_files: list[Path],
    modality: str,
    ablation_ratio: float,
    rng: random.Random,
) -> BuiltContext:
    """Build the final text + image context with deterministic filler insertion.

    `modality` is one of {"text", "image", "both"}.
    `ablation_ratio` is the number of filler pages relative to the real case pages:
    0 = no filler, 1 = as many filler pages as real pages, 2 = twice as many, etc.
    One filler file (text or image) equals one page.

    Sampling is modality-independent: for a given case, ratio, and seed the same
    filler pages are selected and placed at the same positions regardless of
    whether the modality is text, image, or both.  This guarantees that results
    are directly comparable across modalities.
    """
    ctx = BuiltContext(text=None, images=[])

    
    n_real_pages = len(case_summaries)
    ctx.real_page_count = n_real_pages

    if ablation_ratio <= 0 or n_real_pages == 0:
        ctx.text = _join_text(case_summaries) if modality in ("text", "both") else None
        ctx.images = list(case_images) if modality in ("image", "both") else []
        return ctx

    n_filler = round(n_real_pages * ablation_ratio)
    if n_filler <= 0:
        ctx.text = _join_text(case_summaries) if modality in ("text", "both") else None
        ctx.images = list(case_images) if modality in ("image", "both") else []
        return ctx

    # --- Sampling (modality-independent, consumes RNG in a fixed order) ---
    # Filler pools are paired: filler_text_files[i] and filler_image_files[i]
    # represent the same page. Draw one set of indices used for both channels.
    
    pool_size = len(filler_text_files) or 1
    page_indices = [rng.randrange(pool_size) for _ in range(n_filler)]

    # Insertion positions — one set, applied uniformly to both channels.
    positions = sorted(rng.randrange(n_real_pages + 1) for _ in range(n_filler))
    ctx.insertion_positions = positions
    ctx.filler_page_count = n_filler

    # --- Materialize text channel ---
    if modality in ("text", "both") and filler_text_files:
        picked = [filler_text_files[i] for i in page_indices]
        ctx.filler_text_files = picked
        real_chunks = [_read(p) for p in case_summaries]
        filler_chunks = [_read(p) for p in picked]
        ctx.text = _interleave_text(real_chunks, filler_chunks, positions)
    elif modality in ("text", "both"):
        ctx.text = _join_text(case_summaries)

    # --- Materialize image channel ---
    if modality in ("image", "both") and filler_image_files:
        picked = [filler_image_files[i] for i in page_indices]
        ctx.filler_image_files = picked
        ctx.images = _interleave_paths(list(case_images), picked, positions)
    elif modality in ("image", "both"):
        ctx.images = list(case_images)

    return ctx


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _join_text(paths: list[Path]) -> str:
    return "\n\n".join(_read(p) for p in paths)


def _interleave_text(
    real_chunks: list[str],
    filler_chunks: list[str],
    positions: list[int],
) -> str:
    """Interleave filler chunks into real chunks at the given sorted positions."""
    out: list[str] = []
    pi = 0
    for i in range(len(real_chunks) + 1):
        while pi < len(positions) and positions[pi] == i:
            out.append(filler_chunks[pi])
            pi += 1
        if i < len(real_chunks):
            out.append(real_chunks[i])
    return "\n\n".join(out)


def _interleave_paths(
    real: list[Path],
    filler: list[Path],
    positions: list[int],
) -> list[Path]:
    """Interleave filler paths into real paths at the given sorted positions."""
    out: list[Path] = []
    pi = 0
    for i in range(len(real) + 1):
        while pi < len(positions) and positions[pi] == i:
            out.append(filler[pi])
            pi += 1
        if i < len(real):
            out.append(real[i])
    return out
