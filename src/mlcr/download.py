"""Download the MLCR dataset from Hugging Face and optionally reconstruct harness structure."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ID = "Wisedocs/mlcr-dataset"


def download_dataset(output_dir: Path, prepare_for_harness: bool = False) -> None:
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from {REPO_ID}...")
    questions_ds = load_dataset(REPO_ID, "questions", split="train")
    summaries_ds = load_dataset(REPO_ID, "cases_summaries", split="train")
    fillers_ds = load_dataset(REPO_ID, "filler_files", split="train")

    if not prepare_for_harness:
        questions_ds.to_parquet(output_dir / "questions.parquet")
        summaries_ds.to_parquet(output_dir / "cases_summaries.parquet")
        fillers_ds.to_parquet(output_dir / "filler_files.parquet")
        print(f"Saved 3 parquet files to {output_dir}")
        return

    print("Reconstructing harness directory structure...")
    _write_cases(output_dir, questions_ds, summaries_ds)
    _write_fillers(output_dir, fillers_ds)
    print(f"Done. Harness structure written to {output_dir}")


def _write_cases(output_dir: Path, questions_ds, summaries_ds) -> None:
    cases_dir = output_dir / "cases"

    # Group questions by case_uuid
    questions_by_case: dict[str, list[dict]] = {}
    for row in questions_ds:
        questions_by_case.setdefault(row["case_uuid"], []).append(row)

    for case_uuid, rows in questions_by_case.items():
        case_dir = cases_dir / case_uuid
        prompts_dir = case_dir / "prompts"
        answers_dir = case_dir / "answers"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        answers_dir.mkdir(parents=True, exist_ok=True)

        difficulty_rows = []
        for row in rows:
            prompt_id = row["prompt_id"]  # e.g. "q01"
            num = prompt_id[1:]  # e.g. "01"

            (prompts_dir / f"{prompt_id}.txt").write_text(row["question"], encoding="utf-8")
            (answers_dir / f"a{num}.txt").write_text(row["answer"], encoding="utf-8")

            if row["difficulty"]:
                difficulty_rows.append({"prompt": prompt_id, "difficulty": row["difficulty"]})

        if difficulty_rows:
            diff_path = case_dir / "difficulties.csv"
            with open(diff_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["prompt", "difficulty"])
                writer.writeheader()
                writer.writerows(difficulty_rows)

        print(f"  case {case_uuid}: {len(rows)} prompts/answers")

    # Group summaries by case_uuid
    summaries_by_case: dict[str, list[dict]] = {}
    for row in summaries_ds:
        summaries_by_case.setdefault(row["case_uuid"], []).append(row)

    for case_uuid, rows in summaries_by_case.items():
        summaries_dir = cases_dir / case_uuid / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            (summaries_dir / row["summary_file"]).write_text(
                row["summary_text"], encoding="utf-8"
            )

        print(f"  case {case_uuid}: {len(rows)} summaries")


def _write_fillers(output_dir: Path, fillers_ds) -> None:
    fillers_by_subdir: dict[str, list[dict]] = {}
    for row in fillers_ds:
        fillers_by_subdir.setdefault(row["subdir_name"], []).append(row)

    for subdir_name, rows in fillers_by_subdir.items():
        ocr_dir = output_dir / "filler_files" / subdir_name / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            (ocr_dir / row["file_name"]).write_text(row["ocr_text"], encoding="utf-8")

        print(f"  filler {subdir_name}: {len(rows)} OCR files")
