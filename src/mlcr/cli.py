from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from mlcr.config import load_experiment

# Load .env from cwd (and any parent) before reading any provider env vars.
load_dotenv()
from mlcr.matrix import build_matrix
from mlcr.runner import Runner


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mlcr")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run an experiment")
    pr.add_argument("config", type=Path)
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--repo-root", type=Path, default=None)
    pr.add_argument("--resume", metavar="EXPERIMENT_UUID", default=None,
                    help="Resume an existing run: reuse its directory and skip rows "
                         "already completed (status=ok). Use --force to redo all rows.")
    pr.add_argument("--no-previous-runs", action="store_true",
                    help="Disable reuse of results from previous experiments. By default "
                         "the runner looks at sibling run directories (most recent first) "
                         "for an existing successful result with the same identity "
                         "(case+prompt+model+modality+ablation+thinking) and reuses it "
                         "instead of calling the model.")
    pp = sub.add_parser("plan", help="Print the matrix without running")
    pp.add_argument("config", type=Path)
    pp.add_argument("--repo-root", type=Path, default=None)

    ps = sub.add_parser("summarize", help="Rebuild summary.csv from summary.jsonl")
    ps.add_argument("run_dir", type=Path)

    pd = sub.add_parser("download", help="Download dataset from Hugging Face")
    pd.add_argument("--output-dir", type=Path, default=Path("."),
                    help="Directory to write downloaded data (default: current dir)")
    pd.add_argument("--prepare-for-harness", action="store_true",
                    help="Reconstruct cases/ and filler_files/ directory structure "
                         "compatible with the experiment harness")

    from mlcr.scoring import build_arg_parser as _build_score_parser

    psc = sub.add_parser(
        "score",
        help="Score responses with LLM judge (conciseness gate + 3-model majority vote "
        "for completeness + accuracy) and write results_scoring.csv",
    )
    _build_score_parser(psc)

    args = p.parse_args(argv)

    if args.cmd == "run":
        if args.resume:
            # When resuming, use the saved config from the run directory
            repo_root = args.repo_root or Path.cwd()
            cfg_from_cli = load_experiment(args.config, args.repo_root)
            run_dir = (cfg_from_cli.repo_root or repo_root) / cfg_from_cli.output_dir / args.resume
            saved_config = run_dir / "config.yaml"
            if saved_config.is_file():
                cfg = load_experiment(saved_config, args.repo_root)
                config_path = saved_config
            else:
                cfg = cfg_from_cli
                config_path = args.config
            if not run_dir.is_dir():
                print(f"cannot resume: run directory does not exist: {run_dir}", file=sys.stderr)
                return 1
        else:
            cfg = load_experiment(args.config, args.repo_root)
            config_path = args.config
        result = Runner(
            cfg,
            force=args.force,
            dry_run=args.dry_run,
            limit=args.limit,
            confirm=True,
            experiment_uuid=args.resume,
            use_previous_runs=not args.no_previous_runs,
            config_path=config_path,
        ).run()
        print(json.dumps(result, indent=2))
        if result.get("aborted"):
            return 1
        return 0

    if args.cmd == "plan":
        cfg = load_experiment(args.config, args.repo_root)
        rows = build_matrix(cfg)
        print(f"matrix size: {len(rows)}")
        for r in rows:
            print(f"  {r.row_id}  case={r.case_uuid}  case_specific_prompt={r.prompt_path.name}  "
                  f"model={r.model_config_id}  modality={r.modality}  ratio={r.ablation_ratio:g}  "
                  f"thinking={r.thinking}")
        return 0

    if args.cmd == "summarize":
        from mlcr.runner import Runner as _R
        # Minimal helper: load run.json and rerun csv build.
        run_dir = args.run_dir
        rj = json.loads((run_dir / "run.json").read_text())
        from mlcr.config import ExperimentConfig
        cfg = ExperimentConfig.model_validate(rj["config"])
        r = _R(cfg, experiment_uuid=rj["experiment_uuid"], run_dir=run_dir)
        r._rebuild_csv()
        r._write_responses_csv()
        print(f"wrote {r.summary_csv} and {run_dir / 'responses.csv'}")
        return 0

    if args.cmd == "download":
        from mlcr.download import download_dataset

        download_dataset(args.output_dir, prepare_for_harness=args.prepare_for_harness)
        return 0

    if args.cmd == "score":
        from mlcr.scoring import score_run

        judge_model_configs = None
        if args.judge_models:
            repo_root_path = Path(__file__).resolve().parents[2]
            models_dir = repo_root_path / "configs" / "models"
            judge_model_configs = [models_dir / f"{mid}.yaml" for mid in args.judge_models]

        result = score_run(
            args.run_dir,
            judge_model_configs=judge_model_configs,
            max_workers=args.max_workers,
            limit=args.limit,
            checkpoint_every=args.checkpoint_every,
        )
        print(json.dumps(result, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
