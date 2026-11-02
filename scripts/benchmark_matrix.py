"""Run several benchmark_worker.py configurations interleaved, then compare them.

Each --config is LABEL=REQUEST or LABEL=REQUEST@ENGINE_SRC. ENGINE_SRC is the
`src` directory of another engine checkout (benchmarked through PYTHONPATH, as
docs/Benchmarking.md describes); without it the importable `beat_engine` runs.
Round r runs every configuration once, in the order given, each in a fresh
worker, so slow drift of the machine hits all configurations alike. Runs are
written to OUT_DIR/<label>_r<r>.json and then passed to compare_benchmarks.py
in the order given (the first configuration is the baseline).

  python scripts/benchmark_matrix.py --backend cpu --runs 3 --out-dir runs/porous \\
      --config "Air=sawmod_air.json" --config "JCA=sawmod_jca.json" \\
      --config "JCAL=sawmod_jcal.json" --config "JCA limp=sawmod_limp.json"
  python scripts/benchmark_matrix.py --backend cuda --runs 3 --out-dir runs/regression \\
      --config "main=sawmod_air.json@C:/src/BEAT_main/src" --config "branch=sawmod_air.json"

Works the same on Windows, Linux and macOS (the CUDA user step runs it with
--backend cuda). Interior-only requests always solve on CPU, whatever the backend.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def parse_config(spec: str):
    label, _, target = spec.partition("=")
    if not label or not target:
        raise SystemExit(f"--config {spec!r}: expected LABEL=REQUEST[@ENGINE_SRC]")
    request, _, source = target.partition("@")
    # A Windows drive letter ("C:/...") never follows the @, so a plain partition is safe.
    return label, Path(request).resolve(), Path(source).resolve() if source else None


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", required=True, help="LABEL=REQUEST[@ENGINE_SRC]")
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda", "rocm", "metal"))
    parser.add_argument("--precision", default="float32", help="float32 (coupled default) or float64")
    parser.add_argument("--runs", type=int, default=3, help="interleaved rounds (default: %(default)s)")
    parser.add_argument("--threads", default="4", help="Julia threads (default: %(default)s)")
    parser.add_argument("--frequencies", type=float, nargs="+", help="default: benchmark_worker.py's 12")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="run JSON for accuracy (compare_benchmarks.py)")
    parser.add_argument("--variance", type=float, help="run-to-run half-range in %% (compare_benchmarks.py)")
    args = parser.parse_args()

    configs = [parse_config(spec) for spec in args.config]
    labels = [label for label, _, _ in configs]
    if len({slug(label) for label in labels}) != len(labels):
        raise SystemExit("configuration labels must differ (after replacing punctuation)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for round_index in range(1, args.runs + 1):
        for label, request, source in configs:
            out = (args.out_dir / f"{slug(label)}_r{round_index}.json").resolve()  # the worker runs elsewhere
            command = [sys.executable, str(SCRIPTS / "benchmark_worker.py"), "--request", str(request)]
            command += ["--backend", args.backend, "--precision", args.precision, "--threads", args.threads]
            command += ["--label", label, "--out", str(out)]
            if args.frequencies:
                command += ["--frequencies", *map(str, args.frequencies)]
            env = dict(os.environ)
            if source is not None:
                env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(source), env.get("PYTHONPATH")]))
            print(f"[round {round_index}/{args.runs}] {label}", flush=True)
            subprocess.run(command, env=env, check=True)

    compare = [sys.executable, str(SCRIPTS / "compare_benchmarks.py")]
    for label in labels:
        compare += ["--stage", f"{label}={args.out_dir / (slug(label) + '_r*.json')}"]
    if args.reference:
        compare += ["--reference", str(args.reference)]
    if args.variance is not None:
        compare += ["--variance", str(args.variance)]
    subprocess.run(compare, check=True)


if __name__ == "__main__":
    main()
