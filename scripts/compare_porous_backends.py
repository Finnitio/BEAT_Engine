"""Manual CUDA check for porous region materials; unavailable hardware is a failure, never a pass.

Runs `julia_local/tests/porous_coupled_tests.jl` in the CUDA Julia environment with
`BEAT_RUN_POROUS_CUDA=1`: the porous filled tube and the air|porous seam workload are solved on
CUDA (monolithic and condensed, Float64 and Float32) and compared with the CPU Float64 solve;
condensed sweeps must reuse the cuDSS symbolic analysis after the first frequency. Each
comparison is logged as a `porous CUDA vs CPU Float64` info record.

    python scripts/compare_porous_backends.py [--log porous-cuda.log]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from beat_engine import engine_paths

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--log", type=Path, default=Path("porous-cuda.log"), help="Julia test output (default: %(default)s)"
)
parser.add_argument("--threads", default="2", help="Julia threads (default: %(default)s)")
args = parser.parse_args()

doctor = subprocess.run(
    [sys.executable, "-m", "beat_engine", "doctor", "--backend", "cuda"], capture_output=True, text=True
)
if doctor.returncode != 0:
    raise SystemExit(f"`beat_engine doctor --backend cuda` failed; is the CUDA environment set up?\n{doctor.stderr}")
info = json.loads(doctor.stdout)
if not info["backends"]["cuda"]["available"]:
    raise SystemExit(f"CUDA is unavailable: {info['backends']['cuda']}")
paths = engine_paths("cuda")
command = [
    "julia",
    f"--threads={args.threads}",
    "--startup-file=no",
    f"--project={paths.project}",
    str(paths.root / "julia_local/tests/porous_coupled_tests.jl"),
]
with args.log.open("w", encoding="utf-8") as log:
    process = subprocess.run(
        command, env=dict(os.environ, BEAT_RUN_POROUS_CUDA="1"), stdout=log, stderr=subprocess.STDOUT
    )
print(args.log.read_text(encoding="utf-8")[-6000:])
if process.returncode != 0:
    raise SystemExit(f"Porous CUDA check failed (exit {process.returncode}); full output in {args.log}")
print(f"Porous CUDA check passed; full output in {args.log}")
