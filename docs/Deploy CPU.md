# Deploy CPU coupled solves

The Deploy source worker supports `beat_engine_backend: cpu` for boundary and parity-ROM requests. CPU requests use `operator_matrices` assembly. CUDA ROM requests retain their direct-system path.

CPU coupled solves reuse `build_burton_miller_neumann_cpu_system` and `solve_burton_miller_neumann_cpu_system`: the exterior system is factored once, then reused to apply the ROM feedback operator during host-array GMRES. The ROM evaluation is shared with CUDA. The CPU GMRES implementation retains the CUDA path's scaled warm start, Arnoldi iteration, least-squares stopping criterion, iteration limit, and diagnostic tuple.

No worker/request/result schema change is required. CPU results retain boundary state for field-only requests and sweep frequency warm starts, and report `backend: cpu`. Existing CUDA numerical code is unchanged except for dispatch around the CPU branch.

Validation commands:

```powershell
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/deploy_cpu_tests.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl
```

Deploy's `scripts/check_solver_backends.py` provides end-to-end CPU/CUDA checks for boundary and coupled solves, field reuse, and two-frequency sweeps. Release this engine change before the downstream application adopts its wheel/hash pin. No numerical baselines were regenerated.
