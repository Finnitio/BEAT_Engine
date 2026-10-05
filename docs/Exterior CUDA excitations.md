# Exterior CUDA excitations

Exterior-only CUDA solves (`coupled_solver.jl`, `solve_exterior_direct_cuda`)
factor one direct Burton-Miller system per frequency and solve it for every
excitation port. Until this change only the system matrix was shared: excitation
1's right-hand side rode along the system assembly, and every further excitation
called `assemble_burton_miller_rhs_cuda`, which repeats the regular,
symmetry-image and singular quadrature over every element pair. Assembly time
therefore grew with the excitation count. A 60-drive loudspeaker model paid for
60 full pair integrations per frequency, where Metal's fused assembler pays for
one.

## Cached right-hand-side mapping

The right-hand side is linear in the Neumann data:

$$
b = \left(-S - \eta\left(D^{*} + \tfrac12 I_{P1,DP0}\right)\right) q = B q .
$$

`assemble_burton_miller_neumann_system_cuda(...; assemble_operator=true)` stores
the right-hand side as a P1 x DP0 matrix whose column j receives only trial
element j's contributions, as `assemble_burton_miller_rhs_cuda(...;
assemble_operator=true)` already does for Deploy's feedback operator. With unit
Neumann data that matrix is B, formed in the same regular, image, singular and
identity launches as the system matrix and scaled by the same symmetry row
weights. `assemble_burton_miller_neumann_system_columns_cuda` then forms every
excitation's right-hand side as `B * Q` with one cuBLAS GEMM and releases B
before the factorization. One pair integration serves any number of excitations.

The matrix and B are allocated as complex arrays that own their storage; the
kernels write through a derived Float32 view that is released before returning.
Freeing the returned matrix or mapping therefore releases the memory at once.
Previously the matrix was a complex view derived from Float32 storage, which
kept roughly `8 * p1^2` bytes alive until the garbage collector ran.

## Choosing the path

`exterior_rhs_policy.jl` decides per frequency:

- `auto` (default): the cached mapping for two or more excitations, when the
  system matrix, B and 512 MiB fit in the memory CUDA can hand out (free device
  memory plus what CUDA.jl's pool holds unused). One excitation keeps the
  original path. The choice is a pass count, not a machine constant: from two
  excitations on, the mapping replaces at least one full pair integration with a
  GEMM that is negligible beside it.
- `matrix_free`: the original per-excitation integration, for comparisons.
- `cached_operator`: the mapping whenever it fits, even for one excitation.

Set `BEAT_EXTERIOR_RHS_MODE` in the worker environment to force a mode. If B
cannot be allocated despite the check, the assembly falls back to `matrix_free`
with a warning. Result diagnostics report `exterior_rhs_requested_mode`,
`exterior_rhs_mode` (what ran) and `exterior_rhs_operator_bytes`.

B costs `8 * p1 * dp0` bytes in Float32: 8.06 GB for a 44,662-face half model
(22,549 P1 dofs), on top of the 4.07 GB system matrix. Smaller GPUs, or meshes
where both do not fit, keep the original path.

## Validation

The phasor tests compare the mapping-mode system with the one-excitation direct
system, the mapping with the independent CPU operator matrices, and both column
modes with the CPU right-hand side for one and three excitations, in both time
conventions. The CUDA production tests do the same on the `xy` symmetry mesh,
covering image and image-singular corrections and row weights, and require the
two modes to agree to 2e-5. The policy test covers the thresholds and memory
decisions without a GPU. Run:

```text
julia --startup-file=no src/beat_engine/julia_local/tests/exterior_rhs_policy_tests.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_cuda src/beat_engine/julia_local/tests/phasor_standalone.jl
BLAB_RUN_COUPLED_CUDA=1 julia --threads=2 --startup-file=no --project=src/beat_engine/julia_cuda src/beat_engine/julia_local/tests/runtests.jl
```

## Qualification: loudspeaker model, 2026-09-30

Tesla V100-SXM2-32GB (Yandex Cloud `gpu-standard-v2`, driver 535.247, CUDA.jl 6.2
runtime 12.9), Julia 1.12.7, 4 Julia threads, 8 BLAS threads. Exterior-only
compiled request captured from a Boundary Lab project: a loudspeaker cabinet
half model with `x` symmetry, 44,662 faces and 22,549 P1 dofs, 60 excitation
ports (the cone split into moving segments), quadrature order 4 (6-point rule),
singular order 4, Float32, one frequency at 20 kHz, 290 observation points.
`scripts/benchmark_worker.py`, fresh worker per run, sweep 1 compared with
sweep 1.

| Engine | Runs | Assembly, s | Solve, s | Field, s | Wall, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `main` (e6b3037) | 1 | 199.69 | 2.31 | 0.23 | 203.5 |
| cached mapping | 3 | 5.80 (5.80-5.84) | 2.27 (2.27-2.32) | 0.23 | 9.5 (9.5-9.6) |

Median (range). Assembly is 34x faster and the frequency 21x; the solve and
field sections do not change. Peak GPU memory was 12,170 MiB with the mapping
(system matrix 4.07 GB + mapping 8.06 GB); peak worker RSS 2.23-2.26 GB in all
runs.

Against `main` on the same request, every run agreed to relative L2 2.8e-7 over
the 60 x 290 complex pressures (worst single excitation 7.9e-6), worst level
error 0.00003 dB within 30 dB of the peak, and 1.4e-7 on the 60 radiation
impedances: the float32 atomic-order differences expected between two runs of
either path. The full `runtests.jl` with `BLAB_RUN_COUPLED_CUDA=1` and the CPU
reference gate passed on this machine.

Against a Float64 CPU reference (`main`, `bem_backend = cpu`,
`precision = float64`, fixed order-4 quadrature as in the CUDA runs; assembly
1,331 s, solve 74 s, peak RSS 67 GB), `main` and the cached mapping are equally
accurate. Both have relative L2 5.62e-5, worst excitation 1.52e-4 and worst
level error 0.0015 dB on the pressures, and 8.0-8.2e-7 on the impedances. That
difference is float32's floor for this model at 20 kHz, and the change does not
move it.

On a Tesla T4 (16 GB, `standard-v3-t4`) the same request assembles in 17.86 s
(3 runs, 17.62-17.87) with the mapping, and the T4's 12.2 GB peak fits its
memory. The per-excitation path of engine 0.2.0 took 522 s of assembly there
(the same request at 1 kHz).

This qualifies these GPUs and this scene, not every GPU. Smaller GPUs keep the
original path when the mapping does not fit.
