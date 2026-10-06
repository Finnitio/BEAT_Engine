# Coupled CUDA BEM assembly

Coupled CUDA solves with full-matrix diagnostics disabled default to direct
assembly of the combined Burton–Miller pressure operator A and flux operator C.
This applies to both the monolithic CUDA solve and the CUDA/cuDSS statically
condensed FEM-BEM-LEM solve. CPU and ROCm retain individual-operator assembly. Condensed Metal solves have
a separate environment-controlled combined A/C path with host sparse projection;
monolithic Metal solves retain individual operators. See
[Metal Backend](Metal%20Backend.md).
The exterior-only `burton_miller_assembly` option is independent.

## Formulation and data flow

Using the existing P1 pressure and DP0 normal-derivative spaces,

```text
A = 0.5 Mpp - D + alpha H
C = S + alpha (adjD + 0.5 Mpq)
q = Q q_interface + M v + q_known
A p + C Q q_interface + C M v = -C q_known
```

For physical k > 0, alpha is -i/k under `exp(+i omega t)` and +i/k under
`exp(-i omega t)`. CUDA kernels receive the corresponding signed outgoing
wavenumber. See [Phasor Convention](Phasor%20Convention.md). The optimization
does not conjugate responses or change interface orientation conventions.

Each regular triangle pair accumulates nine complex A entries and three complex
C entries inside quadrature, instead of the separate S, D, adjD and H blocks.
This halves the explicit real accumulator and scatter counts from 48 to 24.
Multiple symmetry images can be evaluated sequentially into that same accumulator
set, followed by one scatter. Reflected normal and curl signs remain independent.
Direct singular and image-singular integrations retain the existing compact
correction rules, then scatter combined corrections directly into A/C. No dense
correction planes are materialized. Row weights are applied before adding the
already symmetry-weighted identity matrices.

A and C use interleaved real/imaginary storage that can be reinterpreted as
complex arrays without another full-size allocation. Unlike exterior-only
prescribed-Neumann assembly, coupled assembly must retain C until it projects
the unknown interface flux and transducer motion and forms known-flux RHS values.

Q remains sparse on the GPU. A job-owned CSC cache holds column offsets, incident
DP0 faces, and orientation-weighted values. Each output entry of C Q has one GPU
thread owner that gathers the corresponding column's nonzero contributions;
projection needs no atomics or dense Q. Empty interface maps are supported.
Small motion and prescribed-flux products use ordinary complex matrix products.
C is freed after these blocks are built; A and the blocks are consumed by the
existing coupled matrix assembly and pivoted dense LU. All excitation columns
continue to share the frequency's factorization.

FEM symbolic-analysis reuse, numerical factorization, reconstruction and field
evaluation are unchanged. No lower precision or weaker pivoting is introduced.

## Solver options and diagnostics

These keys belong to the request's open `solver_options` dictionary:

| Option | Default | Meaning |
|---|---|---|
| `coupled_bem_assembly` | `"auto"` | `auto` selects combined CUDA without full diagnostics; `combined` explicitly requires CUDA; `operators` selects the original four-operator path. |
| `coupled_bem_image_fusion` | `true` | Fuse multiple symmetry images in combined assembly; `false` keeps separate image launches. |
| `coupled_bem_max_registers` | `0` | Fused-image kernel register cap. Zero leaves compilation unconstrained; explicit caps must be integers from 32 through 255. |

Full-matrix `validation_diagnostics=true` always selects the original operator
path, including when combined assembly was requested. That path retains the
operators needed for replay/residual diagnostics. The existing restriction
against full diagnostics with static condensation still applies. Explicit
combined assembly on CPU/ROCm and unknown assembly modes fail clearly.

Frequency diagnostics report the **effective** `coupled_bem_assembly`,
`coupled_bem_image_fusion`, and `coupled_bem_max_registers`. Fusion is false and
the cap is zero when there are fewer than two images or the operator path is
used. Requested values remain in execution provenance. Numerical source hashes
include `BeatEngineCudaCoupledBurtonMiller.jl`. Restart persistent workers after
updating engine source.

The sparse projection cache belongs to `prepare_coupled_cache` and is released by
`release_coupled_cache!`. It is reused across frequencies. A cache explicitly
prepared with `coupled_bem_assembly=:operators` also retains dense Q for baseline
comparisons. An ordinary sparse cache can be used for diagnostic/operator solves:
those calls create and free a temporary dense Q. Switching back to combined
assembly reuses the same sparse cache. Existing backend, symmetry and retained
FEM-node compatibility checks remain in force.

## Qualification and tuning

The 200-frequency Multi_region_SAWMOD experiment on an RTX 2080 Ti reduced warmed
wall time from approximately 100.2 s to 72.4 s with fused combined assembly and
sparse projection. A 160-register cap averaged 71.5 s, with a smaller gain than
the main architecture change. The default remains uncapped because register
pressure, occupancy and spills vary by GPU, precision and quadrature. On that
machine the uncapped fused image kernel used more registers but substantially
less atomic-instruction pressure than separate image launches. A lower cap is
not universally faster.

FP32 atomic ordering remains nondeterministic. Small assembly perturbations may
be amplified by cancellation, conditioning or weak responses. The same project
showed substantial pre-existing repeat variation in weak high-frequency pressure
channels in both paths. Same-discretization FP64 original/combined spot checks
agreed within 1.9e-10 relative complex error; this does not establish mesh
convergence or a universal FP32 accuracy bound.

`julia_local/tests/coupled_bem_cuda_tests.jl` tests matrix/projection agreement,
both phasor conventions, symmetry modes, register-cap and separate/fused paths,
complex source columns, coupled solves, cache reuse and diagnostic fallback.
It runs as part of the CUDA `runtests.jl` suite. The standalone CPU reference gate
remains mandatory for numerical changes. See [numerical test instructions](../src/beat_engine/julia_local/tests/README.md).
