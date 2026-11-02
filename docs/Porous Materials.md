# Porous materials

A bounded region's `loss_model` names its **material**. Air is the default; a
porous fill is modelled as an equivalent fluid, Johnson–Champoux–Allard (JCA) or
Johnson–Champoux–Allard–Lafarge (JCAL), with a rigid or a limp frame. Reference: J.-F. Allard and N. Atalla, *Propagation of Sound in Porous Media*,
2nd ed., Wiley 2009 (equation numbers below refer to it).

## Wire format

| `loss_model` | Material |
|---|---|
| `{}`, `{"bulk_loss_factor": η}` | Air (legacy form, unchanged) |
| `{"model": "air", "bulk_loss_factor": η}` | Air, explicit; `bulk_loss_factor` optional, 0 ≤ η ≤ 1 |
| `{"model": "jca", "frame": "rigid", …}` | JCA equivalent fluid with a rigid frame |
| `{"model": "jcal", "frame": "rigid", …}` | JCAL: JCA plus a measured static thermal permeability |
| `{"model": "jca" \| "jcal", "frame": "limp", …}` | Either model with a limp frame (adds the frame density) |

JCA requires all five parameters, in SI; JCAL and the limp frame add one each:

| Key | Symbol | Admissible |
|---|---|---|
| `porosity` | φ | 0 < φ ≤ 1 |
| `flow_resistivity_pa_s_per_m2` | σ | σ > 0 |
| `tortuosity` | α∞ | α∞ ≥ 1 |
| `viscous_characteristic_length_m` | Λ | Λ > 0 |
| `thermal_characteristic_length_m` | Λ′ | Λ′ > 0 |
| `static_thermal_permeability_m2` (`jcal` only) | k0′ | k0′ > 0 |
| `frame_density_kg_per_m3` (`limp` only) | ρ1 | ρ1 > 0 |

Λ′ < Λ is physically unusual but not an engine error; warning about it is the
client's job. A key that belongs to another model or frame (k0′ on `jca`, ρ1 on
`rigid`) is an unknown key.

Unknown models, frames and keys are rejected (they were silently ignored before).
`bulk_loss_factor` exists only for air. Porous materials are allowed only on
`bounded_air` regions. Boundaries of a porous region may be `rigid`, `moving`,
`interface` or `unused`; thermoviscous wall losses, wall impedance (Miki), plane-wave
tube terminations and `impedance` boundaries are rejected. The Python and Julia
contract validators share these cases in `beat_contract/conformance.json`.

Workers advertise `fem_region_material_models: ["air", "jca", "jcal"]` and
`fem_region_material_frames: ["rigid", "limp"]`. The Python engine client refuses a porous
request before submission unless the worker advertises its model and frame, so an
older engine (which ignored the keys) never solves a porous fill as air. Air
requests need no capability. A backend entry in the ready event may narrow the list:
ROCm and Metal announce `fem_region_material_models: ["air"]`, so the client refuses
a coupled porous request on them before submission, naming the backend; CPU and CUDA
omit the key and use the worker list. Interior solves always run on CPU.

## Conventions

**Equivalent fluid.** A porous region is a fluid with the complex, frequency-dependent
*equivalent* density and bulk modulus ρ_eq = ρ/φ and K_eq = K/φ (Eqs. 5.43, 5.44), where
ρ and K are the effective density and bulk modulus of the air in the pores (Eqs. 5.50,
5.51). The unknown is the pore pressure p, the velocity is the macroscopic (volume-averaged)
velocity v = φ v_pore, and jω ρ_eq v = −∇p, ∇·v = −jω p / K_eq. Pressure and normal volume
velocity are then continuous at the material's surface, so the layer can be replaced by this
homogeneous fluid without changing the field outside it (Sec. 5.7; Z_c = √(ρ_eq K_eq) is the
book's Z_c/φ, Eq. 5.42, and k_c = ω √(ρ_eq/K_eq), Eq. 5.41). This is the form the FEM
weak form needs: no porosity factor appears at a region seam or an FEM–BEM interface.

**Phasor.** The book writes fields as e^{+jωt} (so Im ρ_eq < 0 for a passive material).
The engine evaluates every formula with jω = `time_derivative(ω)` of the request's
`solver_options.phasor_convention` (docs/Phasor Convention.md): with `exp(+i omega t)`,
which Boundary Lab sends, every complex quantity equals the book's with j = i; with the
legacy default `exp(-i omega t)` every quantity, ρ_eq and K_eq in the diagnostics
included, is its complex conjugate. The tests check both conventions.

## Equivalent fluid

One JCAL evaluation path computes, with jω the time derivative of the active phasor
convention:

```
rho(w) = rho0 [alpha + nu phi / (j w q0) G(w)],   q0 = eta / sigma,  nu = eta / rho0        (5.50, 5.52)
G(w)   = sqrt(1 + (2 alpha q0 / (phi Lambda))^2 j w / nu)
K(w)   = gamma P0 / (gamma - (gamma - 1) [1 + nu' phi / (j w k0') G'(w)]^-1),  nu' = nu / Pr  (5.51, 5.53)
G'(w)  = sqrt(1 + (2 k0' / (phi Lambda'))^2 j w / nu')
rho_eq = rho / phi,   K_eq = K / phi                                                       (5.43, 5.44)
```

JCAL takes k0′ from the request; JCA supplies k0′ = φΛ′²/8 (Eqs. 5.54/5.55), which
reduces the simplified Lafarge bulk modulus to the Champoux–Allard form, so JCAL
with that k0′ is JCA. η, Pr, γ
are the thermoviscous air constants (`THERMOVISCOUS_AIR`: η = 1.84e-5 Pa·s,
Pr = η c_p / κ ≈ 0.72, γ = 1.4), so wall-loss and porous models use one air.
ρ0 and c0 come from the region. The static pressure is derived as P0 = ρ0 c0² / γ,
so the adiabatic high-frequency limit γP0 equals the region's air bulk modulus
ρ0c0² and a fully open, resistance-free material is exactly the region's air.

A limp frame (light wadding whose fibres move with the air) replaces the rigid-frame
density by Panneton's approximation (Eq. 11.55); the bulk modulus is unchanged:

```
rho_limp = (rho_t rho_eq - rho0^2) / (rho_t + rho_eq - 2 rho0),   rho_t = rho1 + phi rho0     (11.55)
```

It tends to the total apparent density ρ_t as ω → 0 (the fibres move with the air)
and to the rigid ρ_eq as ρ1 → ∞. The frame decouples above roughly
F_d = σφ²/(2πρ1).

## Assembly

The air operator is the Helmholtz form multiplied by ρ0, `K − k² M`, with Neumann
data `∂p/∂n = neumann_scale(ρ0, ω) v`. A porous region r is assembled in the same
normalization:

```
A(w) = sum_r a_r(w) K_r + b_r(w) M_r
air:    a_r = 1,              b_r = -k^2 (1 + i s eta_r)      (s = propagation_sign())
porous: a_r = rho0 / rho_eq,  b_r = -w^2 rho0 / K_eq        (rho_eq = rho_limp for a limp frame)
```

The natural boundary term of a porous region is then ρ0 (1/ρ_eq) ∂p/∂n =
`neumann_scale(ρ0, ω) v`: moving boundaries and transducer couplings keep their
existing terms, and this is exactly the "ρ_eq instead of ρ0" velocity relation of
the physical equivalent fluid. Pressure and volume flux (1/ρ_eq) ∂p/∂n are
continuous across nodes shared by two regions through the weak form alone.

The same holds at an FEM–BEM interface. The coupled solve's interface unknown is
the FEM-side flux ρ0 (1/ρ_eq) ∂p/∂n, which equals the exterior air's ∂p/∂n when the
normal velocity is continuous, so the interface flux blocks, BEM coupling and
`interface_average_normal_velocity` (flux / `neumann_scale(ρ0, ω)`) are unchanged:
the porous region enters the coupling only through ρ_eq in its volume operator. The
coupled caches (`prepare_coupled_cache`, used by the monolithic and condensed
builders) hold the per-region operator on CPU and CUDA when the request has porous
regions or region seams, and `coupled_fem_dynamic_stiffness` forms it per frequency. The Float64 FEM reassembly
for Float32 runs (`BLAB_COUPLED_FEM_FLOAT64`) reassembles each region's `K_r`/`M_r`
in Float64 on the same pattern. Coupled FEM mass is always consistent.

`K_r`, `M_r` are assembled once per request over each region's tetrahedra (P1 or
P2, with the requested consistent/lumped mass blending) and stored as value arrays
on the pattern of the global air operator. Each frequency only rewrites `nzval`, so
`colptr`/`rowval` equal those of the air-only operator. Requests without porous
regions take the original operator path and give bit-identical results.

Diagnostics of porous requests list `fem_region_materials`: per region its
normalized material and, for porous regions, ρ_eq and K_eq at that frequency as
`[real, imag]`. Every result, air-only included, also records each bounded region's
normalized material in `engine_provenance.region_materials`
([BEAT Run Provenance](BEAT%20Run%20Provenance.md)); air-only numerical outputs are unchanged.

## Region seams

Two bounded regions on the same FEM mesh (same `mesh_ids`, different `volume_groups`)
meet at the tetrahedron faces both selections have on their exterior. Each shared face is
classified by its boundary assignments; a side counts as assigned when a mesh triangle on
the face carries a tag that the region assigns to one of its boundaries:

| Region A | Region B | Result |
|---|---|---|
| unassigned | unassigned | **Region seam**: an interior face. The regions share its nodes (corners, and edge nodes for P2), so pressure and volume flux are continuous through the weak form alone, with no boundary or interface object. A triangle on the face whose tag neither region assigns is dropped. |
| assigned | assigned | Thin rigid partition: each region keeps its own (duplicated) nodes and boundary, exactly as before seams existed. |
| assigned | unassigned | Error naming both regions and the assigned one: assign the face in both regions or in neither. |

Faces of regions on different meshes are never seams. A node on a seam face is shared by
the regions joined through seam faces at that node (a union over region pairs, so a chain of
seams shares one node); a region that touches the node only across a thin rigid partition
keeps its own copy, for example where a partition's rim meets a seam between two other
regions.

**Air-only requests with seams** take the per-region path below, not the original air
operator: a seam needs per-element bulk loss. Their results equal the same domain meshed as
one region to rounding (rtol 1e-11 in the tests), not bit for bit. Air-only requests without
seams keep the v0.7.0 path and are bit-identical.

**Bulk loss at seam nodes.** Interior solves and CPU and CUDA coupled solves of a request
with seams use the region operator above, so each region's `bulk_loss_factor` acts on its
own elements (b_r M_r): a seam node gets each side's mass contribution with that side's
factor, and regions with different factors (air|air or air|porous) are well defined. The
ROCm and Metal coupled caches still scale mass rows per node and keep the air operator; there
a seam between regions with different bulk loss factors is rejected before the solve
("Region seam between ... joins different bulk loss factors"), and a seam between regions
with equal factors (for example air|air with the same `bulk_loss_factor`) is exact, since
the per-node factor is then single-valued. Porous regions are refused on these backends
altogether. The aggregate's per-node `bulk_loss_factor_by_vertex` holds the first
region's factor at seam nodes.

**Output layout.** `fem_nodal_pressure` keeps one row per region-local vertex, regions
contiguous in request order (`region_ids`, `node_offsets`, `node_counts`), region-local order
the sorted mesh vertex index. A seam node therefore appears once in each adjoining region's
block with one value; consumers need no change. Internally `domains[r].vertex_indices` maps
region-local vertices to the merged FEM nodes and `domains[r].face_indices` region-local
boundary triangles to the aggregated ones (0 for a dropped seam triangle).

**Diagnostics.** Results of requests with seams carry `fem_region_seams`: per adjoining region
pair `region_ids`, `mesh_id`, `face_count` and `node_count`. Requests without seams are
unchanged (no key, bit-identical results).

## Mesh resolution

A porous region needs a finer mesh than air: its wavelength is shorter and the wave
is damped. The convergence study measures the input-impedance error of a filled tube
over **elements per porous wavelength** N = λ / h, where

- λ = 2π / Re(k_eq), k_eq = ω √(ρ_eq / K_eq), at the maximum solve frequency (at 5 kHz:
  soft foam 57.5 mm, glass wool 52.6 mm, air 68.6 mm);
- h is the mean tetrahedron edge length, which an isotropic mesh of target size h has
  on average.

Setup (`julia_local/tests/porous_tube_helpers.jl`): the piston-driven, rigidly closed
tube of the validation below, built from cubes of side h / 1.26 split into six
tetrahedra (mean edge h), one cube across, so the elements are isotropic. Two rigid-frame
presets (spec preset table): soft foam (Table 11.5: φ 0.99, σ 10 900 Pa·s/m², α∞ 1.02,
Λ/Λ′ 100/130 µm) and glass wool (Table 11.8: φ 0.94, σ 40 000 Pa·s/m², α∞ 1.06,
Λ/Λ′ 56/110 µm). Error metric: |Z_FEM − Z| / |Z| against −j Z_c cot(k_c L), maximized
over f_max ∈ {0.5, 2, 5, 10} kHz, tube lengths of 1, 2 and 4 porous wavelengths at
f_max, and ten frequencies 0.1–1 × f_max (the mesh has N elements per wavelength at
f_max and more below it). Worst relative input-impedance error:

| Order | N | Soft foam | Glass wool |
|---|---|---|---|
| P1 | 4 | 2.3e-01 | 9.7e-02 |
| P1 | 6 | 1.4e-01 | 5.2e-02 |
| P1 | 8 | 7.2e-02 | 2.7e-02 |
| P1 | 9 | 5.4e-02 | 2.2e-02 |
| P1 | 10 | 5.2e-02 | 1.9e-02 |
| P1 | 11 | 4.2e-02 | 1.5e-02 |
| P1 | 12 | 3.4e-02 | 1.3e-02 |
| P1 | 14 | 2.7e-02 | 1.0e-02 |
| P1 | 16 | 2.0e-02 | 7.4e-03 |
| P1 | 20 | 1.3e-02 | 4.9e-03 |
| P1 | 24 | 9.1e-03 | 3.4e-03 |
| P2 | 2 | 5.7e-02 | 6.3e-02 |
| P2 | 3 | 1.8e-02 | 1.9e-02 |
| P2 | 4 | 7.4e-03 | 8.0e-03 |
| P2 | 5 | 3.2e-03 | 4.2e-03 |
| P2 | 6 | 1.8e-03 | 2.4e-03 |
| P2 | 8 | 7.8e-04 | 1.1e-03 |

Smallest N on that grid that meets a target error, both materials:

| Target error | P1 | P2 |
|---|---|---|
| 1 % | 24 | 4 |
| 2 % | 16 | 3 |
| **5 % (mesh-resolution threshold)** | **11** | **3** |

The P1 worst case is soft foam, the least damped (0.8 Np per wavelength at 10 kHz),
at 10 kHz in the shortest tube near its half-wave resonance; glass wool damps the
wave in about one wavelength and needs roughly two thirds of the soft-foam N. The
error is nearly independent of tube length beyond one wavelength, because the
damping limits how far the dispersion error accumulates. P1 converges at second order;
P2 at about third order (the plane wave crosses the tetrahedra obliquely, so the
fourth-order dispersion of 1D P2 is not reached), and needs about a quarter of the
P1 elements per wavelength.

**Threshold.** A porous region whose mesh has fewer than 11 (P1) or 3 (P2) elements
per porous wavelength at the maximum frequency is expected to exceed 5 % input-impedance
error (about 0.4 dB). The 5 % target is a warning level, not an accuracy goal: Boundary Lab's
default air FEM mesh (6 elements per air wavelength) is at about 10 % modal-frequency
error for P1; stricter targets are in the table. The needed edge length is λ / N. The thresholds are data in
`beat_contract/porous-mesh-resolution.json`; Python clients read them with
`beat_engine.beat_contract.porous_mesh_edge_length_m(wavelength_m, element_order)`
(Boundary Lab's mesh warning). The structured cubes are a best case for an
unstructured mesh of the same mean edge, whose edge lengths vary; a client may
compare a high percentile of the region's edges rather than the mean.

Reproduce: `julia --threads=2 --project=src/beat_engine/julia_local
src/beat_engine/julia_local/scripts/porous_convergence_study.jl` prints both tables
(about 2 minutes). The slow test `porous_convergence_tests.jl` (in `runtests.jl` only
with `BEAT_RUN_POROUS_CONVERGENCE=1`; run the file directly otherwise) checks that the
recorded thresholds meet the target for both materials and that one element fewer
per wavelength misses it.

## Current scope

- Interior FEM solves, CPU.
- Coupled FEM–BEM solves on the CPU backend (condensed and monolithic, Float64 and
  Float32), including porous regions that reach an interface to the exterior.
- Coupled FEM–BEM solves on CUDA (monolithic and cuDSS condensed), implemented but
  qualified only by the manual hardware check below; there is no CUDA run in CI.
- Coupled solves on ROCm and Metal refuse porous regions before mesh or device setup
  ("Porous region materials are not supported on the metal backend ..."), rather than
  solving the fill unverified.
- Region seams work in interior solves and in CPU and CUDA coupled FEM–BEM solves (any
  region materials; each region's bulk loss acts on its own elements). On ROCm and Metal
  seams need equal bulk loss on both sides.
- Deploy and speaker-ROM evaluation do not form the FEM operator (exterior/ROM only); the
  ROM export reads the condensed CPU system, which already carries the porous operator.

**Backends.** Every path forms the same operator on the air pattern. The host builds
`sum_r a_r K_r + b_r M_r` (`coupled_fem_dynamic_stiffness`), which the CPU monolithic and
condensed solves, the Float64 FEM reassembly and the CUDA cuDSS condensation consume. The
CUDA monolithic path scatters frequency-invariant device blocks with per-frequency
coefficients: for air-only requests without seams the original `K, M, D_eta M` blocks (code path
unchanged from v0.7.0, shared with ROCm), and only for requests with porous regions or region
seams the per-region `K_r, M_r` blocks (`coupled_fem_operator_blocks`,
`coupled_fem_operator_coefficients`); the CPU suite checks that the per-region scatter equals
the host operator. Because only values change with frequency, `colptr`/`rowval` are identical at every
frequency, which is what the cuDSS (and MUMPS) symbolic-analysis reuse compares
(`fem_symbolic_analysis_reused` in the diagnostics). The CPU suite checks only this
precondition (one pattern across frequencies, `porous_coupled_tests.jl`): the CPU condensed
solve factors with UMFPACK, which redoes symbolic and numeric factorization at every
frequency and reports no reuse, and MUMPS ships only in the Metal environment, which refuses
porous regions. The reuse itself with porous regions is covered only by the CUDA-gated
check (`BEAT_RUN_POROUS_CUDA=1`, manual).

**Manual CUDA check.** On a machine with a functional CUDA environment, run
`python scripts/compare_porous_backends.py`. It runs `porous_coupled_tests.jl` in the CUDA
Julia environment with `BEAT_RUN_POROUS_CUDA=1`: the radiating porous tube (200, 700,
1500 Hz) and the air|porous seam workload, monolithic and condensed, Float64 and Float32,
each against the CPU Float64 solve (tolerance 1e-8 for Float64, 2e-3 for Float32), and the
condensed sweeps must report `fem_symbolic_analysis_reused` from the second frequency on.

## Performance

Measured with the tools and procedure of [Benchmarking](Benchmarking.md): sweep 1 of a
fresh worker per run, runs of all configurations interleaved round by round
(`scripts/benchmark_matrix.py`), 12 log-spaced frequencies 20 Hz–20 kHz,
`exp(+i omega t)` as Boundary Lab sends it, coupled solves in Float32 (condensed, the
Boundary Lab default) and interior solves in Float64.

**Machine (CPU).** AMD Ryzen 9 5900X, Linux VM with 4 cores exposed (1 thread per core),
7.6 GiB RAM, Linux 7.0.14; Julia 1.12.7, `--threads=4` (the worker sets BLAS threads to the
Julia thread count), CPython 3.12. Memory is the worker's peak RSS over the sweep.

**Air-only regression** (branch against v0.7.0, `c82bb7a`, 3 interleaved runs each):

| Workload | Path | c82bb7a wall, median (range) | Branch wall | Change | Peak memory | Outputs |
|---|---|---|---|---|---|---|
| `Multi_region_SAWMOD` (4 regions, 3 drivers, xy symmetry) | coupled, condensed, Float32 | 306.7 s (305.4–307.1, ±0.3 %) | 307.4 s (305.7–310.3, ±0.7 %) | within variance (0.2 %) | 2252 / 2066 MB | bit-identical |
| `compression_driver` (2 regions, termination) | interior, Float64 | 0.20 s (0.17–0.21, ±9.6 %) | 0.21 s (0.21–0.22, ±3.1 %) | within variance | 527 / 524 MB | bit-identical |

Both are bit-identical to the base engine, so the accuracy against the base Float64 CPU
reference is the same for both (SAWMOD Float32: exterior pressure 1.3e-5 relative,
0.002 dB). The SAWMOD sections agree to within 1 % each (BEM operator 137.4 / 138.1 s,
FEM condensation 55.4 / 56.0 s, field 89.8 / 89.8 s); the run-to-run half-range is ±0.7 %.

**Porous overhead.** One project per path, the porous body as Air (the captured project,
unchanged), JCA, JCAL (k0′ = 2e-9 m²) and JCA on a limp frame (ρ1 = 30 kg/m³), soft
fibrous material of Table 11.2 (`scripts/porous_benchmark_requests.py`). "Per frequency" is
the mean over the 12 frequencies of the worker's `assembly_s + solve_s + field_s`; the
operator column is the mean per frequency of `fem_system_s + fem_matrix_cache_s +
cache_setup_s` (the per-region operator formation and its one-off cache):

| Workload | Material | Runs | Wall, median (range) | Per frequency | vs Air | FEM task / freq | Operator / freq | Peak memory | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| `Simple_Sealed`, rear 52 mm of the box (12 474 of 55 356 tets) a porous layer joined by a region seam (727 seam nodes); coupled through the driver | Air, one region (original operator) | 3 | 12.28 s (12.23–12.49) | 988 ms | — | 335 ms | 18 ms | 1103 MB | v0.7.0 air path (A) |
| | Air, two regions + seam | 3 | 36.13 s (35.97–36.30) | 971 ms | (reference) | 324 ms | 47 ms | 1061 MB | per-region path (B) |
| | JCA | 3 | 36.13 s (35.92–36.17) | 976 ms | +0.5 % | 327 ms | 44 ms | 1053 MB | (C) |
| | JCAL | 3 | 36.03 s (35.93–36.18) | 962 ms | −0.9 % | 332 ms | 42 ms | 1037 MB | (C) |
| | JCA limp | 3 | 36.09 s (36.08–36.38) | 973 ms | +0.2 % | 341 ms | 48 ms | 1045 MB | (C) |
| `Multi_region_SAWMOD`, sealed main chamber (11 020 tets) stuffed | Air | 1 | 306.2 s | 25.42 s | (reference) | 4.64 s | 60 ms | 2100 MB | bit-identical to v0.7.0 (A) |
| | JCA | 1 | 305.4 s | 25.34 s | −0.3 % | 4.66 s | 83 ms | 2136 MB | (C) |
| | JCAL | 1 | 305.9 s | 25.39 s | −0.1 % | 4.61 s | 74 ms | 2062 MB | (C) |
| | JCA limp | 1 | 305.1 s | 25.31 s | −0.4 % | 4.53 s | 78 ms | 2137 MB | (C) |
| `compression_driver`, rear chamber (1 780 tets) filled; interior | Air | 3 | 0.22 s (0.21–0.22) | 9.8 ms | (reference) | — | 0.6 ms | 524 MB | bit-identical to v0.7.0 (A) |
| | JCA | 3 | 0.22 s (0.22–0.22) | 10.3 ms | +5 % (ranges overlap) | — | 1.1 ms | 526 MB | (C) |
| | JCAL | 3 | 0.22 s (0.21–0.23) | 9.8 ms | −0.3 % | — | 1.1 ms | 522 MB | (C) |
| | JCA limp | 3 | 0.22 s (0.21–0.22) | 9.8 ms | +0.4 % | — | 1.1 ms | 521 MB | (C) |

**Accuracy.** (A) The SAWMOD and `compression_driver` Air requests are the regression requests
above, bit-identical to v0.7.0 (SAWMOD Float32 against Float64 CPU: exterior pressure 1.3e-5
relative, 0.002 dB); the one-region `Simple_Sealed` Air request takes the same unchanged air
path, whose bit-identity `air_only_pinned_tests.jl` pins, but was not compared on this mesh.
(B) The two-region air request takes the per-region path, equal to one region to rounding
(region-seam tests, rtol 1e-11); not compared on this mesh. (C) Porous outputs of these
projects have no analytical reference; the operator they run is validated by the analytical
tests under Validation: filled tube input impedance 1.7e-4 (JCA), 1.8e-4 (JCAL), 1.7e-4
(limp) relative at 80 slabs; radiating filled tube through the FEM–BEM interface 0.25 %
inlet impedance; air|porous seam against the transfer matrix 1.5e-3.

Per frequency, every porous variant is within run-to-run variance of the same mesh with
Air, so the 10 % target holds without optimization. Peak memory is within 2 % of the
same-mesh Air run, inside the run-to-run spread (SAWMOD air-only runs: 2039–2269 MB). SAWMOD was run
once per configuration, as Benchmarking allows once the variance of the workload is
established (±0.7 % above). The per-region operator
`sum_r a_r K_r + b_r M_r` costs 15–30 ms per frequency more than the original air
arithmetic on these meshes (operator column), against 0.3–4.6 s of FEM condensation;
storing `K_r`, `M_r` as full-length value arrays on the shared pattern costs
2 R nnz values, a few MB here.

The only large difference is outside the frequency loop, and it explains the 36 s against
12 s wall time of the `Simple_Sealed` rows while their per-frequency times agree: the split
`Simple_Sealed` requests carry the mesh inline (`mesh_data`, 3.7 MB), and the worker spends about 24 s per
request hashing that payload for provenance (`mesh_provenance` in
`BeatEngineMeshData.jl`: `sha256(JSON.json(mesh["mesh_data"]))`). The same one-region
request with the mesh inline instead of as a file pays the same 24 s, so it is a cost of
the in-memory mesh transport (unchanged since v0.7.0), not of seams or porous materials.
It is left for its own performance change.

**Reproduce (CPU or CUDA).** In a Boundary Lab environment that has this engine
installed (the capture and split scripts import `blab`, `meshio` and `numpy`), from the
Boundary Lab checkout, with `<engine>` the engine checkout:

```
python <engine>/scripts/capture_boundary_lab_request.py examples/Multi_region_SAWMOD/Multi_region_SAWMOD.blab.json --out req/sawmod.json
python <engine>/scripts/capture_boundary_lab_request.py examples/Simple_Sealed/simple_sealed.blab.json --out req/simple_sealed.json
python <engine>/scripts/capture_boundary_lab_request.py examples/compression_driver/compression_driver.blab.json --out req/compression_driver.json
python <engine>/scripts/porous_benchmark_requests.py req/sawmod.json --body region:sealedmainchamber --out-dir porous
python <engine>/scripts/porous_benchmark_requests.py req/simple_sealed.json --body region:interior-air-1 --split-below-m -0.2 --out-dir porous
python <engine>/scripts/porous_benchmark_requests.py req/compression_driver.json --body region:rear --out-dir porous

python <engine>/scripts/benchmark_matrix.py --backend cuda --runs 3 --out-dir runs/porous_ss --config "Air=porous/simple_sealed_air.json" --config "JCA=porous/simple_sealed_jca.json" --config "JCAL=porous/simple_sealed_jcal.json" --config "JCA limp=porous/simple_sealed_limp.json"
python <engine>/scripts/benchmark_matrix.py --backend cuda --runs 3 --out-dir runs/porous_sawmod --config "Air=porous/sawmod_air.json" --config "JCA=porous/sawmod_jca.json" --config "JCAL=porous/sawmod_jcal.json" --config "JCA limp=porous/sawmod_limp.json"
python <engine>/scripts/benchmark_matrix.py --backend cuda --runs 3 --out-dir runs/regression_sawmod --config "v0.7.0=porous/sawmod_air.json@<v0.7.0 checkout>/src" --config "branch=porous/sawmod_air.json"
```

`--backend cpu` reproduces the CPU table; `--threads` sets the Julia threads (default 4).
The same commands work in PowerShell and cmd. The interior `compression_driver` variants
always solve on CPU, so they need no CUDA run. Windows reports memory as unavailable. The
`--split-below-m` split writes a new mesh for the layer: on another mesh pick a coordinate
that leaves only rigid boundaries in the layer (the script stops otherwise).

### CUDA (manual user step, not measured here)

**Manual step for the user:** this VM has no GPU, so nothing below has been run on CUDA. Fill
the table in on a Windows CUDA machine with the commands above (`--backend cuda`), and run
`python scripts/compare_porous_backends.py` for the CUDA accuracy check. Until then CUDA
support for porous regions and seams is implemented but unmeasured. ROCm is not available to
the user either, so its air-only path (unchanged from v0.7.0) is not re-measured.

| Workload | Material | Runs | Wall, median (range) | Per frequency | vs Air | Peak memory |
|---|---|---|---|---|---|---|
| `Simple_Sealed` + porous layer | Air / JCA / JCAL / JCA limp | | | | | |
| `Multi_region_SAWMOD`, stuffed chamber | Air / JCA / JCAL / JCA limp | | | | | |
| `Multi_region_SAWMOD` regression | v0.7.0 / branch | | | | | |

Machine: GPU, CPU, cores, RAM, driver/CUDA version, Julia version, threads.

## Validation

- `porous_material_tests.jl`: high-frequency limit ρ_eq → α∞ρ0/φ with the leading
  √2 δ_v/Λ asymptotic, K_eq → γP0/φ; low-frequency Im ρ_eq → −σ/ω (e^{+jωt}) and
  K_eq → P0/φ; passivity and phasor conjugacy; open material equals air; JCAL with
  k0′ = φΛ′²/8 equals JCA and a different k0′ changes only K_eq; limp → ρ_t as
  ω → 0 and → rigid as ρ1 → ∞, K_eq unchanged.
- `porous_driver_tests.jl`: a compiled request of a 0.1 m tube completely filled
  with soft fibrous material (Table 11.2), piston at z = 0, rigid end. The FEM input
  impedance matches −j Z_c cot(k_c L) (Z_c = √(ρ_eq K_eq), k_c = ω√(ρ_eq/K_eq),
  written independently in book form as a test helper) at 100–2000 Hz in both phasor
  conventions. Measured relative error (P1):

  | Slabs (h) | 100 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz |
  |---|---|---|---|---|---|
  | 20 (5 mm) | 2.7e-4 | 6.1e-4 | 1.4e-3 | 2.8e-3 | 6.1e-3 |
  | 40 (2.5 mm) | 6.8e-5 | 1.5e-4 | 3.6e-4 | 6.9e-4 | 1.5e-3 |
  | 80 (1.25 mm) | 1.7e-5 | 3.9e-5 | 9.0e-5 | 1.7e-4 | 3.8e-4 |
  | 160 (0.625 mm) | 4.2e-6 | 9.7e-6 | 2.3e-5 | 4.4e-5 | 9.5e-5 |

  The test tolerance is 5e-4 at 80 slabs (about 80 elements per porous wavelength,
  99 mm at 2 kHz) plus a convergence ratio above 3 per halving (measured 4.0).
  The same tube filled with JCAL (k0′ = 2e-9 m², half of φΛ′²/8) and with JCA on a
  limp frame (ρ1 = 30 kg/m³), against the reference helper extended to JCAL and
  Eq. 11.55, meets the same tolerance (measured 1.8e-4 and 1.7e-4 at 80 slabs,
  ratio 4.0); the references differ from rigid JCA by 1.4–4.4 % (JCAL) and 2.9–36 %
  (limp) over the band, so the check discriminates. Further tests: an air tube anchors the piston sign, air and porous regions in one
  request match their separate solves, the porous operator keeps the air pattern,
  legacy and explicit air are bit-identical, ROCm and Metal coupled solves refuse
  porous regions naming the backend while CUDA passes the material checks, and P2 (ten-node) tetrahedra in the porous operator: the soft-foam
  tube at 6 elements per wavelength (mean edge, cubic cells) is 60 times more accurate
  than P1 and converges at about third order.
- `porous_coupled_tests.jl`: a 20 mm square tube, 50 mm long, completely filled with
  the same material, piston at z = 0 and its open end an FEM–BEM interface radiating
  into the exterior (the BEM surface is the tube's outside). The reference is
  independent of the porous coupling: the analytical JCA transfer matrix of the
  filled tube, loaded by the radiation impedance p/v of the same opening taken from
  an air-only coupled solve of the same mesh. At 200, 700 and 1500 Hz (CPU, condensed,
  Float64, 40 slabs) the measured relative errors are at most 0.25 % for the inlet
  impedance (tolerance 0.5 %), 0.86 % for the opening velocity (tolerance 1.5 %) and
  1.1 % for the exterior pressure ratio against the opening velocity ratio (tolerance
  2 %). Doubling the slabs leaves them unchanged: they come from the plane-wave model
  of the opening (the near field differs between air and porous fill), not from the
  discretization. An ideal open end (zero load) would move the opening velocity by
  10–15 times that error, so the test resolves the radiation load. Further tests:
  monolithic equals condensed (rtol 1e-9), Float32 with and without the Float64 FEM
  reassembly agrees (rtol 1e-4 / 1e-3), the negative-time result is the conjugate,
  and air-only coupled requests with legacy and explicit air are bit-identical.
  Air-only coupled results were also checked bit-identical against the previous
  engine (tiny workload: Float32/Float64, condensed/monolithic, Float64 FEM).
- `region_seam_tests.jl` (compiled requests, one mesh): an air tube (0.1 m) in front of a
  rigidly backed soft fibrous layer (0.05 m) as two regions with a region seam, piston at
  z = 0. Against the transfer-matrix solution (Allard & Atalla Ch. 11.3.1 and 11.5.1,
  written as a test helper), at 200, 800 and 2000 Hz in both phasor conventions: the
  surface impedance of the layer (fitted from the air field as an impedance tube does), and
  the cross-section mean pressure in the air and in the layer. Tolerance 2e-3 relative with
  2.5 mm P1 slabs (measured max 1.5e-3, layer pressure at 2 kHz; impedance 2.4e-4) and a
  convergence ratio above 3 per halving (measured 4.0). Nodal values within one cross-section
  of this one-element-wide tube spread by O(h) (slanted tetrahedra); the plane mean is the
  plane-wave quantity. Further tests: air|air seams (P1 and P2, with and without unassigned
  seam triangles) equal the single-region tube (rtol 1e-11); a lossless|lossy air seam matches
  the transfer matrix with the loss on the back side only; a face assigned on both sides
  gives the thin rigid partition with results equal to values pinned from engine 97a56c3;
  one-sided assignment errors with both region ids; a coupled FEM–BEM solve of the tiny
  workload split into two seam-joined regions equals the single-region solve (rtol 1e-9); on
  CPU differing bulk loss across a coupled seam is accepted (monolithic equals condensed,
  rtol 1e-9); on ROCm and Metal it is rejected, CUDA passes the seam check; three regions on
  one mesh, two joined by a seam whose rim lies on a thin rigid partition to the third
  (P1 and P2): the partitioned region keeps its own nodes (zero pressure without a source)
  and the seamed pair equals the single region (rtol 1e-11).
- `air_only_pinned_tests.jl`: SHA-256 of every output array of air-only requests (interior
  P1 in both phasor conventions and P2 tubes, the tiny coupled workload condensed and
  monolithic, Float64) equals the digest produced by v0.7.0 (`c82bb7a`) on the reference
  machine. The pinned bits depend on the platform's floating-point libraries.
- `porous_coupled_tests.jl` (backends): the CUDA per-region scatter blocks with their
  coefficients sum to the host operator (rtol 1e-13, air|porous cache; air-only caches keep
  the v0.7.0 blocks) on one pattern at
  200, 700 and 1500 Hz; per-region caches refuse ROCm and Metal by name; an air|porous
  seam in the tiny coupled workload gives monolithic equal to condensed (rtol 1e-9) and
  Float32 with the Float64 FEM reassembly within 1e-4; the backend comparison used by the
  manual CUDA check runs on CPU (Float64 within 1e-12, Float32 within 2e-3 of CPU Float64).
