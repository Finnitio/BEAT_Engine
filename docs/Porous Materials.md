# Porous materials: implementation notes

A bounded region's `loss_model` names its **material**. Air is the default; a
porous fill is modelled as a rigid-frame equivalent fluid (Johnson–Champoux–Allard,
JCA). Reference: J.-F. Allard and N. Atalla, *Propagation of Sound in Porous Media*,
2nd ed., Wiley 2009 (equation numbers below refer to it).

## Wire format

| `loss_model` | Material |
|---|---|
| `{}`, `{"bulk_loss_factor": η}` | Air (legacy form, unchanged) |
| `{"model": "air", "bulk_loss_factor": η}` | Air, explicit; `bulk_loss_factor` optional, 0 ≤ η ≤ 1 |
| `{"model": "jca", "frame": "rigid", …}` | JCA equivalent fluid with a rigid frame |

JCA requires all five parameters, in SI:

| Key | Symbol | Admissible |
|---|---|---|
| `porosity` | φ | 0 < φ ≤ 1 |
| `flow_resistivity_pa_s_per_m2` | σ | σ > 0 |
| `tortuosity` | α∞ | α∞ ≥ 1 |
| `viscous_characteristic_length_m` | Λ | Λ > 0 |
| `thermal_characteristic_length_m` | Λ′ | Λ′ > 0 |

Unknown models, frames and keys are rejected (they were silently ignored before).
`bulk_loss_factor` exists only for air. Porous materials are allowed only on
`bounded_air` regions. Boundaries of a porous region may be `rigid`, `moving`,
`interface` or `unused`; thermoviscous wall losses, wall impedance (Miki), plane-wave
tube terminations and `impedance` boundaries are rejected. The Python and Julia
contract validators share these cases in `beat_contract/conformance.json`.

Workers advertise `fem_region_material_models: ["air", "jca"]` and
`fem_region_material_frames: ["rigid"]`. The Python engine client refuses a porous
request before submission unless the worker advertises its model and frame, so an
older engine (which ignored the keys) never solves a porous fill as air. Air
requests need no capability.

## Equivalent fluid

One JCAL evaluation path computes, with jω the time derivative of the active phasor
convention (the book's e^{+jωt}; the legacy e^{−iωt} convention conjugates every
quantity):

```
rho(w) = rho0 [alpha + nu phi / (j w q0) G(w)],   q0 = eta / sigma,  nu = eta / rho0        (5.50, 5.52)
G(w)   = sqrt(1 + (2 alpha q0 / (phi Lambda))^2 j w / nu)
K(w)   = gamma P0 / (gamma - (gamma - 1) [1 + nu' phi / (j w k0') G'(w)]^-1),  nu' = nu / Pr  (5.51, 5.53)
G'(w)  = sqrt(1 + (2 k0' / (phi Lambda'))^2 j w / nu')
rho_eq = rho / phi,   K_eq = K / phi                                                       (5.43, 5.44)
```

JCA supplies the static thermal permeability k0′ = φΛ′²/8 (Eqs. 5.54/5.55), which
reduces the simplified Lafarge bulk modulus to the Champoux–Allard form. η, Pr, γ
are the thermoviscous air constants (`THERMOVISCOUS_AIR`: η = 1.84e-5 Pa·s,
Pr = η c_p / κ ≈ 0.72, γ = 1.4), so wall-loss and porous models use one air.
ρ0 and c0 come from the region. The static pressure is derived as P0 = ρ0 c0² / γ,
so the adiabatic high-frequency limit γP0 equals the region's air bulk modulus
ρ0c0² and a fully open, resistance-free material is exactly the region's air.

## Assembly

The air operator is the Helmholtz form multiplied by ρ0, `K − k² M`, with Neumann
data `∂p/∂n = neumann_scale(ρ0, ω) v`. A porous region r is assembled in the same
normalization:

```
A(w) = sum_r a_r(w) K_r + b_r(w) M_r
air:    a_r = 1,              b_r = -k^2 (1 + i s eta_r)      (s = propagation_sign())
porous: a_r = rho0 / rho_eq,  b_r = -w^2 rho0 / K_eq
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
CPU coupled caches (`prepare_coupled_cache`, used by the monolithic and condensed
builders) hold the per-region operator when the request has porous regions, and
`coupled_fem_dynamic_stiffness` forms it per frequency. The Float64 FEM reassembly
for Float32 runs (`BLAB_COUPLED_FEM_FLOAT64`) reassembles each region's `K_r`/`M_r`
in Float64 on the same pattern. Coupled FEM mass is always consistent.

`K_r`, `M_r` are assembled once per request over each region's tetrahedra (P1 or
P2, with the requested consistent/lumped mass blending) and stored as value arrays
on the pattern of the global air operator. Each frequency only rewrites `nzval`, so
`colptr`/`rowval` equal those of the air-only operator. Requests without porous
regions take the original operator path and give bit-identical results.

Diagnostics of porous requests list `fem_region_materials`: per region its
normalized material and, for porous regions, ρ_eq and K_eq at that frequency as
`[real, imag]`. Air-only diagnostics are unchanged.

## Current scope

- Interior FEM solves, CPU.
- Coupled FEM–BEM solves on the CPU backend (condensed and monolithic, Float64 and
  Float32), including porous regions that reach an interface to the exterior.
- Coupled solves on CUDA, ROCm and Metal refuse porous regions before mesh or device
  setup, rather than solving the fill as air.
- Regions on one mesh do not yet share seam vertices; each region is closed by its
  own boundaries.

## Validation

- `porous_material_tests.jl`: high-frequency limit ρ_eq → α∞ρ0/φ with the leading
  √2 δ_v/Λ asymptotic, K_eq → γP0/φ; low-frequency Im ρ_eq → −σ/ω (e^{+jωt}) and
  K_eq → P0/φ; passivity and phasor conjugacy; open material equals air.
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
  Further tests: an air tube anchors the piston sign, air and porous regions in one
  request match their separate solves, the porous operator keeps the air pattern,
  legacy and explicit air are bit-identical, and accelerator coupled backends refuse
  porous regions.
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
