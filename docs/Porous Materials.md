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

`K_r`, `M_r` are assembled once per request over each region's tetrahedra (P1 or
P2, with the requested consistent/lumped mass blending) and stored as value arrays
on the pattern of the global air operator. Each frequency only rewrites `nzval`, so
`colptr`/`rowval` equal those of the air-only operator. Requests without porous
regions take the original operator path and give bit-identical results.

Diagnostics of porous requests list `fem_region_materials`: per region its
normalized material and, for porous regions, ρ_eq and K_eq at that frequency as
`[real, imag]`. Air-only diagnostics are unchanged.

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
every region on that mesh that contains it.

**Bulk loss at seam nodes.** The interior operator for a request with seams is the region
operator above, so each region's `bulk_loss_factor` acts on its own elements
(b_r M_r): a seam node gets each side's mass contribution with that side's factor, and
regions with different factors (air|air or air|porous) are well defined. The coupled FEM–BEM
caches still scale mass rows per node; there a seam between regions with different bulk loss
factors is rejected ("joins different bulk loss factors"), equal factors are exact. The
aggregate's per-node `bulk_loss_factor_by_vertex` holds the first region's factor at seam nodes.

**Output layout.** `fem_nodal_pressure` keeps one row per region-local vertex, regions
contiguous in request order (`region_ids`, `node_offsets`, `node_counts`), region-local order
the sorted mesh vertex index. A seam node therefore appears once in each adjoining region's
block with one value; consumers need no change. Internally `domains[r].vertex_indices` maps
region-local vertices to the merged FEM nodes and `domains[r].face_indices` region-local
boundary triangles to the aggregated ones (0 for a dropped seam triangle).

**Diagnostics.** Results of requests with seams carry `fem_region_seams`: per adjoining region
pair `region_ids`, `mesh_id`, `face_count` and `node_count`. Requests without seams are
unchanged (no key, bit-identical results).

## Current scope

- Interior FEM solves (no FEM–BEM interfaces), CPU.
- Coupled FEM–BEM solves refuse porous regions on every backend (CPU, CUDA, ROCm,
  Metal) before mesh or device setup, rather than solving the fill as air.
- Region seams work in interior solves and, for regions of equal bulk loss, in coupled
  FEM–BEM solves of air regions (porous regions are still refused there).

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
  legacy and explicit air are bit-identical, and coupled solves refuse porous regions.
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
  workload split into two seam-joined regions equals the single-region solve (rtol 1e-9), and
  differing bulk loss across a coupled seam is rejected.
