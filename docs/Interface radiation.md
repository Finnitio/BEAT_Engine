# Interface radiation decomposition

The optional coupled output `interface_radiated_pressure` evaluates radiation
from each interface with the original operating flux held fixed. It requires
`options.points_m`, a nonempty list of finite exterior observation coordinates.
The complex result axes are `(excitation, radiation_source, observation)` in Pa.
The worker advertises this quantity in `optional_output_quantities`.

For the exterior equation `A p = C q`, the existing coupled interface block is
`B = -C F`, where `F` maps interface nodal flux to exterior face derivatives.
For interface index set `I`, replay solves `A p_I = -B[:, I] flux[I]` and uses
`q_I = F[:, I] flux[I]` in the full exterior field representation. This retains
nonuniform flux, diffraction, symmetry, and complex phase. Restricting the
original pressure trace to an interface would not perform this decomposition.

The original pressure/flux solution and all electromechanical quantities remain
unchanged. When requested, both coupled builders retain a host factorization of
the exterior matrix and its interface block. All interfaces and excitations
share that factorization. The retained matrices are copied before accelerator
assembly storage is released. No extra assembly or factorization occurs when
this output is absent. Replay field work is included in field timing.

Source IDs start with compiled interface IDs in order. An optional
`radiation:other-exterior` remainder groups all directly radiating exterior
components, including their scattering. `radiation:total` is evaluated from the
original complete solution and is not a source to add. Metadata records these
IDs and names, points, and RMS convention. Per-frequency diagnostics retain
normalized boundary reconstruction errors. The remainder is defined by subtracting interface traces from the
original traces; it is not a separate recoupled or blocked-port solution.

Tests reconstruct nonuniform interface fields, exercise source ordering and
zero excitation, and include directly radiating exterior motion. The required
standalone reference gate covers these alongside existing numerical references.
Real coupled CPU/CUDA checks should compare complex source sums to the original
total, including near cancellation where total-relative errors are sensitive.
