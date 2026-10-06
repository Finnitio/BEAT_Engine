# BEAT compiled-system wire contract

BEAT owns the numerical wire format. Boundary Lab owns project authoring,
migration, mesh preparation, and translation into that format. Independent clients
can construct JSON requests without importing Boundary Lab models, Qt, or NumPy.

## Authoritative artifacts and versions

- [JSON Schema](../src/beat_engine/beat_contract/system-v1.schema.json): request
  v1 at `urn:beat-engine:system-solve:1`; compiled system v1 and v2 at its
  `#/$defs/compiled_system` fragment.
- [Independent example](../src/beat_engine/beat_contract/example-exterior-request.json):
  a prescribed-velocity exterior request. Mesh filenames are illustrative; supply
  matching Gmsh assets before solving.
- [Conformance corpus](../src/beat_engine/beat_contract/conformance.json): shared
  acceptance/rejection cases for Python and Julia, including a coupled interface.
- [Python validator](../src/beat_engine/beat_contract/__init__.py) and
  [Julia validator](../src/beat_engine/julia_local/src/BeatEngineContract.jl):
  structural validation plus identifier/reference and topology-length checks.

The schema uses the [JSON Schema 2020-12 structure and reference conventions](https://json-schema.org/understanding-json-schema/structuring).
The two lightweight validators implement only the subset used in the shipped
schema. They are not general-purpose JSON Schema implementations. JSON numbers
must additionally be finite; NaN and infinity are never valid JSON extensions.

| Version field | Current value | Meaning |
|---|---:|---|
| Request `schema_version` | 1 | Solve-request envelope |
| Compiled system `contract_version` | 1 or 2 | V2 requires explicit support for source-motion parameters; v1 remains the uniform-normal baseline |
| Frequency result `schema_version` | 2 | Existing typed binary array representation |
| Optional `source_model_version` | Producer-defined positive integer | Advisory producer provenance only |

Compiled-system v1 retains its original uniform-normal source semantics. V2
adds opt-in source-motion fields while retaining existing normal-source requests. The
application does not define the compiled-system version independently. Request
readers reject missing, fractional, boolean, or
unsupported versions before coercing data or opening mesh files. Integral JSON
numbers such as `1.0` are accepted as integers, consistent with JSON Schema.

## Request envelope

| Field | Meaning |
|---|---|
| `compiled_system` | Resolved graph described below |
| `frequencies_hz` | Nonempty array of finite positive frequencies; order is retained |
| `excitation_port_ids` | Nonempty, unique ordered selection of graph input ports |
| `outputs` | Array of output requests with unique `id`, `quantity`, `target_ids`, and `options` |
| `solver_options` | JSON object interpreted by the numerical implementation |
| `cancel_path` | Optional local cancellation marker filename used by the worker |

Output quantities define how to interpret their targets: a target may be a graph
entity or an output domain. The structural validator does not assume every target
is a component ID. Output options include observation points and domain slices.
Backend choice, precision, quadrature, symmetry, retained fields, and transducer
reference voltage remain explicit numerical options. Syntactic acceptance of an
option or component kind does not establish backend support.

The engine contract permits repeated frequencies and retains their sequence;
the application may impose stricter sweep requirements. Excitations remain
independent columns/rows of the response basis, in the requested ID order. Channel
mixing, crossover processing, and display normalization belong to the client.
Voltage-port basis magnitude follows `transducer_reference_voltage_v`; clients
must not assume every physical input is normalized to one volt.

## Compiled graph

The exact required and optional fields are specified in the schema. IDs are
nonempty strings, unique within each entity collection. Names are display labels
and are not identifiers. Collection order is preserved by the application adapter.

| Collection | Numerical content |
|---|---|
| `meshes` | ID, filename, purpose (`bem_surface` or `fem_volume`), positive scale to meters, three-component translation in meters |
| `regions` | Bounded/unbounded air, nonempty mesh references, resolved volume groups, positive sound speed and density, loss model |
| `boundaries` | Owning region, resolved surface group, boundary kind and numerical parameters |
| `interfaces` | Bounded/unbounded boundary references and resolved FEM-to-BEM topology |
| `components` | Kind, nonempty boundary references, numerical component parameters |
| `excitation_ports` | Port ID, component reference, and normal-velocity or voltage input kind |

Resolved groups contain a mesh ID, positive integer physical tag, dimension (2
for surfaces, 3 for volumes), and optional name. Group mesh IDs must belong to
their region. Component and port references must resolve. Boundary material,
component-model, region-loss, and solver-option parameter dictionaries are
engine-interpreted extensibility points. Backend support and parameter semantics
remain validated by the existing solve-plan and numerical code.

`assumptions`, `metadata`, and `source_model_version` are optional provenance.
They do not select formulations or define the authoring-project schema. Existing
metadata used for result interpretation, such as area normalization, is preserved.
Producers may put descriptive additions in metadata; numerical features must use
their defined engine fields/options and satisfy backend capability checks.

### Exterior ideal-velocity-source motion

An `ideal_velocity_source` with a `normal_velocity` port prescribes a unit
1 m/s velocity basis. By default, each owned moving boundary has uniform
outward-normal velocity; the optional positive `boundary_motion_weights` map
multiplies that basis by boundary ID. Legacy requests without either newly
reserved `motion_profile` or `motion_axis` field keep their numerical arithmetic
and byte-identical outputs. Default-path allocation is not identical to v1;
numerical compatibility does not promise identical execution or allocation.

For a rigid piston moving along a fixed global axis, set component parameters
`motion_profile: "rigid_translation"` and `motion_axis: [x, y, z]`, and set
`compiled_system.contract_version` to `2`. The axis
must be finite and nonzero and is normalized by the worker. On each tagged
triangle the signed normal velocity is the boundary weight times
`dot(face_normal, normalized_motion_axis)` m/s. A face tangent to the motion
has zero velocity; a face whose normal opposes it has negative velocity. The
same projection weights the pressure integral reported as
`radiation_impedance` (generalized force per 1 m/s), so its unit remains N·s/m.
Each source component carries its own axis, independent of the observation
frame and of other sources. A component may own several boundaries only when
they share that rigid motion. Sources with different axes must be separate
components and ports; the client mixes their independent complex responses.
The worker never infers, snaps, flips, or takes the absolute value of an axis.
Reversing an axis reverses its Neumann drive and pressure response; the
generalized impedance remains unchanged because the force projection also
reverses. The axis uses the global mesh coordinate frame, after mesh placement.
`motion_axis` without the rigid-translation profile is an error.
A v1 ideal-source request carrying either reserved field is now refused by
design, including `motion_profile: "uniform_normal"` without an axis. This ensures
older v1-only workers reject v2 axial requests even when a caller bypasses
capability negotiation and submits raw JSON.

For a physical x/xy symmetry reduction the axis must lie in every symmetry
plane used (X=0 for x; X=0 and Y=0 for xy). A standalone y reduction is not
supported by the current exterior solver. Rigid-ground imaging is not a physical radiator copy and does not
impose that restriction. Only exterior BEM ideal sources support this profile;
interior/coupled ideal sources reject it. The worker must advertise
`exterior_source_profiles: ["uniform_normal", "rigid_translation"]` before a
client submits the rigid profile. Older workers without this capability are
rejected by negotiation, even though their v1 schema would structurally accept
the open parameters object.

CPU operator assembly and Metal direct/operator assembly consume the same
projected Neumann columns. Deployment boundary solves already accept signed
complex `boundary_neumann` traces: pass the resulting face trace in the deployed
mesh's face order, with its time convention and placement preserved. Deployment
does not interpret compiled component axes. `interface_radiated_pressure` is a
coupled-system output and cannot be requested for an exterior-only source;
coupled electrodynamic transducers retain their existing motion-axis contract.

## Coordinates, paths, topology, and complex values

- Geometry is scaled by `scale_to_m` then translated by `translation_m` in the
  global meter-based coordinate frame. Observation points use that same frame.
  Sound speed is m/s, density is kg/mÂ³, and frequency is Hz.
- V1 mesh paths are worker-local filesystem paths. Relative paths resolve against
  the worker's working directory, not the request JSON file or authoring project.
  Boundary Lab resolves project-relative mesh references before submission.
  Absolute worker-local paths avoid ambiguity. Remote asset transport is not
  defined by this contract.
- Interface indices are **zero-based** positions in the corresponding source
  mesh vertex/surface-triangle arrays, before combined-mesh renumbering. Julia
  performs the conversion to its one-based indices. Reordering a mesh without
  rebuilding its topology map invalidates the request.
- `fem_vertex_indices` and `fem_to_bem_vertex_indices` must have equal lengths.
  `fem_face_indices`, `bem_face_indices`, and `normal_sign` must have equal lengths.
  Indices are nonnegative integers; signs are exactly -1 or +1. Coordinate error
  is nonnegative and measured in meters. Actual index bounds, facet geometry,
  correspondence, interface roles, and supported formulations are checked against
  the mesh and physics by the solver.
- `solver_options.phasor_convention` explicitly selects `exp(+i omega t)` or
  legacy `exp(-i omega t)`. Omitted options retain the legacy convention.
  Workers advertise `phasor_conventions`; clients must negotiate support and
  verify the convention in result diagnostics. Boundary Lab requests positive time. Pressure, current,
  displacement/velocity, and impedance must retain complex values; SPL is not a
  substitute for the response basis.

## Result compatibility and transport

Existing frequency results contain `freq_hz`, ordered `excitation_port_ids`,
`quantities`, and `diagnostics`. Each quantity has an ID, quantity name, unit,
optional target, axis names, numeric array, and metadata. An `excitation` axis
must match the declared excitation count. Result domains define observation
coordinates outside the per-frequency payload.

Result v2 arrays use base64 bytes, numeric `dtype`, explicit `shape`, `order: C`,
and `byte_order: little`. Complex scalars store adjacent real and imaginary
components in the declared complex dtype. Byte count must match shape times
dtype size. Boundary Lab retains result-v1 decimal real/imag decoding for old
workers and historical data; new results continue to use v2. This milestone
does not change result encoding or create a new result archive format.

The existing worker command wraps a request filename and operation. Events remain
`ready`, `status`, `result`, `completed`, `cancelled`, and `failed`. Field-evaluation
operations retain their separate binary-array protocol. The
[worker protocol](BEAT%20Worker%20Protocol.md) defines version/capability negotiation,
event lifecycle, and field-cache lifetime independently of request v1.

## Evolution and validation boundary

Structural records reject unknown fields so misspellings and accidental authoring
fields do not silently reach the solver. New structural fields or changed meanings
require an explicitly supported version and migration strategy. Open metadata and
parameter/option dictionaries permit extensions without changing the surrounding
structure, but do not guarantee that a particular worker understands a new feature.

Python validates before serialization and before reconstructing application
objects. Julia validates at `solve_request` before dispatch, mesh loading, or matrix
assembly. Both use the same schema and conformance corpus. Validation of geometry,
physical feasibility, and available backend implementations is deliberately kept
in the existing numerical/solve-plan layers. Schema validation does not open files
or claim that a structurally valid request can be solved on every backend.

`system_contract.py` is now a Boundary Lab adapter: it maps a fixed list of fields
instead of calling `asdict` on the compiled object. Adding an application dataclass
field therefore cannot silently change the wire format. The `beat_contract`
directory, its schema/examples, and the Julia validator can move with BEAT when
the engine repository is extracted.

## Coupled CUDA assembly options

`solver_options.coupled_bem_assembly` accepts `auto` (default), `combined`, or
`operators`. `coupled_bem_image_fusion` defaults to true;
`coupled_bem_max_registers` defaults to 0, with explicit caps from 32 through 255.
These options affect the coupled CUDA numerical path, independently of the
exterior-only `burton_miller_assembly` option. Diagnostics retain the individual
operators. Unsupported modes/backends fail before frequency assembly. See
[Coupled CUDA Assembly](Coupled%20CUDA%20Assembly.md) for applicability and cache semantics.
`burton_miller_assembly` applies to CUDA and Metal exterior solves; on Metal,
`direct_system` selects the fused assembler described in
[Metal Backend](Metal%20Backend.md), and `coupled_bem_assembly` resolves to
`operators`.

## Exterior radiation impedance matrix

Workers may advertise the additive `radiation_impedance_matrix` optional output
quantity. Clients must negotiate this capability before submission; compiled
system versions 1 and 2 and result version 2 are unchanged. This output requires
an exterior-only system with supported component kinds: `ideal_velocity_source`
and, under compiled contract v3, `electrodynamic_transducer`.
`passive_radiator` is not implemented and fails in every solve kind.
Both server validation and the independent Python contract enforce these rules.

Request `{"id":"zrad","quantity":"radiation_impedance_matrix","target_ids":[],"options":{}}`.
Empty targets select all transducers and requested ideal components in compiled
`components` order. Explicit targets select the same ordered subset for both rows
and columns; unknown, duplicate or unrequested ideal targets fail. Undriven
transducers are valid targets. Unexcited ideal boundaries stay
rigid. Multiple ports on one component share its unit-velocity basis. Solve
requests require at least one excitation port.

The complex array has shape `[N,N]`, units `N*s/m` and axes
`["receiver_component","source_component"]`, without an excitation axis. For
unit component velocity, the definition is

`Z_ij = c_i * sum_f w_i(f) * A_f * (p_j[v1]+p_j[v2]+p_j[v3])/3`.

Here `p_j` is the unit-motion BEM basis pressure for component j, including
all symmetry images, before solving the electrical/mechanical network.
For ideal rows, `w_i` is the boundary motion weight for `uniform_normal`, or
that weight times `normal dot motion_axis` for `rigid_translation`, and
`c_i = physical_radiator_count`: 1 for off and rigid ground, 2 for x, and 4 for xy.
For transducer rows, `w_i = boundary_motion_sign * boundary_motion_weight *
(normal dot normalized_motion_axis)` and `c_i = surface_completion_factor`.
The boundary sign occurs exactly once. A single y-plane reflection would also
imply count 2, but y-only symmetry is not a supported solve mode; y reflections currently enter through xy.
Ground images contribute pressure but are not physical radiators.
Thus ideal rows integrate total force over all physical copies and the diagonal
matches the existing `radiation_impedance` self loads to round-off. Integration
uses Float64 arithmetic on the host, but the weights, axes, areas and normals
have already been rounded to the worker precision before promotion. This does
not recover accuracy lost in Float32 geometry, source parameters or BEM.

Metadata supplies `component_ids`, `kinds`, `row_weights`, `definition` and
`phasor_convention`. The definition is
`force_per_unit_velocity; per-row physical-copy weighting in row_weights`. Every ideal
row has `W_ii = 1`. Transducer rows integrate force per physical copy using
`surface_completion_factor`, and their `W_ii` is `physical_driver_orbit_count`.
Orbit never multiplies a transducer's mechanical feedback row. Metadata also
supplies `surface_completion_factors` and `physical_driver_orbit_counts`.
For ideal rows, `surface_completion_factors` carries the physical copy count
(`physical_radiator_count`), while `physical_driver_orbit_counts` is 1. The
matrix is neither symmetrized nor made passive. Its metadata diagnostics are
`reciprocity_max_rel = max(abs(W*Z - transpose(W*Z)))/max(abs(W*Z))` (zero for a
zero matrix), and `passivity_min_eig`, the minimum eigenvalue of
`(W*Z + (W*Z)')/2`, in `N*s/m`. Numerical discretization can introduce reciprocity
or passivity errors. Both `exp(+i omega t)` and `exp(-i omega t)` are supported;
real-velocity responses conjugate between these conventions.

The existing `radiation_impedance` retains its original shape, units, `radiator`
axis and compiled ideal-component ordering. It requires **every ideal component
to be excited** and refuses otherwise with a clear error; it never shrinks the
ideal radiator axis. Transducers do not appear on this axis. Its values remain
unit-motion self loads, before network superposition. Thus ideal-only v1/v2
requests retain their original ordering and bytes. Use the matrix output when
only an excited subset is wanted. Passive radiators and bounded/coupled matrix
outputs are unsupported.

The opt-in compatibility harness `scripts/compare_exterior_legacy.jl` loads the
driver and original force-integration helper from Git revision
`4839c7e62d45295489cac919c5e4b8d36b1b0f1f` into a separate module. It uses the
unchanged `two_tetrahedra.msh` fixture, two excitation ports in reversed component
order, two frequencies, both precisions, both phasors and both source profiles.
It compares every quantity object and decoded/base64 bytes exactly, including
shape, dtype, axes, units, encoding and metadata, plus frequency and excitation
order. Run-dependent timing/provenance diagnostics are outside this quantity
comparison. Other numerical dependencies are shared and unchanged from that
revision. The harness is deliberately absent from `runtests.jl` and requires the
pinned revision in the local Git object database.

From the repository root, run through the compute broker with `BROKER` and
`JULIA` set to the configured executables:

```sh
"$BROKER" submit --lane compute --expected 2 --priority 2 \
  --requester "Hornlab Fusion add-in redesign (Track B producer)" \
  --purpose "Verify exterior output bytes against the pinned legacy driver" \
  --cwd "$PWD" \
  --shell "'$JULIA' --threads=2 --startup-file=no --project=src/beat_engine/julia_local scripts/compare_exterior_legacy.jl"
"$BROKER" wait <job-id> --timeout 540
```


## Exterior electrodynamic transducers (compiled contract v3)

An exterior system (no `bounded_air` region) containing any
`electrodynamic_transducer` **requires** `compiled_system.contract_version: 3`,
even when none of its voltage ports is requested. Versions 1 and 2 refuse such
systems. Version 3 is also accepted for ideal-only and bounded/coupled systems;
it is not required there and does not opt them into changed physics. This is an
additive version extension: announcements contain `compiled_system: [1,2,3]`,
while the request envelope stays v1 and results stay v2. Source-profile version
guards accept v2 and v3. Existing readiness probes can check containment of
supported versions and ignore additional capabilities.

Workers advertise `exterior_component_kinds` as
`["ideal_velocity_source","electrodynamic_transducer"]`. Exterior submission
negotiation checks **every** compiled component, including undriven components,
against that list. A missing list means `["ideal_velocity_source"]`; merely
advertising v3 is insufficient. The independent Julia server also validates the
v3 rule, precision and limits before solving. The schema enum extends to
`[1,2,3]` without changing other schema fields. `passive_radiator` remains refused
and unadvertised.

Transducers use the coupled path's LEM parameter names, rigid-translation axis,
boundary signs/weights, optional semi-inductance and optional sealed rear chamber.
Only BEM tags in the active exterior region may be attached. BEM precision must
be explicitly `float64` (the exterior default is still float32, which is refused
here). CPU is the qualified backend for exterior transducers. Metal is explicitly
refused: Apple GPU BEM supports Float32 only. CUDA and ROCm use generic floating
types and can represent Float64; their exterior transducer paths remain
unqualified by hardware gates in this slice. Geometry, motion/force operators,
LEM parameters and the small dense complex network all use Float64. `solver_options.transducer_reference_voltage_v`
defines voltage-port amplitude (default 2.83 V, finite and positive, validated
before solving in both contract validators). This is the phasor voltage directly: the worker does not insert a square-root-of-two factor.

The motion basis contains all transducers in compiled order, followed by the
requested ideal sources, once per component. Ideal ports prescribe 1 m/s;
transducer ports prescribe reference voltage. Multiple ports on the same driver
produce equal columns. Unrequested ideal boundaries stay rigid. Every undriven
transducer has V=0 through its electrical impedance: **shorted only**, with no
open-circuit termination option.

For each frequency the existing multi-RHS BEM solves unit-motion pressure P and
Neumann Q. The shared coupled BEM helper assembles signed projection
`b = sign * weight * (normal dot normalized_axis)`, and nodal force coefficient
`completion * b * area / 3`, with exactly one motion sign. With Z = FᵀP, the
mechanical/electrical rows are

```text
(diag(Zm) + Z_DD) u_D - diag(Bl) i = -Z_DS u_S
                  diag(Bl) u_D + diag(Ze) i = V
```

Zm and Ze use the coupled engine's unchanged impedance/phasor helpers. Boundary
and field outputs are then computed from P·U and Q·U in requested-port order.
Both `exp(+i omega t)` and `exp(-i omega t)` are supported. This changes exterior
transducer physics only; ideal-only requests keep their existing solver path.

`diaphragm_velocity` and `voice_coil_current` are strictly opt-in on exterior
solves. Each requested output has shape `[excitation,transducer]`, axes
`["excitation","transducer"]`, units `m/s` and `A` respectively, and the coupled
metadata: `component_ids`, `surface_completion_factors`, and
`physical_driver_orbit_counts`, in compiled transducer order. Undriven transducers
are included. Excursion `u / time_derivative(omega)` and input impedance `V/i`
remain client calculations. The matrix output is independent of port amplitude;
its transducer rows are per-copy loads, while ideal rows retain total-copy loads.

### Signed effective volume area and acoustic conversion

`radiation_impedance_matrix.metadata.effective_volume_area_m2` supplies S_j, the
signed integral of motion factor b_j over physical moving surfaces, in matrix
component order. Real symmetry copies are included; ground images are excluded.
Transducer values use exactly the assembled force coefficients times orbit;
ideal values use the same geometry and precision as their force integration.
The metadata includes `effective_volume_area_definition`,
`effective_volume_area_cancellation_ratio` (one value per component),
`effective_volume_area_zero_or_near_cancelling` (one boolean per component), and
`effective_volume_area_cancellation_relative_tolerance: 1e-2`.
The ratio is `abs(S) / integral_physical(abs(b))`, defined as zero when the
absolute area is zero. A component is flagged when the ratio is at most
`1e-2`, including the zero-area case. This is a default cancellation threshold;
clients may apply their own threshold using the published ratio.
Closed translating spheres/dipoles have cancelling signed area; the flag is
expected and does not invalidate their mechanical impedance or pressure output.

There is no second acoustic matrix. For non-cancelling S, let
`D_S = diag(effective_volume_area_m2)` and `W = diag(row_weights)`. Clients may
convert the mechanical matrix Z_m to acoustic units with
`Z_a = D_S⁻¹ (W Z_m) D_S⁻¹` (`Pa*s/m³`). Clients must refuse this conversion for
zero-area components, and should refuse near-cancelling components according to
their chosen threshold: a dipole's local velocity cannot be represented by a
single nonzero volume flow. No absolute-value substitution for S is valid.

### Current physical limits and compatibility gates

Exterior transducers support **off and rigid ground** symmetry in this slice.
Ground has completion=orbit=1 and contributes an image only to the Green function.
`x` and `xy` transducer symmetry are explicitly refused and deferred to the next
slice; the existing coupled completion×orbit rule alone does not qualify them.
Ideal-only x/xy support is retained.

The exterior BEM mesh must consist of closed, consistently outward-wound solids.
The worker checks two oppositely oriented incidences per edge and positive signed
volume per connected shell. In ground mode, a single-incidence edge is allowed
when both endpoints lie on Y=0 (within the ground tolerance of 1e-6 m): the
reflected surface supplies the other incidence and closes the solid. Signed
volume is computed about an origin on Y=0, so a virtual ground cap contributes
zero and the reflected solid has twice the positive half-solid volume. Ground
triangles lying flat on Y=0 remain refused because they coincide with their images.

Connectivity uses vertex indices, not geometric welding. Multiple independent
closed meshes are accepted, but adjoining meshes with unwelded seams can be
refused even when their coordinates meet; weld those seams before submission.
Bodies sharing an indexed edge create a non-manifold edge (more than two
incidences) and are refused; keep disjoint closed bodies separate. These checks
do not certify self-intersections or overlapping/touching geometry.
Open or two-sided thin diaphragms are unsupported.
An open-back driver needs closed reconstructed front/rear moving surfaces and
surrounding rigid surfaces to model rear radiation. A front-only cone omits rear
loading. The optional sealed chamber supplies lumped stiffness only.
Compression drivers are not modelled by putting a transducer on a throat: there
is no compression chamber, area transformer or phase plug model. Use an ideal
throat source until those loads are modelled.

The standalone gates `tests/exterior_transducer_tests.jl` and
`tests/exterior_impedance_matrix_tests.jl` cover analytical dipole response, both
phasors, dense monolithic elimination, shorted undriven response, mixed ports,
area metadata, opt-in outputs and refusals. `scripts/compare_exterior_legacy.jl`
retains the ideal-only byte gate. `scripts/compare_coupled_transducer_operators.jl`
loads the pre-split coupled source at `10cba4fc`, compares every sparse operator's
structure and value bytes on packaged coupled fixtures in both precisions, and
compares all requested coupled result quantity bytes in both conventions.
No frozen baseline is rewritten. CPU qualification does not qualify accelerator
backends or make performance claims.
