module PorousCoupledTests
# Coupled FEM-BEM requests with a porous region at an interface to the exterior: a square
# tube driven by a piston radiates from its open end. The porous fill is checked against an
# independent reference: the analytical JCA transfer matrix of the filled tube, loaded by the
# radiation impedance of the same opening taken from an air-only coupled solve.
using Test, LinearAlgebra
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

include(joinpath(@__DIR__, "porous_reference.jl"))

"""
Plane-wave transfer of a uniform section of length L (e^{+j omega t}, specific impedances):
the inlet impedance p0/v0 for an outlet load zL = pL/vL, and the velocity ratio vL/v0.
"""
function reference_section(density, bulk_modulus, frequency, length_m, load)
    omega = 2pi * frequency
    zc, kc = sqrt(density * bulk_modulus), omega * sqrt(density / bulk_modulus)
    t11, t12, t21, t22 = cos(kc * length_m), im * zc * sin(kc * length_m), im * sin(kc * length_m) / zc, cos(kc * length_m)
    return (impedance=(t11 * load + t12) / (t21 * load + t22), velocity_ratio=1 / (t21 * load + t22))
end

# Cross-section of the tube on the unit square. The two interior vertices have degree 5, so the
# opening triangulation is not three-colourable and the projected DP0 interface transfer has
# full column rank (see docs/Projected interface flux.md).
const SECTION_POINTS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.525, 0.7), (0.525, 0.2875)]
const SECTION_TRIANGLES = [(3, 2, 5), (4, 3, 5), (5, 1, 4), (2, 1, 6), (6, 1, 5), (5, 2, 6)]

"""
Square tube along z extruded from `SECTION_TRIANGLES`, three tetrahedra per prism (diagonals
chosen by vertex order, so neighbouring prisms conform). FEM tags: 1 volume, 2 piston (z=0),
3 opening (z=L), 4 walls. The BEM surface is the tube's outer surface with outward normals:
tag 2 the opening, tag 1 the rigid outside of the tube.
"""
function radiating_tube_meshes(slabs; length_m, width_m)
    vertices = [(width_m * x, width_m * y, iz * length_m / slabs) for iz in 0:slabs for (x, y) in SECTION_POINTS]
    node(i, iz) = i + 6iz
    tetrahedra = NTuple{4,Int}[]
    for iz in 0:slabs-1, triangle in SECTION_TRIANGLES
        a, b, c = sort(collect(triangle))
        a0, b0, c0, a1, b1, c1 = node(a, iz), node(b, iz), node(c, iz), node(a, iz + 1), node(b, iz + 1), node(c, iz + 1)
        append!(tetrahedra, [(a0, b0, c0, c1), (a0, b0, b1, c1), (a0, a1, b1, c1)])
    end
    counts = Dict{NTuple{3,Int},Int}()
    for t in tetrahedra, f in ((t[1], t[2], t[3]), (t[1], t[2], t[4]), (t[1], t[3], t[4]), (t[2], t[3], t[4]))
        key = Tuple(sort(collect(f))); counts[key] = get(counts, key, 0) + 1
    end
    centre = [width_m / 2, width_m / 2, length_m / 2]
    point(v) = collect(vertices[v])
    outward(f) = dot(cross(point(f[2]) - point(f[1]), point(f[3]) - point(f[1])),
                     sum(point, f) / 3 - centre) > 0 ? f : (f[1], f[3], f[2])
    faces = sort!([outward(f) for (f, n) in counts if n == 1])
    tag(f) = all(v -> vertices[v][3] == 0, f) ? 2 : all(v -> vertices[v][3] == length_m, f) ? 3 : 4
    matrix(rows) = reduce(vcat, [permutedims(collect(row)) for row in rows])
    cell(kind, rows, tags) = Dict("type" => kind,
        "connectivity" => coupled_workload_packed(Int64.(matrix(rows) .- 1), "<i8"),
        "physical_tags" => coupled_workload_packed(Int64.(tags), "<i8"))
    points = coupled_workload_packed(Float64.(matrix(vertices)), "<f8")
    resource(id, purpose, data) = Dict{String,Any}("id" => id, "name" => id, "file" => "", "purpose" => purpose,
        "scale_to_m" => 1.0, "translation_m" => [0.0, 0.0, 0.0], "mesh_data" => data)
    fem = resource("mesh:tube", "fem_volume", Dict("schema_version" => 1, "points" => points,
        "cells" => [cell("triangle", faces, tag.(faces)), cell("tetra", tetrahedra, ones(Int, length(tetrahedra)))],
        "physical_names" => Dict("Volume" => [1, 3], "Piston" => [2, 2], "Opening" => [3, 2], "Walls" => [4, 2])))
    # The BEM surface keeps only the tube's surface vertices (an unused vertex is a singular P1 DOF).
    surface = sort!(unique!(reduce(vcat, collect.(faces))))
    renumber = Dict(v => i for (i, v) in enumerate(surface))
    bem_faces = [map(v -> renumber[v], f) for f in faces]
    bem = resource("mesh:outside", "bem_surface", Dict("schema_version" => 1,
        "points" => coupled_workload_packed(Float64.(matrix(vertices[surface])), "<f8"),
        "cells" => [cell("triangle", bem_faces, [tag(f) == 3 ? 2 : 1 for f in faces])],
        "physical_names" => Dict("Body" => [1, 2], "Opening" => [2, 2])))
    # P1 area weights of the piston and opening nodes (one third of each adjacent triangle).
    area(f) = norm(cross(point(f[2]) - point(f[1]), point(f[3]) - point(f[1]))) / 2
    weights(t) = begin
        w = zeros(length(vertices))
        for f in faces
            tag(f) == t && (w[collect(f)] .+= area(f) / 3)
        end
        w ./ sum(w)
    end
    return fem, bem, (piston=weights(2), opening=weights(3))
end

function radiating_tube_request(loss_model; frequencies, slabs=40, length_m=0.05, width_m=0.02,
        precision="float64", static_condensation=true, convention="exp(+i omega t)")
    fem_resource, bem_resource, weights = radiating_tube_meshes(slabs; length_m, width_m)
    fem = translated_volume_mesh(fem_resource, Float64)
    bem = translated_boundary_mesh(bem_resource, Float64)
    map = build_conforming_interface_map(fem, bem, 3, 2; coordinate_tolerance=1e-9)
    edges = Dict{Tuple{Int,Int},Int}()
    for index in map.bem_face_indices
        a, b, c = bem.faces[index]
        for (u, v) in ((a, b), (b, c), (c, a))
            edges[minmax(u, v)] = get(edges, minmax(u, v), 0) + 1
        end
    end
    group(mesh, dimension, tag, name) = Dict("mesh_id" => mesh, "dimension" => dimension, "tag" => tag, "name" => name)
    boundary(id, region, kind, mesh, tag) = Dict("id" => id, "name" => id, "region_id" => region,
        "kind" => kind, "group" => group(mesh, 2, tag, id), "parameters" => Dict())
    region(id, kind, mesh, volumes, loss) = Dict("id" => id, "name" => id, "kind" => kind, "mesh_ids" => [mesh],
        "volume_groups" => volumes, "sound_speed_m_per_s" => C0, "density_kg_per_m3" => RHO0, "loss_model" => loss)
    field_point = [width_m / 2, width_m / 2, length_m + 0.5]
    request = Dict{String,Any}(
        "schema_version" => 1,
        "compiled_system" => Dict{String,Any}(
            "id" => "system:radiating-tube", "name" => "Radiating tube", "contract_version" => 1,
            "meshes" => [bem_resource, fem_resource],
            "regions" => [
                region("region:exterior", "unbounded_air", "mesh:outside", [], Dict()),
                region("region:tube", "bounded_air", "mesh:tube", [group("mesh:tube", 3, 1, "Volume")], loss_model),
            ],
            "boundaries" => [
                boundary("boundary:piston", "region:tube", "moving", "mesh:tube", 2),
                boundary("boundary:fem-opening", "region:tube", "interface", "mesh:tube", 3),
                boundary("boundary:walls", "region:tube", "rigid", "mesh:tube", 4),
                boundary("boundary:bem-opening", "region:exterior", "interface", "mesh:outside", 2),
                boundary("boundary:body", "region:exterior", "rigid", "mesh:outside", 1),
            ],
            "interfaces" => [Dict("id" => "interface:opening", "name" => "Opening",
                "bounded_boundary_id" => "boundary:fem-opening", "unbounded_boundary_id" => "boundary:bem-opening",
                "topology" => Dict(
                    "fem_vertex_indices" => map.fem_vertex_indices .- 1,
                    "fem_to_bem_vertex_indices" => map.fem_to_bem_vertex_indices .- 1,
                    "fem_face_indices" => map.fem_face_indices .- 1,
                    "bem_face_indices" => map.bem_face_indices .- 1,
                    "normal_sign" => map.normal_sign,
                    "max_coordinate_error" => maximum(norm(fem.vertices[i] - bem.vertices[j])
                        for (i, j) in zip(map.fem_vertex_indices, map.fem_to_bem_vertex_indices)),
                    "fem_facets_on_tetra_boundary" => length(map.fem_face_indices),
                    "bem_boundary_edges" => count(==(1), values(edges))))],
            "components" => [Dict("id" => "component:piston", "name" => "Piston", "kind" => "ideal_velocity_source",
                "boundary_ids" => ["boundary:piston"], "parameters" => Dict())],
            "excitation_ports" => [Dict("id" => "excitation:piston", "name" => "Piston",
                "component_id" => "component:piston", "kind" => "normal_velocity")]),
        "frequencies_hz" => collect(Float64, frequencies),
        "excitation_port_ids" => ["excitation:piston"],
        "outputs" => [
            Dict("id" => "pressure", "quantity" => "fem_nodal_pressure", "target_ids" => [], "options" => Dict()),
            Dict("id" => "opening", "quantity" => "interface_average_normal_velocity",
                "target_ids" => ["domain:interfaces"], "options" => Dict()),
            Dict("id" => "far", "quantity" => "exterior_pressure", "target_ids" => [],
                "options" => Dict("points_m" => [field_point], "observation_domains" => [Dict("id" => "observation:far",
                    "quantity_id" => "acoustic:pressure:far", "offset" => 0, "count" => 1)])),
        ],
        "solver_options" => Dict{String,Any}("precision" => precision, "bem_backend" => "cpu",
            "quadrature_order" => 3, "singular_order" => 3, "regular_quadrature_mode" => "fixed",
            "validation_diagnostics" => false, "cache_frequency_invariant" => true,
            "static_condensation" => static_condensation, "symmetry" => "off", "phasor_convention" => convention))
    return JSON.parse(JSON.json(request)), weights
end

function decode(quantity)
    T = quantity["values"]["dtype"] == "complex64" ? ComplexF32 : ComplexF64
    return ComplexF64.(reinterpret(T, base64decode(quantity["values"]["content_base64"])))
end
quantity(result, id) = only(q for q in result["quantities"] if q["id"] == id)

"""Piston-face and opening averages of one result (one excitation): pressures and velocities into +z."""
function tube_observables(result, weights)
    p = decode(quantity(result, "pressure"))
    # The unit normal-velocity basis drives the piston along its outward normal (-z), so the
    # velocity into the tube is -1 m/s; the interface velocity is bounded-region outward (+z).
    return (inlet_pressure=dot(weights.piston, p), inlet_velocity=-1.0,
        opening_pressure=dot(weights.opening, p), opening_velocity=only(decode(quantity(result, "opening"))),
        far_pressure=only(decode(quantity(result, "far"))))
end

solve(request) = solve_coupled_workload(request).results

@testset "porous fill at a radiating opening: JCA transfer matrix with the air radiation load" begin
    frequencies = (200.0, 700.0, 1500.0)
    air_request, weights = radiating_tube_request(Dict{String,Any}(); frequencies)
    porous_request, _ = radiating_tube_request(SOFT_FIBROUS; frequencies)
    air_results, porous_results = solve(air_request), solve(porous_request)
    @test length(air_results) == length(porous_results) == length(frequencies)
    for (frequency, air_result, porous_result) in zip(frequencies, air_results, porous_results)
        @test porous_result["diagnostics"]["formulation"] == "fem_interface_condensed"
        air, porous = tube_observables(air_result, weights), tube_observables(porous_result, weights)
        # Radiation load of the opening, from the air-only solve of the same mesh.
        load = air.opening_pressure / air.opening_velocity
        fluid = reference_jca(SOFT_FIBROUS, frequency)
        expected = reference_section(fluid.density, fluid.bulk_modulus, frequency, 0.05, load)
        impedance = porous.inlet_pressure / porous.inlet_velocity
        velocity_ratio = porous.opening_velocity / porous.inlet_velocity
        # Stated tolerance: 0.5 % on the inlet impedance, 1.5 % on the opening velocity and 2 % on
        # the exterior pressure (measured at most 0.25 %, 0.86 % and 1.1 %, at 1.5 kHz). The
        # disagreement is not discretization (doubling the slabs leaves it unchanged) but the
        # plane-wave model of the opening: the near field at the opening differs between air and
        # porous fill, so the air solve's load and opening velocity profile are approximate.
        impedance_error = abs(impedance - expected.impedance) / abs(expected.impedance)
        velocity_error = abs(velocity_ratio - expected.velocity_ratio) / abs(expected.velocity_ratio)
        @test impedance_error < 5e-3
        @test velocity_error < 1.5e-2
        # The exterior field is radiated by the opening's velocity in both solves.
        @test porous.far_pressure / air.far_pressure ≈ porous.opening_velocity / air.opening_velocity rtol=2e-2
        # The reference resolves the radiation load: an ideal open end (zero load) moves the
        # opening velocity by more than five times the observed disagreement (measured 10-15x).
        open_end = reference_section(fluid.density, fluid.bulk_modulus, frequency, 0.05, 0.0im)
        @test abs(open_end.velocity_ratio - expected.velocity_ratio) / abs(expected.velocity_ratio) > 5velocity_error
        materials = porous_result["diagnostics"]["fem_region_materials"]
        @test only(materials)["material"] == SOFT_FIBROUS
        @test !haskey(air_result["diagnostics"], "fem_region_materials")
    end
end

"""Complex nodal, interface and exterior values of each result, in one vector per result."""
all_values(results) = [reduce(vcat, decode.(result["quantities"])) for result in results]

@testset "porous coupled CPU paths agree: monolithic, condensed, Float32 with Float64 FEM" begin
    frequencies = (700.0,)
    condensed = all_values(solve(first(radiating_tube_request(SOFT_FIBROUS; frequencies))))
    monolithic_results = solve(first(radiating_tube_request(SOFT_FIBROUS; frequencies, static_condensation=false)))
    @test only(monolithic_results)["diagnostics"]["formulation"] != "fem_interface_condensed"
    @test only(all_values(monolithic_results)) ≈ only(condensed) rtol=1e-9
    # Float32 condensed run with the Float64 FEM reassembly (BLAB_COUPLED_FEM_FLOAT64): the
    # reassembled operator must be the porous one, or the results would differ at O(1).
    single = withenv("BLAB_COUPLED_FEM_FLOAT64" => "on") do
        solve(first(radiating_tube_request(SOFT_FIBROUS; frequencies, precision="float32")))
    end
    @test only(single)["diagnostics"]["fem_matrix_precision"] == "float64"
    @test only(all_values(single)) ≈ only(condensed) rtol=1e-4
    plain_single = withenv("BLAB_COUPLED_FEM_FLOAT64" => "off") do
        solve(first(radiating_tube_request(SOFT_FIBROUS; frequencies, precision="float32")))
    end
    @test only(plain_single)["diagnostics"]["fem_matrix_precision"] == "float32"
    @test only(all_values(plain_single)) ≈ only(condensed) rtol=1e-3
end

@testset "porous coupled solve: negative-time convention is the complex conjugate" begin
    frequencies = (1500.0,)
    positive = all_values(solve(first(radiating_tube_request(SOFT_FIBROUS; frequencies))))
    negative = all_values(solve(first(radiating_tube_request(SOFT_FIBROUS; frequencies,
        convention="exp(-i omega t)"))))
    @test only(negative) ≈ conj.(only(positive)) rtol=1e-9
end

@testset "air-only coupled requests take the unchanged air path" begin
    legacy = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    explicit = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    interior = only(filter(r -> r["kind"] == "bounded_air", explicit["compiled_system"]["regions"]))
    interior["loss_model"] = Dict("model" => "air", "bulk_loss_factor" => interior["loss_model"]["bulk_loss_factor"])
    for (a, b) in zip(solve(legacy), solve(explicit))
        @test [reinterpret(UInt32, decode(q)) for q in a["quantities"]] ==
              [reinterpret(UInt32, decode(q)) for q in b["quantities"]]
        @test !haskey(a["diagnostics"], "fem_region_materials")
    end
end

"""
A CPU coupled cache of the radiating tube. With `porous`, the tube's first half is air and its
second half (the tetrahedra of the slabs nearer the opening) the soft fibrous fill, as two
per-region materials; otherwise the cache keeps the air operator.
"""
function radiating_tube_cache(; porous, bem_backend=:cpu, slabs=12)
    fem_resource, bem_resource, _ = radiating_tube_meshes(slabs; length_m=0.05, width_m=0.02)
    fem = translated_volume_mesh(fem_resource, Float64)
    bem = translated_boundary_mesh(bem_resource, Float64)
    map = build_conforming_interface_map(fem, bem, 3, 2; coordinate_tolerance=1e-9)
    half = length(fem.tetrahedra) ÷ 2
    regions = porous ? (tetrahedron_ranges=[1:half, (half + 1):length(fem.tetrahedra)],
        materials=[parse_region_material(Dict{String,Any}("bulk_loss_factor" => 0.03), Float64),
                   parse_region_material(SOFT_FIBROUS, Float64)]) : nothing
    return prepare_coupled_cache(fem, bem, map; quadrature_order=3, singular_order=3, bem_backend,
        bulk_loss_factor_by_vertex=fill(0.03, length(fem.vertices)), fem_regions=regions)
end

@testset "FEM operator terms (device scatter blocks) sum to the host operator on one pattern" begin
    # Air-only caches keep no region operator: the device path scatters the original K, M,
    # D_eta M blocks for them.
    @test isnothing(radiating_tube_cache(; porous=false).region_operator)
    cache = radiating_tube_cache(; porous=true)
    blocks = coupled_fem_operator_blocks(cache.region_operator)
    @test length(blocks) == 4
    patterns = Set{Tuple{Vector{Int},Vector{Int}}}()
    for frequency in (200.0, 700.0, 1500.0)
        omega = 2pi * frequency
        host = coupled_fem_dynamic_stiffness(cache, omega, C0, RHO0)
        coefficients = coupled_fem_operator_coefficients(cache.region_operator, omega, C0, RHO0)
        @test length(coefficients) == length(blocks)
        # The device path scatters each block with its coefficient into the dense system.
        scattered = zeros(ComplexF64, size(host))
        for (block, coefficient) in zip(blocks, coefficients)
            rows, columns, values = findnz(block)
            for (row, column, value) in zip(rows, columns, values)
                scattered[row, column] += coefficient * value
            end
        end
        @test scattered ≈ Matrix(host) rtol=1e-13
        # The cuDSS and MUMPS symbolic analyses are reused when the pattern is unchanged.
        push!(patterns, (copy(host.colptr), copy(host.rowval)))
    end
    @test length(patterns) == 1
    @test all(block -> (block.colptr, block.rowval) == only(patterns), blocks)
end

@testset "per-region coupled caches: CPU and CUDA only, ROCm and Metal refuse by name" begin
    for backend in (:rocm, :metal)
        @test_throws "not supported on the $backend backend" radiating_tube_cache(; porous=true, bem_backend=backend)
    end
    cuda = BeatEngineCore.CUDA_MODULE
    if cuda !== nothing && cuda.functional()
        # With a functional CUDA.jl the cache builds and holds the per-region device blocks.
        cache = radiating_tube_cache(; porous=true, bem_backend=:cuda)
        @test cache.bem_backend == :cuda
        @test !isnothing(cache.region_operator)
        @test length(cache.device_sparse_blocks.fem_operator) ==
              length(coupled_fem_operator_blocks(cache.region_operator))
        release_coupled_cache!(cache)
    else
        message = try
            radiating_tube_cache(; porous=true, bem_backend=:cuda)
            ""
        catch exception
            sprint(showerror, exception)
        end
        # Without a functional CUDA.jl the cache stops at its first device allocation, past the
        # material check.
        @test occursin("CUDA", message) && !occursin("not supported on the cuda backend", message)
    end
end

"""
The tiny coupled workload with its FEM cube split by a region seam: the six tetrahedra on the
interface face stay air (`region:interior`), the other ten are the soft fibrous fill
(`region:upper`, volume tag 6) with the radiator and the rigid walls.
"""
function porous_seam_request(; static_condensation=true, precision="float64")
    request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    request["solver_options"]["precision"] = precision
    request["solver_options"]["static_condensation"] = static_condensation
    system = request["compiled_system"]
    fem = only(m for m in system["meshes"] if m["id"] == "mesh:interior")
    tetra = only(c for c in fem["mesh_data"]["cells"] if c["type"] == "tetra")
    tetra["physical_tags"] = coupled_workload_packed(Int64[fill(1, 6); fill(6, 10)], "<i8")
    fem["mesh_data"]["physical_names"]["Upper"] = [6, 3]
    interior = only(r for r in system["regions"] if r["id"] == "region:interior")
    upper = deepcopy(interior)
    upper["id"], upper["name"], upper["loss_model"] = "region:upper", "Upper", SOFT_FIBROUS
    upper["volume_groups"] = [Dict("mesh_id" => "mesh:interior", "dimension" => 3, "tag" => 6, "name" => "Upper")]
    push!(system["regions"], upper)
    for boundary in system["boundaries"]
        boundary["id"] in ("boundary:radiator", "boundary:wall") && (boundary["region_id"] = "region:upper")
    end
    return request
end

@testset "porous seam in a coupled CPU solve: monolithic equals condensed" begin
    condensed = solve(porous_seam_request())
    monolithic = solve(porous_seam_request(; static_condensation=false))
    air = solve(JSON.parse(JSON.json(coupled_workload_request(; tiny=true))))
    @test length(condensed) == length(monolithic) == 2
    for (c, m, a) in zip(all_values(condensed), all_values(monolithic), all_values(air))
        @test c ≈ m rtol=1e-9
        # The fill changes the result (the air workload differs at O(1)).
        @test !isapprox(c, a; rtol=1e-2)
    end
    @test only(condensed[1]["diagnostics"]["fem_region_seams"])["region_ids"] == ["region:interior", "region:upper"]
    single = withenv("BLAB_COUPLED_FEM_FLOAT64" => "on") do
        solve(porous_seam_request(; precision="float32"))
    end
    for (c, s) in zip(all_values(condensed), all_values(single))
        @test s ≈ c rtol=1e-4
    end
end

"""
    porous_backend_comparison(backend; tube_frequencies)

Solve the porous tube and the air|porous seam workload on `backend` (monolithic and condensed,
Float64 and Float32) and report, per result, the relative difference to the CPU Float64 solve
of the same formulation and whether the condensed sweep reused the FEM symbolic analysis. The manual CUDA check
(BEAT_RUN_POROUS_CUDA=1 in the CUDA environment; scripts/compare_porous_backends.py) runs it
with `"cuda"`; the CPU suite runs it with `"cpu"` so the check itself stays exercised.
"""
function porous_backend_comparison(backend; tube_frequencies=(200.0, 700.0, 1500.0))
    report = NamedTuple[]
    cases = [
        ("porous tube", condensed -> first(radiating_tube_request(SOFT_FIBROUS;
            frequencies=tube_frequencies, static_condensation=condensed))),
        ("air/porous seam", condensed -> begin
            porous_seam_request(; static_condensation=condensed)
        end),
    ]
    for (name, make) in cases, condensed in (false, true)
        reference = all_values(solve(make(condensed)))
        for precision in ("float64", "float32")
            request = make(condensed)
            request["solver_options"]["precision"] = precision
            request["solver_options"]["bem_backend"] = backend
            results = solve(request)
            for (index, (a, b)) in enumerate(zip(reference, all_values(results)))
                diagnostics = results[index]["diagnostics"]
                push!(report, (case=name, condensed, precision, frequency_index=index,
                    finite=all(isfinite, b), relative_difference=norm(b - a) / norm(a),
                    analysis_reused=diagnostics["fem_symbolic_analysis_reused"],
                    materials=haskey(diagnostics, "fem_region_materials")))
            end
        end
    end
    return report
end

@testset "porous backend comparison runs (CPU against itself)" begin
    rows = porous_backend_comparison("cpu"; tube_frequencies=(700.0,))
    @test length(rows) == 2 * 2 * (1 + 2)
    @test all(row -> row.finite && row.materials, rows)
    # The CUDA tolerances below hold for the CPU paths themselves.
    @test all(row -> row.relative_difference < (row.precision == "float64" ? 1e-12 : 2e-3), rows)
end

if get(ENV, "BEAT_RUN_POROUS_CUDA", "0") == "1"
    @testset "porous regions on CUDA agree with CPU (monolithic, condensed; float64, float32)" begin
        @test BeatEngineCore.cuda_module().functional()
        for row in porous_backend_comparison("cuda")
            @info "porous CUDA vs CPU Float64" row
            @test row.finite && row.materials
            @test row.relative_difference < (row.precision == "float64" ? 1e-8 : 2e-3)
            # A condensed sweep analyses the FEM pattern once and refactors afterwards.
            row.condensed && @test row.analysis_reused == (row.frequency_index > 1)
        end
    end
end

end # module
