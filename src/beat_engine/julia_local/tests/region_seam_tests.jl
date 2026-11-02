module RegionSeamTests
# Compiled-request tests for region seams: two bounded regions on one FEM mesh whose shared
# faces carry no boundary assignment are coupled through shared nodes (ADR 0005). Complex
# pressures come out and are compared with an independent transfer-matrix reference.
using Test, LinearAlgebra
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

include(joinpath(@__DIR__, "porous_reference.jl"))
const PLUS, MINUS = "exp(+i omega t)", "exp(-i omega t)"

"""Fluid-layer transfer matrix [p; v]_front = T [p; v]_back (Allard & Atalla Ch. 11.3.1, e^{+j omega t})."""
function layer_transfer_matrix(density, bulk_modulus, frequency, thickness)
    omega = 2pi * frequency
    zc, kc = sqrt(density * bulk_modulus), omega * sqrt(density / bulk_modulus)
    return [cos(kc * thickness) im * zc * sin(kc * thickness);
            im * sin(kc * thickness) / zc cos(kc * thickness)]
end

"""
Transfer-matrix solution of an air tube of length `air_m` in front of a rigidly backed porous
layer of thickness `layer_m` (book Ch. 11.3.1 and 11.5.1), driven by a unit velocity into the
fluid at z = 0. Returns the surface impedance of the layer (Ch. 11.5.1, -j Zc cot(kc d)) and
pressure functions in the air (0 <= z <= air_m) and in the layer (air_m <= z <= air_m + layer_m).
"""
function reference_lined_tube(fluid, frequency; air_m, layer_m)
    z0 = RHO0 * C0
    surface_impedance = (layer_transfer_matrix(fluid.density, fluid.bulk_modulus, frequency, layer_m) *
                         [1.0; 0.0]) |> v -> v[1] / v[2]
    air = layer_transfer_matrix(RHO0, RHO0 * C0^2, frequency, air_m)
    # [p0; v0] = air * [Zs; 1] * v_s with v0 = 1.
    front = air * [surface_impedance; 1.0]
    seam_velocity = 1 / front[2]
    seam_pressure = surface_impedance * seam_velocity
    air_pressure(z) = (layer_transfer_matrix(RHO0, RHO0 * C0^2, frequency, air_m - z) *
                       [seam_pressure; seam_velocity])[1]
    layer_pressure(z) = (layer_transfer_matrix(fluid.density, fluid.bulk_modulus, frequency,
                         air_m + layer_m - z) * [1.0; 0.0])[1] * seam_velocity /
                        (layer_transfer_matrix(fluid.density, fluid.bulk_modulus, frequency, layer_m) *
                         [1.0; 0.0])[2]
    return (; surface_impedance, air_pressure, layer_pressure, input_impedance=front[1] / front[2],
              characteristic_impedance=z0)
end

"""
Structured square tube along z on one mesh, six tetrahedra per slab: slabs below
`front_slabs` have volume tag 1 (front), the rest tag 5 (back). Triangle tags: 2 piston (z=0),
3 rigid end, 4 front walls, 7 back walls; `seam_triangles=true` adds triangles with tag 6 on
the shared plane (Gmsh writes none for an unassigned surface). `order=2` adds edge midpoint
nodes (after the corners) and writes tetra10/triangle6 cells in meshio/VTK order.
"""
function lined_tube_mesh(front_slabs, back_slabs; front_m, back_m, width_m=0.02, seam_triangles=false, order=1)
    slabs = front_slabs + back_slabs
    # Exact plane coordinates so that the face classification below is exact.
    z_of(iz) = iz == front_slabs ? front_m : iz == slabs ? front_m + back_m :
               iz < front_slabs ? iz * front_m / front_slabs : front_m + (iz - front_slabs) * back_m / back_slabs
    vertices = [(ix * width_m, iy * width_m, z_of(iz)) for iz in 0:slabs for iy in 0:1 for ix in 0:1]
    node(ix, iy, iz) = 1 + ix + 2iy + 4iz
    tetrahedra, volume_tags = NTuple{4,Int}[], Int[]
    for iz in 0:slabs-1
        v = [node(0,0,iz), node(1,0,iz), node(0,1,iz), node(1,1,iz),
             node(0,0,iz+1), node(1,0,iz+1), node(0,1,iz+1), node(1,1,iz+1)]
        for t in ((1,2,4,8), (1,4,3,8), (1,3,7,8), (1,7,5,8), (1,5,6,8), (1,6,2,8))
            push!(tetrahedra, ntuple(i -> v[t[i]], 4))
            push!(volume_tags, iz < front_slabs ? 1 : 5)
        end
    end
    counts = Dict{NTuple{3,Int},Int}()
    for t in tetrahedra, f in ((t[1],t[2],t[3]), (t[1],t[2],t[4]), (t[1],t[3],t[4]), (t[2],t[3],t[4]))
        key = Tuple(sort(collect(f))); counts[key] = get(counts, key, 0) + 1
    end
    total_m = z_of(slabs)
    on_plane(f, z) = all(v -> vertices[v][3] == z, f)
    faces = sort!([f for (f, n) in counts if n == 1])
    tag(f) = on_plane(f, 0.0) ? 2 : on_plane(f, total_m) ? 3 :
             maximum(vertices[v][3] for v in f) <= front_m ? 4 : 7
    tags = tag.(faces)
    if seam_triangles
        seam = sort!([f for (f, n) in counts if n == 2 && on_plane(f, front_m)])
        append!(faces, seam); append!(tags, fill(6, length(seam)))
    end
    if order == 2
        midpoint = Dict{Tuple{Int,Int},Int}()
        middle(a, b) = get!(midpoint, minmax(a, b)) do
            push!(vertices, Tuple((collect(vertices[a]) .+ collect(vertices[b])) ./ 2))
            length(vertices)
        end
        # VTK edge order: tetra10 (0,1),(1,2),(0,2),(0,3),(1,3),(2,3); triangle6 (0,1),(1,2),(2,0).
        tetrahedra = [(t..., middle(t[1], t[2]), middle(t[2], t[3]), middle(t[1], t[3]),
                       middle(t[1], t[4]), middle(t[2], t[4]), middle(t[3], t[4])) for t in tetrahedra]
        faces = [(f..., middle(f[1], f[2]), middle(f[2], f[3]), middle(f[3], f[1])) for f in faces]
    end
    matrix(rows) = reduce(vcat, [permutedims(collect(row)) for row in rows])
    cell(kind, rows, tags) = Dict("type" => kind,
        "connectivity" => coupled_workload_packed(Int64.(matrix(rows) .- 1), "<i8"),
        "physical_tags" => coupled_workload_packed(Int64.(tags), "<i8"))
    names = Dict("Front" => [1, 3], "Back" => [5, 3], "Piston" => [2, 2], "End" => [3, 2],
                 "FrontWalls" => [4, 2], "BackWalls" => [7, 2])
    seam_triangles && (names["Seam"] = [6, 2])
    data = Dict("schema_version" => 1,
        "points" => coupled_workload_packed(Float64.(matrix(vertices)), "<f8"),
        "cells" => [cell(order == 2 ? "triangle6" : "triangle", faces, tags),
                    cell(order == 2 ? "tetra10" : "tetra", tetrahedra, volume_tags)],
        "physical_names" => names)
    return data, [v[3] for v in vertices]
end

"""
Request for the lined tube. `regions` is a list of `(id, loss_model, volume_tags, boundaries)`
with boundaries `(name, kind, tag)`; all regions share the one mesh.
"""
function lined_tube_request(data, regions; frequencies, convention=PLUS)
    mesh = "mesh:tube"
    boundaries = Any[]
    region_wire = map(regions) do (id, loss_model, volume_tags, region_boundaries)
        for (name, kind, tag) in region_boundaries
            push!(boundaries, Dict("id" => "boundary:$id:$name", "name" => name, "region_id" => "region:$id",
                "kind" => kind, "group" => Dict("mesh_id" => mesh, "dimension" => 2, "tag" => tag, "name" => name),
                "parameters" => Dict()))
        end
        Dict("id" => "region:$id", "name" => id, "kind" => "bounded_air", "mesh_ids" => [mesh],
            "volume_groups" => [Dict("mesh_id" => mesh, "dimension" => 3, "tag" => t, "name" => "V$t")
                                for t in volume_tags],
            "sound_speed_m_per_s" => C0, "density_kg_per_m3" => RHO0, "loss_model" => loss_model)
    end
    piston = only(b["id"] for b in boundaries if b["name"] == "piston")
    request = Dict{String,Any}("schema_version" => 1,
        "compiled_system" => Dict{String,Any}("id" => "system:lined", "name" => "Lined tube", "contract_version" => 1,
            "meshes" => [Dict("id" => mesh, "name" => "Tube", "file" => "", "purpose" => "fem_volume",
                "scale_to_m" => 1.0, "translation_m" => [0.0, 0.0, 0.0], "mesh_data" => data)],
            "regions" => region_wire, "boundaries" => boundaries, "interfaces" => [],
            "components" => [Dict("id" => "component:piston", "name" => "Piston", "kind" => "ideal_velocity_source",
                "boundary_ids" => [piston], "parameters" => Dict())],
            "excitation_ports" => [Dict("id" => "excitation:piston", "name" => "Piston",
                "component_id" => "component:piston", "kind" => "normal_velocity")]),
        "frequencies_hz" => collect(Float64, frequencies),
        "excitation_port_ids" => ["excitation:piston"],
        "outputs" => [Dict("id" => "pressure", "quantity" => "fem_nodal_pressure", "target_ids" => [], "options" => Dict())],
        "solver_options" => Dict{String,Any}("precision" => "float64", "phasor_convention" => convention))
    return JSON.parse(JSON.json(request))
end

# Partition request below at 1700 Hz solved by engine 97a56c3 (before region seams): front
# block first node, last node and sum of squared magnitudes.
const PARTITION_GOLDEN = (-4746.804427413869 - 1806.5824291044323im,
                          4730.471112453199 + 1813.2353792084134im, 1.6094718568497362e9)
const FRONT = [("piston", "moving", 2), ("walls", "rigid", 4)]
const BACK = [("end", "rigid", 3), ("backwalls", "rigid", 7)]
two_regions(front_model, back_model; front_extra=[], back_extra=[]) = [
    ("front", front_model, [1], vcat(FRONT, front_extra)),
    ("back", back_model, [5], vcat(BACK, back_extra))]
one_region(model) = [("tube", model, [1, 5], vcat(FRONT, BACK))]

decode(q) = copy(reinterpret(ComplexF64, base64decode(q["values"]["content_base64"])))
"""Nodal pressures of excitation 1 per region block, from the wire layout."""
function region_pressures(result)
    quantity = only(result["quantities"])
    values = decode(quantity)
    metadata = quantity["metadata"]
    @test length(values) == sum(metadata["node_counts"])
    return Dict(id => values[(offset + 1):(offset + count)]
                for (id, offset, count) in zip(metadata["region_ids"], metadata["node_offsets"], metadata["node_counts"]))
end

"""Mean of `values` over the nodes of each plane z = const, with the sorted plane positions."""
function section_means(values, zs)
    planes = sort(unique(zs))
    return planes, [sum(values[zs .== plane]) / count(==(plane), zs) for plane in planes]
end

# Region-local node order is the sorted full-mesh vertex index (the layout blab rebuilds):
# the front region holds the nodes with z <= front_m, the back region those with z >= front_m.
front_nodes(z, front_m) = findall(<=(front_m), z)
back_nodes(z, front_m) = findall(>=(front_m), z)

@testset "region seams" begin

@testset "region seam: air tube with a rigid-backed JCA layer matches the transfer matrix" begin
    front_m, back_m, frequencies = 0.1, 0.05, (200.0, 800.0, 2000.0)
    errors = Dict{Int,Vector{Float64}}()
    for refinement in (1, 2)
        data, z = lined_tube_mesh(40refinement, 20refinement; front_m, back_m)
        for convention in (PLUS, MINUS)
            request = lined_tube_request(data, two_regions(Dict{String,Any}(), SOFT_FIBROUS); frequencies, convention)
            run = solve_coupled_workload(request)
            @test !run.outcome.cancelled && length(run.results) == length(frequencies)
            convention == PLUS && (errors[refinement] = Float64[])
            for (frequency, result) in zip(frequencies, run.results)
                seams = result["diagnostics"]["fem_region_seams"]
                @test only(seams)["region_ids"] == ["region:front", "region:back"]
                @test only(seams)["face_count"] == 2 && only(seams)["node_count"] == 4
                blocks = region_pressures(result)
                zf, zb = z[front_nodes(z, front_m)], z[back_nodes(z, front_m)]
                pf, pb = blocks["region:front"], blocks["region:back"]
                # Seam nodes appear in both region blocks with one value.
                @test pf[zf .== front_m] == pb[zb .== front_m]
                reference = reference_lined_tube(reference_jca(SOFT_FIBROUS, frequency), frequency; air_m=front_m, layer_m=back_m)
                expected(f, zs) = convention == PLUS ? f.(zs) : conj.(f.(zs))
                # The unit normal-velocity basis drives out of the tube: velocity into the fluid is -1.
                # Compare cross-section means (the four nodes of a plane), as a tube microphone
                # sees the field; the one-element cross-section of slanted tetrahedra spreads the
                # nodal values of one plane by O(h), which cancels in the mean.
                (zfu, air), (zbu, layer) = section_means(-pf, zf), section_means(-pb, zb)
                air_error = norm(air - expected(reference.air_pressure, zfu)) / norm(expected(reference.air_pressure, zfu))
                layer_error = norm(layer - expected(reference.layer_pressure, zbu)) / norm(expected(reference.layer_pressure, zbu))
                # Surface impedance from the air field, as an impedance tube measures it: fit
                # p = A e^{-jkz} + B e^{+jkz} (lossless air, exact k) and evaluate p/v at the seam.
                k = 2pi * frequency / C0 * (convention == PLUS ? 1 : -1)
                basis = hcat(exp.(-im * k .* zfu), exp.(im * k .* zfu))
                a, b = basis \ air
                zs = RHO0 * C0 * (a * exp(-im * k * front_m) + b * exp(im * k * front_m)) /
                     (a * exp(-im * k * front_m) - b * exp(im * k * front_m))
                zs_expected = convention == PLUS ? reference.surface_impedance : conj(reference.surface_impedance)
                impedance_error = abs(zs - zs_expected) / abs(zs_expected)
                convention == PLUS && push!(errors[refinement], max(air_error, layer_error, impedance_error))
                @test air_error < 2e-3
                @test layer_error < 2e-3
                @test impedance_error < 2e-3
            end
        end
    end
    # Stated tolerance: 0.2 % relative error in the air pressure, layer pressure and surface
    # impedance with 2.5 mm P1 slabs (40 elements per porous wavelength at 2 kHz; measured
    # maximum 1.5e-3, layer pressure at 2 kHz), and second-order convergence when the slabs are
    # halved (measured ratio 4.0).
    @test all(errors[2] .< errors[1] ./ 3)
end

@testset "region seam: air|air seam reproduces the single-region tube" begin
    front_m, back_m, frequencies = 0.1, 0.05, (300.0, 1700.0)
    lossy = Dict{String,Any}("bulk_loss_factor" => 0.1)
    for (seam_triangles, order) in ((false, 1), (true, 1), (false, 2), (true, 2))
        data, z = lined_tube_mesh(30, 15; front_m, back_m, seam_triangles, order)
        seam = solve_coupled_workload(lined_tube_request(data, two_regions(lossy, lossy); frequencies)).results
        single = solve_coupled_workload(lined_tube_request(data, one_region(lossy); frequencies)).results
        for (s, r) in zip(seam, single)
            blocks = region_pressures(s)
            reference = region_pressures(r)["region:tube"]
            @test blocks["region:front"] ≈ reference[front_nodes(z, front_m)] rtol=1e-11
            @test blocks["region:back"] ≈ reference[back_nodes(z, front_m)] rtol=1e-11
            @test !haskey(r["diagnostics"], "fem_region_seams")
            # Two seam triangles: 4 corner nodes, plus 5 edge nodes for quadratic tetrahedra.
            @test only(s["diagnostics"]["fem_region_seams"])["node_count"] == (order == 2 ? 9 : 4)
        end
    end
end

@testset "region seam: bulk loss acts per region on its own elements" begin
    # Lossless air in front of a lossy air region: each region's bulk loss acts on its own
    # elements only, so the transfer matrix with a lossy back fluid is the reference.
    front_m, back_m, frequency = 0.1, 0.05, 900.0
    eta = 0.2
    data, z = lined_tube_mesh(80, 40; front_m, back_m)
    result = only(solve_coupled_workload(lined_tube_request(data,
        two_regions(Dict{String,Any}(), Dict{String,Any}("bulk_loss_factor" => eta)); frequencies=(frequency,))).results)
    # Air bulk loss: k^2 -> k^2 (1 - j eta) (e^{+j omega t}), i.e. K = rho0 c0^2 / (1 - j eta).
    lossy_air = (density=RHO0, bulk_modulus=RHO0 * C0^2 / (1 - im * eta))
    reference = reference_lined_tube(lossy_air, frequency; air_m=front_m, layer_m=back_m)
    blocks = region_pressures(result)
    @test -blocks["region:front"] ≈ reference.air_pressure.(z[front_nodes(z, front_m)]) rtol=2e-3
    @test -blocks["region:back"] ≈ reference.layer_pressure.(z[back_nodes(z, front_m)]) rtol=2e-3
end

@testset "assigned on both sides: thin rigid partition, unchanged from before seams" begin
    front_m, back_m, frequencies = 0.1, 0.05, (300.0, 1700.0)
    data, z = lined_tube_mesh(30, 15; front_m, back_m, seam_triangles=true)
    partition = two_regions(Dict{String,Any}("bulk_loss_factor" => 0.05), SOFT_FIBROUS;
        front_extra=[("partition", "rigid", 6)], back_extra=[("partition", "rigid", 6)])
    results = solve_coupled_workload(lined_tube_request(data, partition; frequencies)).results
    for (frequency, result) in zip(frequencies, results)
        @test !haskey(result["diagnostics"], "fem_region_seams")
        blocks = region_pressures(result)
        @test length(blocks["region:front"]) == 4 * 31 && length(blocks["region:back"]) == 4 * 16
        # The back region is sealed off: no pressure. The front is a rigidly closed tube.
        @test norm(blocks["region:back"]) == 0
        lossy_air = (density=RHO0, bulk_modulus=RHO0 * C0^2 / (1 - im * 0.05))
        expected = -im * sqrt(lossy_air.density * lossy_air.bulk_modulus) *
                   cot(2pi * frequency * sqrt(lossy_air.density / lossy_air.bulk_modulus) * front_m)
        piston = findall(==(0.0), z)
        @test -sum(blocks["region:front"][piston]) / 4 ≈ expected rtol=2e-2
    end
    # Pinned values from engine 97a56c3 (before region seams): the partition path is unchanged.
    front = region_pressures(results[2])["region:front"]
    @test front[1] == PARTITION_GOLDEN[1]
    @test front[end] == PARTITION_GOLDEN[2]
    @test sum(abs2, front) == PARTITION_GOLDEN[3]
end

@testset "a shared face assigned on one side only is an error naming both regions" begin
    data, _ = lined_tube_mesh(6, 3; front_m=0.1, back_m=0.05, seam_triangles=true)
    for (front_extra, back_extra, assigned) in (([("partition", "rigid", 6)], [], "front"),
                                                ([], [("partition", "rigid", 6)], "back"))
        request = lined_tube_request(data, two_regions(Dict{String,Any}(), SOFT_FIBROUS; front_extra, back_extra);
            frequencies=(500.0,))
        @test validate_system_request(request) === nothing
        system = request["compiled_system"]
        error_message = try
            aggregate_fem_domains(system["meshes"], system["regions"], system["boundaries"], Float64)
            ""
        catch error
            sprint(showerror, error)
        end
        @test occursin("\"region:front\"", error_message) && occursin("\"region:back\"", error_message)
        @test occursin("assigned only in region \"region:$assigned\"", error_message)
    end
end

@testset "aggregate_fem_domains merges seam nodes and keeps region-local rows" begin
    data, z = lined_tube_mesh(6, 3; front_m=0.1, back_m=0.05)
    request = lined_tube_request(data, two_regions(Dict{String,Any}(), SOFT_FIBROUS); frequencies=(500.0,))
    system = request["compiled_system"]
    domains = aggregate_fem_domains(system["meshes"], system["regions"], system["boundaries"], Float64)
    front, back = domains.domains
    @test (front.vertex_count, back.vertex_count) == (28, 16)
    @test length(domains.mesh.vertices) == 28 + 16 - 4
    @test (front.node_offset, back.node_offset) == (0, 28)
    # The back region's first four rows are the seam nodes shared with the front region.
    @test back.vertex_indices[1:4] == front.vertex_indices[25:28]
    @test only(domains.region_seams).node_count == 4
    # Seam faces are interior: no boundary face lies on the shared plane.
    @test !any(face -> all(v -> domains.mesh.vertices[v][3] == 0.1, face), domains.mesh.boundary_faces)
end

"""
Two square columns side by side along x on one mesh, `front_slabs + back_slabs` slabs along z,
six (Kuhn) tetrahedra per cube. Column x0 below `front_m` has volume tag 1 (A), above it tag 5
(B); column x1 has tag 8 (C). Triangle tags: 2 piston (x0, z=0), 4 walls of A, 7 walls of B,
10 walls of C, 9 the partition plane x = width between the columns. A|B is a region seam
whose rim lies on the partition, so C touches the seam nodes there. `order=2` adds edge
midpoint nodes as in `lined_tube_mesh`.
"""
function partitioned_column_mesh(front_slabs, back_slabs; front_m, back_m, width_m=0.02, order=1)
    slabs = front_slabs + back_slabs
    z_of(iz) = iz == front_slabs ? front_m : iz < front_slabs ? iz * front_m / front_slabs :
               front_m + (iz - front_slabs) * back_m / back_slabs
    vertices = [(ix * width_m, iy * width_m, z_of(iz)) for iz in 0:slabs for iy in 0:1 for ix in 0:2]
    node(ix, iy, iz) = 1 + ix + 3iy + 6iz
    tetrahedra, volume_tags = NTuple{4,Int}[], Int[]
    for iz in 0:slabs-1, ix in 0:1
        v = [node(ix,0,iz), node(ix+1,0,iz), node(ix,1,iz), node(ix+1,1,iz),
             node(ix,0,iz+1), node(ix+1,0,iz+1), node(ix,1,iz+1), node(ix+1,1,iz+1)]
        for t in ((1,2,4,8), (1,4,3,8), (1,3,7,8), (1,7,5,8), (1,5,6,8), (1,6,2,8))
            push!(tetrahedra, ntuple(i -> v[t[i]], 4))
            push!(volume_tags, ix == 1 ? 8 : iz < front_slabs ? 1 : 5)
        end
    end
    counts = Dict{NTuple{3,Int},Int}()
    for t in tetrahedra, f in ((t[1],t[2],t[3]), (t[1],t[2],t[4]), (t[1],t[3],t[4]), (t[2],t[3],t[4]))
        key = Tuple(sort(collect(f))); counts[key] = get(counts, key, 0) + 1
    end
    on_x(f, x) = all(v -> vertices[v][1] == x, f)
    exterior = sort!([f for (f, n) in counts if n == 1])
    partition = sort!([f for (f, n) in counts if n == 2 && on_x(f, width_m)])
    tag(f) = any(v -> vertices[v][1] > width_m, f) ? 10 :
             all(v -> vertices[v][3] == 0.0, f) ? 2 :
             maximum(vertices[v][3] for v in f) <= front_m ? 4 : 7
    faces, tags = vcat(exterior, partition), vcat(tag.(exterior), fill(9, length(partition)))
    if order == 2
        midpoint = Dict{Tuple{Int,Int},Int}()
        middle(a, b) = get!(midpoint, minmax(a, b)) do
            push!(vertices, Tuple((collect(vertices[a]) .+ collect(vertices[b])) ./ 2))
            length(vertices)
        end
        tetrahedra = [(t..., middle(t[1], t[2]), middle(t[2], t[3]), middle(t[1], t[3]),
                       middle(t[1], t[4]), middle(t[2], t[4]), middle(t[3], t[4])) for t in tetrahedra]
        faces = [(f..., middle(f[1], f[2]), middle(f[2], f[3]), middle(f[3], f[1])) for f in faces]
    end
    matrix(rows) = reduce(vcat, [permutedims(collect(row)) for row in rows])
    cell(kind, rows, tags) = Dict("type" => kind,
        "connectivity" => coupled_workload_packed(Int64.(matrix(rows) .- 1), "<i8"),
        "physical_tags" => coupled_workload_packed(Int64.(tags), "<i8"))
    names = Dict("A" => [1, 3], "B" => [5, 3], "C" => [8, 3], "Piston" => [2, 2], "AWalls" => [4, 2],
                 "BWalls" => [7, 2], "Partition" => [9, 2], "CWalls" => [10, 2])
    data = Dict("schema_version" => 1,
        "points" => coupled_workload_packed(Float64.(matrix(vertices)), "<f8"),
        "cells" => [cell(order == 2 ? "triangle6" : "triangle", faces, tags),
                    cell(order == 2 ? "tetra10" : "tetra", tetrahedra, volume_tags)],
        "physical_names" => names)
    return data, vertices
end

@testset "three regions: a partition rim on a seam keeps the partitioned region separate" begin
    # A|B is a region seam; C is walled off from A and B by a partition assigned on both sides.
    # The seam nodes on the partition rim belong to C as well, but C must keep its own copies:
    # C has no source (zero pressure) and A|B reproduce the single region over column x0.
    front_m, back_m, frequencies = 0.1, 0.05, (300.0, 1700.0)
    lossy = Dict{String,Any}("bulk_loss_factor" => 0.1)
    for order in (1, 2)
        data, vertices = partitioned_column_mesh(12, 6; front_m, back_m, order)
        partition = ("partition", "rigid", 9)
        a = ("a", lossy, [1], [("piston", "moving", 2), ("walls", "rigid", 4), partition])
        b = ("b", lossy, [5], [("walls", "rigid", 7), partition])
        c = ("c", Dict{String,Any}(), [8], [("walls", "rigid", 10), partition])
        column = ("column", lossy, [1, 5], [("piston", "moving", 2), ("awalls", "rigid", 4),
                                             ("bwalls", "rigid", 7), partition])
        three = lined_tube_request(data, [a, b, c]; frequencies)
        system = three["compiled_system"]
        domains = aggregate_fem_domains(system["meshes"], system["regions"], system["boundaries"], Float64)
        seam_nodes = order == 2 ? 9 : 4
        @test [seam.region_ids for seam in domains.region_seams] == [["region:a", "region:b"]]
        @test length(domains.mesh.vertices) == sum(d.vertex_count for d in domains.domains) - seam_nodes
        x0 = [i for (i, v) in enumerate(vertices) if v[1] <= 0.02]
        za = [vertices[i][3] for i in x0]
        split_results = solve_coupled_workload(three).results
        single_results = solve_coupled_workload(lined_tube_request(data, [column]; frequencies)).results
        for (s, r) in zip(split_results, single_results)
            blocks = region_pressures(s)
            reference = region_pressures(r)["region:column"]
            @test norm(blocks["region:c"]) == 0
            @test blocks["region:a"] ≈ reference[za .<= front_m] rtol=1e-11
            @test blocks["region:b"] ≈ reference[za .>= front_m] rtol=1e-11
        end
    end
end

"""
The tiny coupled workload with its FEM cube split into two bounded regions on the one mesh:
the six tetrahedra on the interface (bottom) face stay in `region:interior`, the other ten
move to `region:upper` (volume tag 6) with the radiator and the side walls. The faces between
them carry no triangle: a region seam.
"""
function split_coupled_request(; upper_loss=Dict{String,Any}("bulk_loss_factor" => 0.02))
    request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    request["solver_options"]["precision"] = "float64"
    system = request["compiled_system"]
    fem = only(m for m in system["meshes"] if m["id"] == "mesh:interior")
    tetra = only(c for c in fem["mesh_data"]["cells"] if c["type"] == "tetra")
    tetra["physical_tags"] = coupled_workload_packed(Int64[fill(1, 6); fill(6, 10)], "<i8")
    fem["mesh_data"]["physical_names"]["Upper"] = [6, 3]
    interior = only(r for r in system["regions"] if r["id"] == "region:interior")
    upper = deepcopy(interior)
    upper["id"], upper["name"], upper["loss_model"] = "region:upper", "Upper Air", upper_loss
    upper["volume_groups"] = [Dict("mesh_id" => "mesh:interior", "dimension" => 3, "tag" => 6, "name" => "Upper")]
    push!(system["regions"], upper)
    for boundary in system["boundaries"]
        boundary["id"] in ("boundary:radiator", "boundary:wall") && (boundary["region_id"] = "region:upper")
    end
    return request
end

@testset "region seam in a coupled FEM-BEM solve reproduces the single region" begin
    single = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    single["solver_options"]["precision"] = "float64"
    split = split_coupled_request()
    @test validate_system_request(split) === nothing
    for request in (single, split)
        push!(request["outputs"], Dict("id" => "pressure", "quantity" => "fem_nodal_pressure",
            "target_ids" => [], "options" => Dict()))
    end
    # Region-local rows are the sorted full-mesh vertices of each region (0-based cube corners
    # 0-5 and centre 10 at the bottom; corners 0-3, 6-9 and centre 10 above).
    bottom, top = [0, 1, 2, 3, 4, 5, 10] .+ 1, [0, 1, 2, 3, 6, 7, 8, 9, 10] .+ 1
    for (s, r) in zip(solve_coupled_workload(split).results, solve_coupled_workload(single).results)
        seam = only(s["diagnostics"]["fem_region_seams"])
        @test seam["region_ids"] == ["region:interior", "region:upper"]
        @test seam["face_count"] == 4  # the four bottom edges, each with the centre vertex
        pressure(result) = only(q for q in result["quantities"] if q["quantity"] == "fem_nodal_pressure")
        blocks = region_pressures(Dict("quantities" => [pressure(s)]))
        reference = decode(pressure(r))
        @test blocks["region:interior"] ≈ reference[bottom] rtol=1e-9
        @test blocks["region:upper"] ≈ reference[top] rtol=1e-9
        velocity(result) = decode(only(q for q in result["quantities"] if q["quantity"] == "diaphragm_velocity"))
        @test velocity(s) ≈ velocity(r) rtol=1e-9
    end
end

@testset "coupled seam with differing bulk loss: per-region operator on CPU and CUDA, refused on ROCm and Metal" begin
    # CPU coupled solves form each region's bulk loss on its own elements, as interior solves do.
    lossy = Dict{String,Any}("bulk_loss_factor" => 0.05)
    condensed = solve_coupled_workload(split_coupled_request(; upper_loss=lossy)).results
    monolithic_request = split_coupled_request(; upper_loss=lossy)
    monolithic_request["solver_options"]["static_condensation"] = false
    monolithic = solve_coupled_workload(monolithic_request).results
    equal = solve_coupled_workload(split_coupled_request()).results
    velocity(result) = decode(only(q for q in result["quantities"] if q["quantity"] == "diaphragm_velocity"))
    for (c, m, e) in zip(condensed, monolithic, equal)
        @test velocity(c) ≈ velocity(m) rtol=1e-9
        # The extra loss in the upper region changes the load on the diaphragm.
        @test !isapprox(velocity(c), velocity(e); rtol=1e-4)
        materials = c["diagnostics"]["fem_region_materials"]
        @test [entry["material"]["bulk_loss_factor"] for entry in materials] == [0.02, 0.05]
    end
    for backend in ("rocm", "metal")
        request = split_coupled_request(; upper_loss=lossy)
        request["solver_options"]["bem_backend"] = backend
        @test_throws "joins different bulk loss factors" solve_coupled_workload(request)
    end
    # CUDA uses the per-region operator too: without a device the solve fails at the first
    # device call, not at the seam check.
    request = split_coupled_request(; upper_loss=lossy)
    request["solver_options"]["bem_backend"] = "cuda"
    message = try
        solve_coupled_workload(request)
        ""
    catch exception
        sprint(showerror, exception)
    end
    @test occursin("CUDA", message) && !occursin("bulk loss", message)
end

end # region seams

end # module
