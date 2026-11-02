module PorousDriverTests
# Compiled-request tests for porous region materials: a request goes in, complex
# pressures come out and are compared with independent analytical references.
using Test, LinearAlgebra
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

const RHO0, C0 = 1.21, 343.0
# Soft fibrous material, Allard & Atalla 2nd ed. Table 11.2, rigid frame.
const SOFT_FIBROUS = Dict{String,Any}(
    "model" => "jca", "frame" => "rigid", "porosity" => 0.98,
    "flow_resistivity_pa_s_per_m2" => 25000.0, "tortuosity" => 1.02,
    "viscous_characteristic_length_m" => 90e-6, "thermal_characteristic_length_m" => 180e-6)

"""Analytical JCA (book form, e^{+j omega t}): Allard & Atalla Eqs. 5.50-5.55 with q0' = phi Lambda'^2/8."""
function reference_jca(parameters, frequency; rho0=RHO0, c0=C0)
    j, omega = im, 2pi * frequency
    phi = parameters["porosity"]; sigma = parameters["flow_resistivity_pa_s_per_m2"]
    alpha = parameters["tortuosity"]
    lambda_v = parameters["viscous_characteristic_length_m"]
    lambda_t = parameters["thermal_characteristic_length_m"]
    eta, gamma = 1.84e-5, 1.4
    prandtl = eta * 1005.0 / 0.0257
    p0 = rho0 * c0^2 / gamma
    rho = rho0 * alpha * (1 + sigma * phi / (j * omega * rho0 * alpha) *
          sqrt(1 + 4j * alpha^2 * eta * rho0 * omega / (sigma^2 * lambda_v^2 * phi^2)))
    bulk = gamma * p0 / (gamma - (gamma - 1) /
           (1 + 8eta / (j * lambda_t^2 * prandtl * omega * rho0) *
            sqrt(1 + j * rho0 * omega * prandtl * lambda_t^2 / (16eta))))
    return (density=rho / phi, bulk_modulus=bulk / phi)
end

"""Input impedance of a rigidly closed tube of length L: Z = -j Z_c cot(k_c L) (e^{+j omega t})."""
function reference_tube_impedance(density, bulk_modulus, frequency, length_m; convention)
    omega = 2pi * frequency
    zc, kc = sqrt(density * bulk_modulus), omega * sqrt(density / bulk_modulus)
    z = -im * zc * cot(kc * length_m)
    return convention == "exp(+i omega t)" ? z : conj(z)
end

"""Structured square tube along z, six tetrahedra per slab; tags: 1 volume, 2 z=0, 3 z=L, 4 walls."""
function tube_mesh_data(slabs; length_m, width_m=0.02, offset_x=0.0)
    vertices = [(offset_x + ix * width_m, iy * width_m, iz * length_m / slabs)
                for iz in 0:slabs for iy in 0:1 for ix in 0:1]
    node(ix, iy, iz) = 1 + ix + 2iy + 4iz
    tetrahedra = NTuple{4,Int}[]
    for iz in 0:slabs-1
        v = [node(0,0,iz), node(1,0,iz), node(0,1,iz), node(1,1,iz),
             node(0,0,iz+1), node(1,0,iz+1), node(0,1,iz+1), node(1,1,iz+1)]
        for t in ((1,2,4,8), (1,4,3,8), (1,3,7,8), (1,7,5,8), (1,5,6,8), (1,6,2,8))
            push!(tetrahedra, ntuple(i -> v[t[i]], 4))
        end
    end
    counts = Dict{NTuple{3,Int},Int}()
    for t in tetrahedra, f in ((t[1],t[2],t[3]), (t[1],t[2],t[4]), (t[1],t[3],t[4]), (t[2],t[3],t[4]))
        key = Tuple(sort(collect(f))); counts[key] = get(counts, key, 0) + 1
    end
    faces = sort!([f for (f, n) in counts if n == 1])
    tag(f) = all(v -> vertices[v][3] == 0, f) ? 2 : all(v -> vertices[v][3] == length_m, f) ? 3 : 4
    matrix(rows) = reduce(vcat, [permutedims(collect(row)) for row in rows])
    cell(kind, rows, tags) = Dict("type" => kind,
        "connectivity" => coupled_workload_packed(Int64.(matrix(rows) .- 1), "<i8"),
        "physical_tags" => coupled_workload_packed(Int64.(tags), "<i8"))
    data = Dict("schema_version" => 1,
        "points" => coupled_workload_packed(Float64.(matrix(vertices)), "<f8"),
        "cells" => [cell("triangle", faces, tag.(faces)), cell("tetra", tetrahedra, ones(Int, length(tetrahedra)))],
        "physical_names" => Dict("Volume" => [1, 3], "Piston" => [2, 2], "End" => [3, 2], "Walls" => [4, 2]))
    inlet = findall(v -> v[3] == 0, vertices)
    return data, inlet
end

"""One tube per entry `(loss_model, slabs, length_m)`, each its own mesh, region and piston port."""
function tube_request(tubes...; frequencies, convention="exp(+i omega t)")
    meshes, regions, boundaries, components, ports, inlets = [], [], [], [], [], []
    for (index, (loss_model, slabs, length_m)) in enumerate(tubes)
        data, inlet = tube_mesh_data(slabs; length_m, offset_x=0.1 * (index - 1))
        mesh, region = "mesh:tube$index", "region:tube$index"
        push!(meshes, Dict("id" => mesh, "name" => "Tube $index", "file" => "", "purpose" => "fem_volume",
            "scale_to_m" => 1.0, "translation_m" => [0.0, 0.0, 0.0], "mesh_data" => data))
        push!(regions, Dict("id" => region, "name" => "Tube $index", "kind" => "bounded_air",
            "mesh_ids" => [mesh], "volume_groups" => [Dict("mesh_id" => mesh, "dimension" => 3, "tag" => 1, "name" => "Volume")],
            "sound_speed_m_per_s" => C0, "density_kg_per_m3" => RHO0, "loss_model" => loss_model))
        for (name, kind, tag) in (("piston", "moving", 2), ("end", "rigid", 3), ("walls", "rigid", 4))
            push!(boundaries, Dict("id" => "boundary:$name$index", "name" => name, "region_id" => region,
                "kind" => kind, "group" => Dict("mesh_id" => mesh, "dimension" => 2, "tag" => tag, "name" => name),
                "parameters" => Dict()))
        end
        push!(components, Dict("id" => "component:piston$index", "name" => "Piston $index",
            "kind" => "ideal_velocity_source", "boundary_ids" => ["boundary:piston$index"], "parameters" => Dict()))
        push!(ports, Dict("id" => "excitation:piston$index", "name" => "Piston $index",
            "component_id" => "component:piston$index", "kind" => "normal_velocity"))
        push!(inlets, inlet)
    end
    request = Dict{String,Any}("schema_version" => 1,
        "compiled_system" => Dict{String,Any}("id" => "system:tube", "name" => "Filled tube", "contract_version" => 1,
            "meshes" => meshes, "regions" => regions, "boundaries" => boundaries, "interfaces" => [],
            "components" => components, "excitation_ports" => ports),
        "frequencies_hz" => collect(Float64, frequencies),
        "excitation_port_ids" => [port["id"] for port in ports],
        "outputs" => [Dict("id" => "pressure", "quantity" => "fem_nodal_pressure", "target_ids" => [], "options" => Dict())],
        "solver_options" => Dict{String,Any}("precision" => "float64", "phasor_convention" => convention))
    return JSON.parse(JSON.json(request)), inlets
end

decode(q) = copy(reinterpret(ComplexF64, base64decode(q["values"]["content_base64"])))
"""Nodal pressures as (excitation, node) and the region node offsets."""
function pressures(result)
    quantity = only(result["quantities"])
    values = decode(quantity)
    nodes = sum(quantity["metadata"]["node_counts"])
    return permutedims(reshape(values, nodes, :)), quantity["metadata"]["node_offsets"]
end

# The unit normal-velocity basis drives the boundary along its outward normal, i.e. out of
# the tube: the velocity into the fluid at the piston is -1 m/s.
input_impedance(p, excitation, inlet, offset) = -sum(p[excitation, offset .+ inlet]) / length(inlet)

@testset "porous filled tube: input impedance against -j Zc cot(kc L)" begin
    length_m, frequencies = 0.1, (100.0, 500.0, 1000.0, 2000.0)
    for convention in ("exp(+i omega t)", "exp(-i omega t)")
        errors = Dict{Int,Vector{Float64}}()
        for slabs in (40, 80)
            request, inlets = tube_request((SOFT_FIBROUS, slabs, length_m); frequencies, convention)
            run = solve_coupled_workload(request)
            @test !run.outcome.cancelled && length(run.results) == length(frequencies)
            errors[slabs] = map(zip(frequencies, run.results)) do (frequency, result)
                p, offsets = pressures(result)
                fluid = reference_jca(SOFT_FIBROUS, frequency)
                expected = reference_tube_impedance(fluid.density, fluid.bulk_modulus, frequency, length_m; convention)
                abs(input_impedance(p, 1, only(inlets), offsets[1]) - expected) / abs(expected)
            end
        end
        # Stated tolerance: 0.05 % relative impedance error with 1.25 mm P1 slabs (about 80
        # elements per porous wavelength, 99 mm at 2 kHz; measured 1.7e-4), and second-order
        # convergence under refinement (measured ratio 4.0).
        @test maximum(errors[80]) < 5e-4
        @test all(errors[80] .< errors[40] ./ 3)
    end
end

@testset "air tube anchors the piston sign: Z = -j rho0 c0 cot(k L)" begin
    length_m, frequencies = 0.1, (300.0, 1200.0)
    request, inlets = tube_request((Dict{String,Any}(), 80, length_m); frequencies)
    for (frequency, result) in zip(frequencies, solve_coupled_workload(request).results)
        p, offsets = pressures(result)
        expected = reference_tube_impedance(RHO0, RHO0 * C0^2, frequency, length_m; convention="exp(+i omega t)")
        @test input_impedance(p, 1, only(inlets), offsets[1]) ≈ expected rtol=2e-3
    end
end

@testset "air and porous regions in one request match their separate solves" begin
    lossy_air = Dict{String,Any}("bulk_loss_factor" => 0.1)
    frequencies = (250.0, 1500.0)
    mixed, _ = tube_request((lossy_air, 30, 0.1), (SOFT_FIBROUS, 30, 0.08); frequencies)
    air, _ = tube_request((lossy_air, 30, 0.1); frequencies)
    porous, _ = tube_request((SOFT_FIBROUS, 30, 0.08); frequencies)
    for (m, a, p) in zip((solve_coupled_workload(r).results for r in (mixed, air, porous))...)
        pm, offsets = pressures(m)
        pa, _ = pressures(a)
        pp, _ = pressures(p)
        tube1 = (offsets[1] + 1):offsets[2]
        tube2 = (offsets[2] + 1):size(pm, 2)
        @test pm[1, tube1] ≈ pa[1, :] rtol=1e-10
        @test pm[2, tube2] ≈ pp[1, :] rtol=1e-10
        @test norm(pm[1, tube2]) == 0 && norm(pm[2, tube1]) == 0
        materials = m["diagnostics"]["fem_region_materials"]
        @test [entry["region_id"] for entry in materials] == ["region:tube1", "region:tube2"]
        @test materials[1]["material"] == Dict("model" => "air", "bulk_loss_factor" => 0.1)
        @test materials[2]["material"] == SOFT_FIBROUS
        fluid = reference_jca(SOFT_FIBROUS, m["freq_hz"])
        @test complex(materials[2]["equivalent_density_kg_per_m3"]...) ≈ fluid.density rtol=1e-12
        @test !haskey(materials[1], "equivalent_density_kg_per_m3")
        @test !haskey(a["diagnostics"], "fem_region_materials")
    end
end

@testset "porous regions keep the air operator's sparsity pattern" begin
    request, _ = tube_request((Dict{String,Any}(), 12, 0.1), (SOFT_FIBROUS, 12, 0.1); frequencies=(800.0,))
    system = request["compiled_system"]
    domains = aggregate_fem_domains(system["meshes"], system["regions"], system["boundaries"], Float64)
    @test domains.has_porous_regions
    stiffness, mass = assemble_fem_matrices(domains.mesh)
    for weight in (1.0, 0.5)
        blended = blend_fem_mass_matrix(mass, weight)
        air = assemble_fem_dynamic_stiffness(stiffness, blended, 2pi * 800 / C0)
        operator = prepare_region_fem_operator(domains.mesh, [d.tetrahedron_range for d in domains.domains],
            [d.material for d in domains.domains], stiffness, mass; consistent_mass_weight=weight)
        porous = assemble_region_fem_operator(operator, 2pi * 800, RHO0, C0)
        @test porous.colptr == air.colptr && porous.rowval == air.rowval
        # The air region's block is the air operator itself.
        air_nodes = 1:domains.domains[1].vertex_count
        @test porous[air_nodes, air_nodes] ≈ air[air_nodes, air_nodes] rtol=1e-14
    end
end

@testset "air-only requests take the unchanged air path" begin
    frequencies = (400.0, 1700.0)
    legacy, _ = tube_request((Dict{String,Any}("bulk_loss_factor" => 0.05), 20, 0.1); frequencies)
    explicit, _ = tube_request((Dict{String,Any}("model" => "air", "bulk_loss_factor" => 0.05), 20, 0.1); frequencies)
    for (a, b) in zip(solve_coupled_workload(legacy).results, solve_coupled_workload(explicit).results)
        @test reinterpret(UInt64, decode(only(a["quantities"]))) == reinterpret(UInt64, decode(only(b["quantities"])))
        @test !haskey(a["diagnostics"], "fem_region_materials")
    end
end

@testset "coupled FEM-BEM solves refuse porous regions on accelerator backends" begin
    for backend in ("cuda", "rocm", "metal")
        request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
        interior = only(filter(r -> r["kind"] == "bounded_air", request["compiled_system"]["regions"]))
        interior["loss_model"] = SOFT_FIBROUS
        request["solver_options"]["bem_backend"] = backend
        @test validate_system_request(request) === nothing
        @test_throws "Porous region materials in coupled FEM-BEM solves are currently supported only on the CPU" solve_coupled_workload(request)
    end
end

end # module
