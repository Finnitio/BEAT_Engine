# Filled-tube fixture for the porous convergence study: a P1 or P2 tube, a compiled request
# with a piston and a rigid end, and the analytical input impedance. Included by
# porous_driver_tests.jl, porous_convergence_tests.jl and scripts/porous_convergence_study.jl;
# it expects BeatEngineCompiledDriver.jl and CompiledCoupledWorkload.jl to be included first.

const TUBE_RHO0, TUBE_C0 = 1.21, 343.0

# Rigid-frame JCA parameters of the spec's two study presets (Allard & Atalla 2nd ed.).
const SOFT_FOAM = Dict{String,Any}(  # Table 11.5
    "model" => "jca", "frame" => "rigid", "porosity" => 0.99,
    "flow_resistivity_pa_s_per_m2" => 10900.0, "tortuosity" => 1.02,
    "viscous_characteristic_length_m" => 100e-6, "thermal_characteristic_length_m" => 130e-6)
const GLASS_WOOL = Dict{String,Any}(  # Table 11.8
    "model" => "jca", "frame" => "rigid", "porosity" => 0.94,
    "flow_resistivity_pa_s_per_m2" => 40000.0, "tortuosity" => 1.06,
    "viscous_characteristic_length_m" => 56e-6, "thermal_characteristic_length_m" => 110e-6)

"""
Analytical rigid-frame JCA equivalent fluid in book form (e^{+j omega t}), written
independently of the engine: Allard & Atalla Eqs. 5.50-5.53 with q0' = phi Lambda'^2/8.
"""
function tube_reference_jca(parameters, frequency; rho0=TUBE_RHO0, c0=TUBE_C0)
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
    q0 = phi * lambda_t^2 / 8
    nu_thermal = eta / (rho0 * prandtl)
    alpha_thermal = 1 + nu_thermal * phi / (j * omega * q0) *
                    sqrt(1 + 4j * omega * q0^2 / (nu_thermal * lambda_t^2 * phi^2))
    bulk = gamma * p0 / (gamma - (gamma - 1) / alpha_thermal)
    return (density=rho / phi, bulk_modulus=bulk / phi)
end

"""Porous wavelength: the spatial period 2 pi / Re(k_eq) of the wave, k_eq = omega sqrt(rho_eq / K_eq)."""
function porous_wavelength(parameters, frequency)
    fluid = tube_reference_jca(parameters, frequency)
    return 2pi / real(2pi * frequency * sqrt(fluid.density / fluid.bulk_modulus))
end

"""Input impedance of a rigidly closed tube of length L: Z = -j Z_c cot(k_c L) (e^{+j omega t})."""
function tube_reference_impedance(parameters, frequency, length_m)
    fluid = tube_reference_jca(parameters, frequency)
    omega = 2pi * frequency
    zc, kc = sqrt(fluid.density * fluid.bulk_modulus), omega * sqrt(fluid.density / fluid.bulk_modulus)
    return -im * zc * cot(kc * length_m)
end

"""
Structured square tube along z (one hexahedron across, six tetrahedra per slab), order 1
(tetra/triangle) or 2 (tetra10/triangle6, meshio ordering). Tags: 1 volume, 2 z=0,
3 z=L, 4 walls. Returns the mesh data and the node indices on z=0.
"""
function porous_tube_mesh_data(slabs; length_m, order=1, width_m=0.02)
    vertices = [(ix * width_m, iy * width_m, iz * length_m / slabs)
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
    face_tags = tag.(faces)
    tetra_rows, face_rows = collect.(tetrahedra), collect.(faces)
    if order == 2
        midpoints = Dict{Tuple{Int,Int},Int}()
        function midpoint(a, b)
            key = minmax(a, b)
            get!(midpoints, key) do
                push!(vertices, ntuple(i -> (vertices[a][i] + vertices[b][i]) / 2, 3))
                length(vertices)
            end
        end
        # meshio/VTK edge order: tetra10 01 12 02 03 13 23, triangle6 01 12 20.
        tetra_rows = [[collect(t); [midpoint(t[a], t[b]) for (a, b) in ((1,2), (2,3), (1,3), (1,4), (2,4), (3,4))]]
                      for t in tetrahedra]
        face_rows = [[collect(f); [midpoint(f[a], f[b]) for (a, b) in ((1,2), (2,3), (3,1))]] for f in faces]
    end
    matrix(rows) = reduce(vcat, [permutedims(collect(row)) for row in rows])
    cell(kind, rows, tags) = Dict("type" => kind,
        "connectivity" => coupled_workload_packed(Int64.(matrix(rows) .- 1), "<i8"),
        "physical_tags" => coupled_workload_packed(Int64.(tags), "<i8"))
    data = Dict("schema_version" => 1,
        "points" => coupled_workload_packed(Float64.(matrix(vertices)), "<f8"),
        "cells" => [cell(order == 2 ? "triangle6" : "triangle", face_rows, face_tags),
                    cell(order == 2 ? "tetra10" : "tetra", tetra_rows, ones(Int, length(tetra_rows)))],
        "physical_names" => Dict("Volume" => [1, 3], "Piston" => [2, 2], "End" => [3, 2], "Walls" => [4, 2]))
    return data, findall(v -> v[3] == 0, vertices)
end

"""Compiled interior request of one filled tube driven by a unit-velocity piston (e^{+i omega t})."""
function porous_tube_request(material, slabs; length_m, frequencies, order=1, width_m=0.02)
    data, inlet = porous_tube_mesh_data(slabs; length_m, order, width_m)
    mesh, region = "mesh:tube", "region:tube"
    boundaries = [Dict("id" => "boundary:$name", "name" => name, "region_id" => region, "kind" => kind,
                       "group" => Dict("mesh_id" => mesh, "dimension" => 2, "tag" => tag, "name" => name),
                       "parameters" => Dict())
                  for (name, kind, tag) in (("piston", "moving", 2), ("end", "rigid", 3), ("walls", "rigid", 4))]
    system = Dict{String,Any}("id" => "system:tube", "name" => "Filled tube", "contract_version" => 1,
        "meshes" => [Dict("id" => mesh, "name" => "Tube", "file" => "", "purpose" => "fem_volume",
            "scale_to_m" => 1.0, "translation_m" => [0.0, 0.0, 0.0], "mesh_data" => data)],
        "regions" => [Dict("id" => region, "name" => "Tube", "kind" => "bounded_air", "mesh_ids" => [mesh],
            "volume_groups" => [Dict("mesh_id" => mesh, "dimension" => 3, "tag" => 1, "name" => "Volume")],
            "sound_speed_m_per_s" => TUBE_C0, "density_kg_per_m3" => TUBE_RHO0, "loss_model" => material)],
        "boundaries" => boundaries, "interfaces" => [],
        "components" => [Dict("id" => "component:piston", "name" => "Piston", "kind" => "ideal_velocity_source",
            "boundary_ids" => ["boundary:piston"], "parameters" => Dict())],
        "excitation_ports" => [Dict("id" => "excitation:piston", "name" => "Piston",
            "component_id" => "component:piston", "kind" => "normal_velocity")])
    request = Dict{String,Any}("schema_version" => 1, "compiled_system" => system,
        "frequencies_hz" => collect(Float64, frequencies), "excitation_port_ids" => ["excitation:piston"],
        "outputs" => [Dict("id" => "pressure", "quantity" => "fem_nodal_pressure", "target_ids" => [], "options" => Dict())],
        "solver_options" => Dict{String,Any}("precision" => "float64", "phasor_convention" => "exp(+i omega t)"))
    return JSON.parse(JSON.json(request)), inlet
end

# Mean edge length of the six Kuhn tetrahedra of a cube of side s: each has three axis edges,
# two face diagonals and one body diagonal. The study's element size h is this mean edge, which
# is what an isotropic mesh of target size h has on average.
const CUBE_MEAN_EDGE = (3 + 2sqrt(2) + sqrt(3)) / 6

"""
Relative input-impedance errors of a filled tube about `wavelengths` porous wavelengths long,
meshed with `elements_per_wavelength` elements of mean edge h per porous wavelength at
`f_max`, at `frequencies`. Cells are cubes, so every tetrahedron is as isotropic as an
unstructured mesh of size h: a tube wider than its slabs pollutes the plane wave with the
transverse error of long flat elements. The piston drives along its outward normal, so the
velocity into the fluid is -1 m/s.
"""
function porous_tube_errors(material, order, elements_per_wavelength; f_max, wavelengths, frequencies)
    wavelength = porous_wavelength(material, f_max)
    side = wavelength / elements_per_wavelength / CUBE_MEAN_EDGE
    slabs = max(1, round(Int, wavelengths * wavelength / side))
    length_m = slabs * side
    request, inlet = porous_tube_request(material, slabs; length_m, frequencies, order, width_m=side)
    run = solve_coupled_workload(request)
    run.outcome.cancelled && error("Filled-tube solve was cancelled.")
    return map(zip(frequencies, run.results)) do (frequency, result)
        values = copy(reinterpret(ComplexF64, base64decode(only(result["quantities"])["values"]["content_base64"])))
        impedance = -sum(values[inlet]) / length(inlet)
        expected = tube_reference_impedance(material, frequency, length_m)
        abs(impedance - expected) / abs(expected)
    end
end

# Study grid: maximum frequencies, tube lengths in porous wavelengths at f_max, and the band
# below each f_max (the mesh has N elements per wavelength at f_max and more below it).
const POROUS_STUDY_MATERIALS = ("soft foam" => SOFT_FOAM, "glass wool" => GLASS_WOOL)
const POROUS_STUDY_MAXIMUM_FREQUENCIES_HZ = (500.0, 2000.0, 5000.0, 10000.0)
const POROUS_STUDY_TUBE_WAVELENGTHS = (1, 2, 4)
const POROUS_STUDY_BAND = 0.1:0.1:1.0

"""Worst relative input-impedance error of `material` over the study grid at N elements per wavelength."""
function porous_study_worst_error(material, order, elements_per_wavelength)
    return maximum(maximum(porous_tube_errors(material, order, elements_per_wavelength;
                       f_max, wavelengths, frequencies=f_max .* POROUS_STUDY_BAND))
                   for f_max in POROUS_STUDY_MAXIMUM_FREQUENCIES_HZ for wavelengths in POROUS_STUDY_TUBE_WAVELENGTHS)
end
