# Shared compiled-contract workload. Full requests use the frozen, packaged
# coupled fixtures; the CPU bundle/CI use a five-vertex analogue of that graph.
function coupled_workload_mesh(id, name, file, purpose)
    return Dict{String,Any}("id" => id, "name" => name, "file" => file,
        "purpose" => purpose, "scale_to_m" => 0.001,
        "translation_m" => [0.0, 0.0, 0.0])
end

function coupled_workload_packed(values, dtype)
    flat = ndims(values) == 1 ? vec(values) : vec(permutedims(values))
    bits = htol.(reinterpret(UInt64, flat))
    return Dict("dtype" => dtype, "shape" => collect(size(values)),
        "data" => base64encode(reinterpret(UInt8, bits)))
end

function tiny_coupled_workload_meshes()
    # A tetrahedron split about its centroid leaves an actual FEM interior
    # vertex even when the radiator vertices are retained. All faces point out.
    points = Float64[0 0 0; 80 0 0; 0 80 0; 0 0 80; 20 20 20]
    faces = Int64[0 2 1; 0 1 3; 1 2 3; 2 0 3]
    tets = Int64[0 2 1 4; 0 1 3 4; 1 2 3 4; 2 0 3 4]
    cell(kind, indices, tags) = Dict("type" => kind,
        "connectivity" => coupled_workload_packed(indices, "<i8"),
        "physical_tags" => coupled_workload_packed(tags, "<i8"))
    fem = coupled_workload_mesh("mesh:interior", "Interior", "", "fem_volume")
    fem["mesh_data"] = Dict("schema_version" => 1,
        "points" => coupled_workload_packed(points, "<f8"),
        "cells" => [cell("triangle", faces, Int64[3, 2, 4, 4]),
                    cell("tetra", tets, ones(Int64, 4))],
        "physical_names" => Dict("Volume" => [1, 3], "Radiator" => [2, 2],
                                 "Interface" => [3, 2], "Volume_boundary" => [4, 2]))
    bem = coupled_workload_mesh("mesh:exterior", "Exterior", "", "bem_surface")
    bem["mesh_data"] = Dict("schema_version" => 1,
        "points" => coupled_workload_packed(points[1:4, :], "<f8"),
        "cells" => [cell("triangle", faces, Int64[2, 1, 1, 1])],
        "physical_names" => Dict("ExteriorBox" => [1, 2], "Interface" => [2, 2]))
    return fem, bem
end

function coupled_workload_request(; tiny::Bool=false, bem_backend::String="cpu")
    root = joinpath(ENGINE_DIR, "tests", "fixtures")
    fem_resource, bem_resource = if tiny
        tiny_coupled_workload_meshes()
    else
        files = (joinpath(root, "femvolume.msh"), joinpath(root, "exterior_conforming.msh"))
        all(isfile, files) || error("Packaged coupled workload fixtures missing under $root")
        (coupled_workload_mesh("mesh:interior", "Interior", files[1], "fem_volume"),
         coupled_workload_mesh("mesh:exterior", "Exterior", files[2], "bem_surface"))
    end
    fem = translated_volume_mesh(fem_resource, Float64)
    bem = translated_boundary_mesh(bem_resource, Float64)
    map = build_conforming_interface_map(fem, bem, 3, 2; coordinate_tolerance=1e-6)
    edges = Dict{Tuple{Int,Int},Int}()
    for index in map.bem_face_indices
        a, b, c = bem.faces[index]
        for (u, v) in ((a, b), (b, c), (c, a))
            edge = minmax(u, v)
            edges[edge] = get(edges, edge, 0) + 1
        end
    end
    group(mesh, dimension, tag, name) = Dict("mesh_id" => mesh,
        "dimension" => dimension, "tag" => tag, "name" => name)
    boundary(id, name, region, kind, mesh, tag) = Dict("id" => id, "name" => name,
        "region_id" => region, "kind" => kind, "group" => group(mesh, 2, tag, name),
        "parameters" => Dict())
    region(id, name, kind, mesh, volumes, loss) = Dict("id" => id, "name" => name,
        "kind" => kind, "mesh_ids" => [mesh], "volume_groups" => volumes,
        "sound_speed_m_per_s" => 343.0, "density_kg_per_m3" => 1.21, "loss_model" => loss)
    output(id, quantity, targets; options=Dict()) = Dict("id" => id, "quantity" => quantity,
        "target_ids" => targets, "options" => options)
    return Dict{String,Any}(
        "schema_version" => 1,
        "compiled_system" => Dict(
            "id" => "system:coupled-warmup", "name" => "Coupled warmup", "contract_version" => 1,
            "meshes" => [bem_resource, fem_resource],
            "regions" => [
                region("region:exterior", "Exterior Air", "unbounded_air", "mesh:exterior", [], Dict()),
                region("region:interior", "Interior Air", "bounded_air", "mesh:interior",
                    [group("mesh:interior", 3, 1, "Volume")], Dict("bulk_loss_factor" => 0.02)),
            ],
            "boundaries" => [
                boundary("boundary:radiator", "Radiator", "region:interior", "moving", "mesh:interior", 2),
                boundary("boundary:fem-interface", "Interface", "region:interior", "interface", "mesh:interior", 3),
                boundary("boundary:wall", "Volume_boundary", "region:interior", "rigid", "mesh:interior", 4),
                boundary("boundary:bem-interface", "Interface", "region:exterior", "interface", "mesh:exterior", 2),
                boundary("boundary:exterior", "ExteriorBox", "region:exterior", "rigid", "mesh:exterior", 1),
            ],
            "interfaces" => [Dict("id" => "interface:opening", "name" => "Interface",
                "bounded_boundary_id" => "boundary:fem-interface",
                "unbounded_boundary_id" => "boundary:bem-interface",
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
            "components" => [Dict("id" => "component:driver", "name" => "Radiator",
                "kind" => "electrodynamic_transducer", "boundary_ids" => ["boundary:radiator"],
                "parameters" => Dict("bl_n_per_a" => 7.9, "cms_m_per_n" => 0.00086,
                    "le_h" => 0.00029, "mmd_kg" => 0.0397, "re_ohm" => 3.3,
                    "rms_n_s_per_m" => 1.25, "motion_profile" => "rigid_translation",
                    "motion_axis" => (tiny ? [0.0, 1.0, 0.0] : [0.0, 0.0, 1.0]),
                    "symmetry_role" => "complete_representative", "surface_completion_factor" => 1,
                    "physical_driver_orbit_count" => 1, "fractional_symmetry_axes" => []))],
            "excitation_ports" => [Dict("id" => "port:voltage", "name" => "Radiator voltage",
                "component_id" => "component:driver", "kind" => "voltage")]),
        "frequencies_hz" => [500.0, 1000.0], "excitation_port_ids" => ["port:voltage"],
        "outputs" => [
            output("ui:exterior-pressure", "exterior_pressure", [];
                options=Dict("points_m" => [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                    "observation_domains" => [Dict("id" => "observation:warmup",
                        "quantity_id" => "acoustic:pressure:warmup", "offset" => 0, "count" => 2)])),
            output("mechanical:diaphragm-velocity", "diaphragm_velocity", ["components:electrodynamic-transducers"]),
            output("electrical:voice-coil-current", "voice_coil_current", ["components:electrodynamic-transducers"]),
            output("acoustic:interface-average-normal-velocity", "interface_average_normal_velocity", ["domain:interfaces"]),
        ],
        "solver_options" => Dict("precision" => "float32", "bem_backend" => bem_backend,
            "quadrature_order" => 1, "singular_order" => 1, "regular_quadrature_mode" => "fixed",
            "validation_diagnostics" => false, "cache_frequency_invariant" => true,
            "static_condensation" => true, "symmetry" => "off"))
end

# Resolve the engine's Metal defaults with installation-machine overrides
# cleared, then apply them to CPU BEM assembly. Never launch an engine kernel
# while generating a package image (as in the exterior workload).
function coupled_workload_environment(; mumps::Bool)
    cc = BeatEngineCoupledCondensed
    modes = (
        "BLAB_COUPLED_TRANSDUCER_CONDENSATION" => cc._transducer_condensation_mode,
        "BLAB_COUPLED_DENSE_FLOAT64" => b -> cc._coupled_mode("BLAB_COUPLED_DENSE_FLOAT64", b; metal_default=:off),
        "BLAB_COUPLED_DENSE_REFINEMENT" => b -> cc._coupled_mode("BLAB_COUPLED_DENSE_REFINEMENT", b),
        "BLAB_COUPLED_FEM_FLOAT64" => b -> cc._coupled_mode("BLAB_COUPLED_FEM_FLOAT64", b),
        "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION" => cc._interface_pressure_elimination_mode,
        "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => cc._interface_flux_elimination_mode,
        "BLAB_COUPLED_INTERFACE_MASS_SOLVER" => cc._interface_mass_solver_selection,
        "BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => b -> cc._coupled_mode("BLAB_COUPLED_INTERFACE_MASS_OVERLAP", b),
        "BLAB_COUPLED_INTERFACE_BLOCKS" => cc._interface_blocks_mode,
        "BLAB_COUPLED_DEMAND_RECONSTRUCTION" => b -> cc._coupled_mode("BLAB_COUPLED_DEMAND_RECONSTRUCTION", b),
        "BLAB_COUPLED_FEM_SOLVER" => cc._fem_solver_selection,
    )
    # Every coupled and MUMPS override from the installing environment is cleared, so the
    # workload resolves the engine's own defaults (quadrature, Schur blocks, threads included).
    inherited = [name for name in keys(ENV)
                 if startswith(name, "BLAB_COUPLED_") || startswith(name, "BLAB_MUMPS_")]
    resolved = withenv((name => nothing for name in inherited)..., (name => nothing for (name, _) in modes)...) do
        Dict(name => string(select(:metal)) for (name, select) in modes)
    end
    # One entry per name, applied in order: clear the inherited override, then the resolved
    # workload default, then the host workload's own solver and overlap choices.
    effective = Dict{String,Union{Nothing,String}}(name => nothing for name in inherited)
    merge!(effective, resolved)
    mumps || (effective["BLAB_COUPLED_FEM_SOLVER"] = "umfpack")
    # There is no device work to overlap in the surrogate host workload.
    effective["BLAB_COUPLED_STAGE_OVERLAP"] = "off"
    return Pair{String,Union{Nothing,String}}[name => effective[name] for name in sort!(collect(keys(effective)))]
end

function solve_coupled_workload(request)
    # Capture the real JSON event path on a file-backed stream (redirect_stdout
    # cannot use an IOBuffer). Return diagnostics for precompile and CI gates.
    return mktemp() do _, io
        outcome = redirect_stdout(io) do
            solve_request(request; event_mode=true)
        end
        flush(io)
        seekstart(io)
        events = [JSON.parse(line) for line in eachline(io) if !isempty(line)]
        results = [event["result"] for event in events if get(event, "type", nothing) == "result"]
        return (; outcome, results)
    end
end

function check_coupled_workload(run; mumps::Bool)
    @assert !run.outcome.cancelled
    @assert run.outcome.solved_count == 2 && length(run.results) == 2
    for result in run.results
        d = result["diagnostics"]
        @assert d["formulation"] == "fem_interface_condensed"
        @assert d["fem_condensation_backend"] == (mumps ? "mumps_seq" : "cpu_umfpack") d
        @assert d["linear_solver"] == (mumps ? "cpu_mumps_seq_schur_plus_dense_lu" : "cpu_umfpack_schur_plus_dense_lu") d
        @assert d["interface_elimination"] == "flux" d
        @assert d["interface_mass_solver"] == "cholmod" d
        @assert d["fem_matrix_precision"] == "float64" d
        @assert d["transducer_count"] == 1 && d["interface_count"] == 1 d
        @assert d["fem_interior_reconstruction"] == "skipped" d
        @assert isempty(d["coupled_optimization_fallback_reasons"]) d
        @assert d["dense_solver"] in ("lu_float32_refined", "lu_float64_fallback") d
        if d["dense_solver"] == "lu_float64_fallback"
            @warn "BEAT coupled workload refinement fell back" reason=d["dense_refinement_fallback_reason"]
        end
    end
    mumps && @assert run.results[2]["diagnostics"]["fem_symbolic_analysis_reused"]
    d = run.results[end]["diagnostics"]
    @info "BEAT compiled coupled workload complete" frequencies=length(run.results) fem_condensation_backend=d["fem_condensation_backend"] interface_mass_solver=d["interface_mass_solver"] dense_solver=d["dense_solver"]
    return nothing
end

function reset_compiled_workload_state!()
    for cleanup in (release_all_bem_field_evaluation_caches!,
                    BeatEngineCoupledCondensed.BeatEngineMumps.reset_precompile_state!,
                    BeatEngineCoupledCondensed.reset_accelerate_zgemm!)
        try
            cleanup()
        catch exception
            @warn "BEAT compiled workload native cleanup failed" cleanup exception=(exception, catch_backtrace())
        end
    end
    provenance = BeatEngineContract.BeatEngineProvenance
    provenance.IDENTITY[] = nothing
    provenance.RUNTIME[] = nothing
    RUN_MESH_PROVENANCE[] = Any[]
    # The driver releases per-request systems/caches in finally. Collect the
    # now-unreachable CHOLMOD/UMFPACK factors before writing the package image.
    GC.gc(true)
    return nothing
end

function precompile_coupled_workload(; tiny::Bool, mumps::Bool)
    try
        request = JSON.parse(JSON.json(coupled_workload_request(; tiny)))
        withenv(coupled_workload_environment(; mumps)...) do
            check_coupled_workload(solve_coupled_workload(request); mumps)
        end
    catch exception
        @warn "BEAT compiled coupled workload failed" tiny mumps exception=(exception, catch_backtrace())
    finally
        try
            reset_compiled_workload_state!()
        catch exception
            @warn "BEAT compiled coupled workload cleanup failed" exception=(exception, catch_backtrace())
        end
    end
end
