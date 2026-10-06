#!/usr/bin/env julia
# Run through the job broker on a Metal GPU. Never regenerate baselines.
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupledCondensed
using LinearAlgebra, SparseArrays, StaticArrays, Printf

const FIXTURES = joinpath(@__DIR__, "..", "tests", "fixtures")
# Matrix differences are summation-order noise, not the solution tolerances.
const MATRIX_RTOL = 5e-6
const MATRIX_ATOL = 1e-12
const SOLUTION_RTOL = 1e-3
const FLUX_RTOL = 2e-3 # Same conditioning allowance as validate_metal_coupled.

function check(name, reference, candidate; rtol=MATRIX_RTOL, atol=MATRIX_ATOL)
    size(reference) == size(candidate) || error("$name: shape mismatch")
    all(isfinite, reference) && all(isfinite, candidate) || error("$name: nonfinite values")
    delta = norm(candidate - reference)
    scale = norm(reference)
    @printf "  %-42s relative=%.6e bound=%.6e\n" name delta / max(scale, eps(Float64)) atol + rtol * scale
    delta <= atol + rtol * scale || error("$name exceeds atol=$atol + rtol=$rtol * norm(reference)")
end

# The bundled coupled meshes span both mirror planes. Translate both together
# into the positive fundamental domain, rather than incorrectly folding a full
# object onto itself. This models separated identical mirrored coupled objects;
# it preserves connectivity, interface orientation and every physical tag.
function fixture_arm(symmetry)
    fem = load_gmsh41_volume(joinpath(FIXTURES, "femvolume.msh"), .001f0)
    bem = load_gmsh22_with_tags(joinpath(FIXTURES, "exterior_conforming.msh"), .001f0)
    dx = symmetry in (:x, :xy) ? .01f0 - min(minimum(v[1] for v in fem.vertices), minimum(v[1] for v in bem.vertices)) : 0f0
    dy = symmetry == :xy ? .01f0 - min(minimum(v[2] for v in fem.vertices), minimum(v[2] for v in bem.vertices)) : 0f0
    shift = SVector(dx, dy, 0f0)
    fem = VolumeMesh{Float32}(fem.vertices .+ Ref(shift), fem.tetrahedra, fem.tetra_physical_tags,
        fem.boundary_faces, fem.boundary_physical_tags, fem.physical_names,
        fem.quadratic_tetrahedra, fem.quadratic_boundary_faces)
    bem = BoundaryMesh(bem.vertices .+ Ref(shift), bem.faces, bem.physical_tags)
    validate_symmetry_fundamental_domain!(bem, symmetry)
    return fem, bem
end

function check_matrices(mesh, prepared, k, singular_order)
    operators = assemble_regular_galerkin_operators(mesh, prepared.p1, prepared.dp0, k, prepared.rule;
        backend=:metal, skip_singular=false, singular_order=singular_order,
        device_cache=prepared.device_cache, singular_cache=prepared.singular_cache,
        device_singular_cache=prepared.device_singular_cache, symmetry_mode=prepared.symmetry_mode)
    host_operators = nothing
    combined = nothing
    try
        host_operators = metal_host_operators(operators)
        a, rhs = burton_miller_neumann_matrices(host_operators,
            prepared.identity_p1_p1, prepared.identity_p1_dp0, k)
        combined = BeatEngineCore.assemble_coupled_burton_miller_metal(mesh, prepared, k)
        host = BeatEngineCore.metal_host_coupled_burton_miller(combined)
        check("A", a, host.a)
        check("C", -rhs, host.c)
        # Complex sources, multiple RHS, sparse signed Q and empty maps.
        f = size(rhs, 2)
        q = prepared.interface_operators.bem_flux
        motion = reshape(ComplexF32[complex(sin(i), cos(i)) for i in 1:f], :, 1)
        prescribed = hcat(motion, conj.(motion))
        blocks = BeatEngineCoupledCondensed.project_metal_coupled_host_blocks(host, q, motion, prescribed)
        check("C Q", -(rhs * ComplexF32.(q)), blocks.bem_interface_block)
        check("C motion", -(rhs * motion), blocks.bem_motion_block)
        check("-C prescribed", rhs * prescribed, blocks.bem_prescribed_rhs)
        empty_blocks = BeatEngineCoupledCondensed.project_metal_coupled_host_blocks(host,
            spzeros(Float32, f, 0), zeros(ComplexF32, f, 0), zeros(ComplexF32, f, 0))
        size(empty_blocks.bem_prescribed_rhs, 2) == 0 || error("empty projection failed")
    finally
        release_operator_storage!(host_operators === nothing ? operators : host_operators)
        combined === nothing || BeatEngineCore.release_metal_coupled_burton_miller!(combined)
    end
end

# Small fundamental-domain mesh with vertices on both mirror planes: exercises
# nontrivial orbit weights and image-singular corrections, which translations
# of the full coupled fixture cannot exercise. Also cover chunk boundaries,
# signed sparse flux maps, both storage modes and different singular part splits.
function check_plane_meshes(singular_order)
    mesh = BoundaryMesh(SVector{3,Float32}[(0,0,0),(.06,0,0),(0,.05,0),(0,0,.04)],
        [(1,3,2),(1,2,4),(1,4,3),(2,3,4)], ones(Int, 4))
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    for symmetry in (:off, :x, :xy), order in (1, 2)
        validate_symmetry_fundamental_domain!(mesh, symmetry)
        rule = triangle_rule(Float32, order)
        sc = build_singular_correction_cache(mesh, singular_order)
        for (storage, chunk, parts) in (("shared", "1", "1"), ("private", "3", "4"))
            withenv("BLAB_METAL_OPERATOR_STORAGE" => storage, "BLAB_METAL_GATHER_CHUNK" => chunk,
                    "BLAB_METAL_SINGULAR_PARTS" => parts) do
                dc = build_metal_regular_assembly_cache(mesh, p1, dp0, rule;
                    singular_order=singular_order, symmetry_mode=symmetry)
                dsc = build_metal_singular_correction_cache(sc)
                identity_rule = triangle_rule(Float32, 2)
                prepared = (p1=p1, dp0=dp0, rule=rule, device_cache=dc,
                    singular_cache=sc, device_singular_cache=dsc, symmetry_mode=symmetry,
                    identity_p1_p1=assemble_l2_identity_matrix(mesh, p1, dp0, identity_rule, :p1, :p1; symmetry_mode=symmetry),
                    identity_p1_dp0=assemble_l2_identity_matrix(mesh, p1, dp0, identity_rule, :p1, :dp0; symmetry_mode=symmetry),
                    interface_operators=(bem_flux=sparse(Float32[1 0; 0 -1; .3 .7; 0 0]),))
                try
                    for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
                        with_phasor_convention(convention) do
                            println("plane mesh $symmetry $convention q$order $storage chunk=$chunk parts=$parts")
                            check_matrices(mesh, prepared, 7f0, singular_order)
                        end
                    end
                finally
                    release_metal_regular_assembly_cache!(dc)
                    release_metal_singular_correction_cache!(dsc)
                end
            end
        end
    end
end

function check_solutions(fem, bem, mapping, cache, symmetry, order, singular_order, transducer, excitations, velocity)
    results = map(("operators", "combined")) do mode
        withenv("BLAB_METAL_COUPLED_BEM_ASSEMBLY" => mode) do
            system = build_condensed_coupled_system(fem, bem, mapping, 500f0, 343f0, 1.21f0;
                cache=cache, quadrature_order=order, singular_order=singular_order,
                symmetry_mode=symmetry, bulk_loss_factor=.01f0, transducers=[transducer],
                prescribed_bem_normal_velocity=velocity)
            try
                system.coupled_bem_assembly == Symbol(mode) || error("effective mode mismatch")
                solve_condensed_coupled_excitations(system, excitations)
            finally
                release_condensed_coupled_system!(system)
            end
        end
    end
    for (i, (reference, candidate)) in enumerate(zip(results...))
        for key in (:fem_pressure, :bem_pressure, :bem_neumann, :interface_flux,
                    :diaphragm_velocity, :voice_coil_current)
            check("excitation $i $key", getproperty(reference, key), getproperty(candidate, key);
                rtol=key == :interface_flux ? FLUX_RTOL : SOLUTION_RTOL, atol=1e-7)
        end
    end
end

# Cross-frequency reuse on one condensed cache: the cached identity scatter, gather tables and
# buffer release must survive later assemblies, including one on another task (as the coupled
# sweep pipeline's producer does). System 1 stays alive while systems 2 and 3 assemble and is
# solved again afterwards; a dangling view into a released or reused buffer changes its answer.
# UMFPACK, because the MUMPS default keeps one solver per cache (its analysis is reused across
# frequencies), so a system built earlier in the same cache is meant to be solved before the next
# build -- as the driver and the coupled sweep pipeline always do.
check_sweep(args...) = withenv(() -> _check_sweep(args...), "BLAB_COUPLED_FEM_SOLVER" => "umfpack")

function _check_sweep(fem, bem, mapping, cache, symmetry, order, singular_order, transducer, excitations, velocity)
    frequencies = (300f0, 500f0, 900f0)
    build(frequency) = build_condensed_coupled_system(fem, bem, mapping, frequency, 343f0, 1.21f0;
        cache=cache, quadrature_order=order, singular_order=singular_order,
        symmetry_mode=symmetry, bulk_loss_factor=.01f0, transducers=[transducer],
        prescribed_bem_normal_velocity=velocity)
    references = map(frequencies) do frequency
        withenv("BLAB_METAL_COUPLED_BEM_ASSEMBLY" => "operators") do
            system = build(frequency)
            try
                solve_condensed_coupled_excitations(system, excitations)
            finally
                release_condensed_coupled_system!(system)
            end
        end
    end
    withenv("BLAB_METAL_COUPLED_BEM_ASSEMBLY" => "combined") do
        first_system = build(frequencies[1])
        second_system = third_system = nothing
        try
            first_solution = solve_condensed_coupled_excitations(first_system, excitations)
            second_system = build(frequencies[2])
            third_system = fetch(Threads.@spawn build(frequencies[3]))
            candidates = (
                solve_condensed_coupled_excitations(first_system, excitations),
                solve_condensed_coupled_excitations(second_system, excitations),
                solve_condensed_coupled_excitations(third_system, excitations),
            )
            for (i, (a, b)) in enumerate(zip(first_solution, candidates[1]))
                for key in (:bem_pressure, :interface_flux, :voice_coil_current)
                    isequal(getproperty(a, key), getproperty(b, key)) ||
                        error("sweep: frequency 1 excitation $i $key changed after later assemblies")
                end
            end
            for (f, (reference, candidate)) in enumerate(zip(references, candidates))
                for (i, (r, c)) in enumerate(zip(reference, candidate))
                    for key in (:fem_pressure, :bem_pressure, :bem_neumann, :interface_flux,
                                :diaphragm_velocity, :voice_coil_current)
                        check("sweep $(frequencies[f]) Hz excitation $i $key", getproperty(r, key), getproperty(c, key);
                            rtol=key == :interface_flux ? FLUX_RTOL : SOLUTION_RTOL, atol=1e-7)
                    end
                end
            end
        finally
            for system in (first_system, second_system, third_system)
                system === nothing || release_condensed_coupled_system!(system)
            end
        end
    end
end

function main()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Run under julia_metal on Apple Silicon")
    metal.functional() || error("Metal unavailable")
    order = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "2"))
    singular_order = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "2"))
    # Force the deterministic reference configuration; diagnostic modes have
    # separate policy tests and must not silently skip this equivalence gate.
    withenv("BLAB_METAL_ASSEMBLY_MODE" => "native", "BLAB_METAL_REGULAR_KERNEL_MODE" => "pair_gather",
            "BLAB_METAL_SINGULAR_MODE" => "native", "BLAB_METAL_SINGULAR_WRITEBACK" => "gather") do
        check_plane_meshes(singular_order)
        for symmetry in (:off, :x, :xy)
            fem, bem = fixture_arm(symmetry)
            mapping = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
            radiator = physical_tag(fem, 2, "Radiator")
            transducer = ElectrodynamicTransducer{Float32}("combined-validation", [radiator], Float32[1],
                [1], Float32[-1], SVector(0f0, 0f0, 1f0), 2f0, 1,
                6f0, .0005f0, 7f0, .015f0, .0005f0, 1f0)
            moving = assemble_transducer_operators(fem, bem, [transducer])
            retained = sort(unique(vcat(mapping.fem_vertex_indices, findnz(moving.fem_surface)[1])))
            cache = prepare_condensed_coupled_cache(fem, bem, mapping;
                quadrature_order=order, singular_order=singular_order, bem_backend=:metal,
                symmetry_mode=symmetry, retained_fem_vertices=retained,
                bulk_loss_factor_by_vertex=fill(.01f0, length(fem.vertices)))
            velocity = sparse(reshape(Float32.(bem.physical_tags .== 1), :, 1))
            excitations = [
                (kind=:normal_velocity, fem_boundary_tags=[radiator], fem_boundary_weights=Float32[1],
                 bem_source_index=0, transducer_index=0, amplitude=ComplexF32(.3, .7)),
                (kind=:voltage, fem_boundary_tags=Int[], fem_boundary_weights=Float32[],
                 bem_source_index=0, transducer_index=1, amplitude=ComplexF32(2.83, -.4)),
                (kind=:normal_velocity, fem_boundary_tags=Int[], fem_boundary_weights=Float32[],
                 bem_source_index=1, transducer_index=0, amplitude=ComplexF32(-.2, .4)),
            ]
            try
                prepared = merge(cache.base, cache.quadrature_bundles[order])
                for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
                    with_phasor_convention(convention) do
                        println("$symmetry $convention q$order/s$singular_order")
                        check_matrices(bem, prepared, 2f0 * Float32(pi) * 500f0 / 343f0, singular_order)
                        check_solutions(fem, bem, mapping, cache, symmetry, order, singular_order,
                            transducer, excitations, velocity)
                        check_sweep(fem, bem, mapping, cache, symmetry, order, singular_order,
                            transducer, excitations, velocity)
                    end
                end
            finally
                release_condensed_coupled_cache!(cache)
            end
        end
    end
    println("METAL_COUPLED_COMBINED_VALIDATION_OK")
end
main()
