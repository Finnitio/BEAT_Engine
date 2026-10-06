# Sequential MUMPS Schur backend (BLAB_COUPLED_FEM_SOLVER=mumps). Needs the helpers in
# coupled_condensed_test_setup.jl and an environment that lists MUMPS_seq_jll: the Metal one.
# Included by coupled_condensed_tests.jl when the package is present, and by mumps_tests.jl.

@testset "Condensed MUMPS Schur backend algebra" begin
    library = BeatEngineCoupledCondensed.BeatEngineMumps.mumps_library()
    # This file runs only where MUMPS_seq_jll is in the environment, so a library that does not
    # load here is a packaging regression, not a reason to skip.
    @test library.available
    @test library.version == "5.9.1"
    @test isempty(BeatEngineCoupledCondensed.BeatEngineMumps.layout_mismatches())
    relative(reference, candidate) = norm(ComplexF64.(candidate) - ComplexF64.(reference)) /
                                     norm(ComplexF64.(reference))

    for T in (Float64, Float32)
        base, operators, retained = condensed_synthetic_case(T)
        # MUMPS runs LDLᵀ (SYM=2), so the synthetic block has to be complex symmetric like a FEM one.
        system = SparseMatrixCSC{Complex{T},Int}(base + transpose(base))
        vertex_count = size(system, 1)
        interior = setdiff(1:vertex_count, retained)
        tolerance = T === Float64 ? 1e-12 : 1e-5
        reference = BeatEngineCoupledCondensed._build_condensation(system, operators, retained)
        candidate = BeatEngineCoupledCondensed._build_condensation(
            system, operators, retained; fem_solver=:mumps,
        )
        try
            @test reference.backend == :cpu_umfpack
            @test candidate.backend == :mumps_seq
            @test candidate.fem_solver_requested == :mumps
            @test isnothing(candidate.fem_solver_fallback_reason)
            @test candidate.mumps_owned
            @test eltype(candidate.schur) == Complex{T}
            @test relative(reference.schur, candidate.schur) < tolerance
            planted = ComplexF64[1 + 0.1im * v for v in 1:vertex_count, _ in 1:2]
            planted[:, 2] .*= -0.5
            forcing = Complex{T}.(Matrix(system * planted))
            reduced_reference, interior_reference = BeatEngineCoupledCondensed._forward_schur(reference, forcing)
            reduced_candidate, interior_candidate = BeatEngineCoupledCondensed._forward_schur(candidate, forcing)
            @test eltype(reduced_candidate) == Complex{T}
            @test relative(reduced_reference, reduced_candidate) < tolerance
            @test interior_candidate == interior_reference
            recovered, residual = BeatEngineCoupledCondensed._backward_schur(
                candidate, interior_candidate, candidate.schur \ reduced_candidate,
            )
            @test relative(planted, recovered) < (T === Float64 ? 1e-10 : 1e-4)
            @test maximum(residual) < (T === Float64 ? 1e-10 : 1e-5)
            # A Γ-only forcing leaves f_I = 0; the interior rows must still annihilate the result.
            gamma_forcing = zeros(Complex{T}, vertex_count, 1)
            gamma_forcing[retained, 1] .= 1 + 0.5im
            gamma_reduced, gamma_interior = BeatEngineCoupledCondensed._forward_schur(candidate, gamma_forcing)
            gamma_recovered, gamma_residual = BeatEngineCoupledCondensed._backward_schur(
                candidate, gamma_interior, candidate.schur \ gamma_reduced,
            )
            @test maximum(gamma_residual) < (T === Float64 ? 1e-10 : 1e-5)
            @test norm(ComplexF64.(system[interior, :]) * ComplexF64.(gamma_recovered[:, 1])) /
                  norm(gamma_recovered) < (T === Float64 ? 1e-10 : 1e-5)
        finally
            BeatEngineCoupledCondensed._release_condensation!(reference)
            BeatEngineCoupledCondensed._release_condensation!(candidate)
        end
        # An owned solver is freed with its condensation.
        @test !candidate.mumps_solver.initialized

        # Transducer condensation: the low-rank fields from one reduction and one internal solve
        # must equal UMFPACK's explicit interior solves and transpose solves.
        surface = sparse([interior[1], interior[4], retained[2], interior[9], retained[5]],
            [1, 1, 1, 2, 2], T[0.3, 0.2, 0.25, 0.4, 0.1], vertex_count, 2)
        force = sparse([interior[1], interior[4], retained[2], interior[9], retained[5]],
            [1, 1, 1, 2, 2], T[0.6, 0.1, 0.5, 0.2, 0.3], vertex_count, 2)
        motion_reference = BeatEngineCoupledCondensed._build_condensation(
            system, operators, retained; motion_surface=surface, motion_force=force,
        )
        motion_candidate = BeatEngineCoupledCondensed._build_condensation(
            system, operators, retained; motion_surface=surface, motion_force=force, fem_solver=:mumps,
        )
        try
            @test motion_candidate.backend == :mumps_seq
            @test motion_candidate.transducer_condensed
            for field in (:motion_gamma, :force_gamma, :motion_solution, :force_solution, :motion_force_correction)
                @test relative(getproperty(motion_reference, field), getproperty(motion_candidate, field)) <
                      (T === Float64 ? 1e-12 : 1e-6)
            end
            @test motion_candidate.motion_interior == motion_reference.motion_interior
        finally
            BeatEngineCoupledCondensed._release_condensation!(motion_reference)
            BeatEngineCoupledCondensed._release_condensation!(motion_candidate)
        end
    end

    base, operators, retained = condensed_synthetic_case(Float64)
    symmetric = SparseMatrixCSC{ComplexF64,Int}(base + transpose(base))
    umfpack = BeatEngineCoupledCondensed._build_condensation(symmetric, operators, retained)

    # The analysis lives in the store and is reused while the pattern and Γ are unchanged.
    store = Dict{Symbol,Any}()
    first_build = BeatEngineCoupledCondensed._build_condensation(
        symmetric, operators, retained; fem_solver=:mumps, mumps_store=store,
    )
    shifted = symmetric + 0.75im * sparse(I, size(symmetric)...)
    second_build = BeatEngineCoupledCondensed._build_condensation(
        shifted, operators, retained; fem_solver=:mumps, mumps_store=store,
    )
    @test !first_build.analysis_reused
    @test second_build.analysis_reused
    @test second_build.mumps_solver === first_build.mumps_solver === store[:solver]
    @test !second_build.mumps_owned && second_build.factorization_cached
    @test store[:solver].analysis_count == 1
    shifted_reference = BeatEngineCoupledCondensed._build_condensation(shifted, operators, retained)
    @test relative(shifted_reference.schur, second_build.schur) < 1e-12
    # Releasing a cached condensation leaves the solver to its store.
    BeatEngineCoupledCondensed._release_condensation!(second_build)
    @test store[:solver].initialized
    # A new pattern (one more symmetric pair) is re-analysed, still correct.
    widened = copy(symmetric)
    free_pair = first((i, j) for i in 1:size(widened, 1), j in 1:size(widened, 2) if i > j && iszero(widened[i, j]))
    widened[free_pair...] = 0.1 - 0.2im
    widened[reverse(free_pair)...] = 0.1 - 0.2im
    widened_build = BeatEngineCoupledCondensed._build_condensation(
        widened, operators, retained; fem_solver=:mumps, mumps_store=store,
    )
    @test !widened_build.analysis_reused
    @test store[:solver].analysis_count == 2
    widened_reference = BeatEngineCoupledCondensed._build_condensation(widened, operators, retained)
    @test relative(widened_reference.schur, widened_build.schur) < 1e-12
    # A different thread count replaces the solver.
    threaded_build = withenv("BLAB_MUMPS_THREADS" => "2") do
        BeatEngineCoupledCondensed._build_condensation(
            widened, operators, retained; fem_solver=:mumps, mumps_store=store,
        )
    end
    @test threaded_build.mumps_threads == 2
    @test !widened_build.mumps_solver.initialized
    @test relative(widened_reference.schur, threaded_build.schur) < 1e-12
    solver = store[:solver]
    @test_throws "BLAB_MUMPS_THREADS" withenv(() -> BeatEngineCoupledCondensed.BeatEngineMumps.mumps_threads(),
        "BLAB_MUMPS_THREADS" => "0")

    # SYM=2 reads one triangle, so an unsymmetric block must not be factored by MUMPS: it falls
    # back to UMFPACK with the reason, and the store is cleared of the suspect solver.
    # Same pattern as the analysed matrix, so this reaches the numeric symmetry check.
    unsymmetric = copy(widened)
    unsymmetric[free_pair...] += 1e-6
    fallback = @test_logs (:warn,) match_mode = :any BeatEngineCoupledCondensed._build_condensation(
        unsymmetric, operators, retained; fem_solver=:mumps, mumps_store=store,
    )
    @test fallback.backend == :cpu_umfpack
    @test fallback.fem_solver_requested == :mumps
    @test occursin("complex-symmetric", fallback.fem_solver_fallback_reason)
    @test !haskey(store, :solver) && !solver.initialized
    @test fallback.schur ≈ BeatEngineCoupledCondensed._build_condensation(unsymmetric, operators, retained).schur rtol = 1e-12
    # A structurally unsymmetric block is refused at analysis.
    structural = @test_logs (:warn,) match_mode = :any BeatEngineCoupledCondensed._build_condensation(
        base, operators, retained; fem_solver=:mumps,
    )
    @test structural.backend == :cpu_umfpack
    @test occursin("structurally symmetric", structural.fem_solver_fallback_reason)

    # No retained vertices (no interfaces, transducer surfaces condensed): MUMPS Schur mode cannot
    # take an empty Schur set, so UMFPACK runs without a failed MUMPS attempt and says why.
    empty_count = size(symmetric, 1)
    empty_operators = InterfaceOperators(
        spzeros(Float64, empty_count, 0), spzeros(Float64, 4, 0), spzeros(Float64, 0, empty_count), spzeros(Float64, 0, 4),
    )
    empty_surface = sparse([2, 7, 11], [1, 1, 1], [0.3, 0.2, 0.25], empty_count, 1)
    empty_force = sparse([2, 7, 13], [1, 1, 1], [0.6, 0.1, 0.5], empty_count, 1)
    empty_store = Dict{Symbol,Any}()
    empty_mumps = @test_logs min_level = Base.CoreLogging.Warn BeatEngineCoupledCondensed._build_condensation(
        symmetric, empty_operators, Int[];
        motion_surface=empty_surface, motion_force=empty_force, fem_solver=:mumps, mumps_store=empty_store,
    )
    empty_umfpack = BeatEngineCoupledCondensed._build_condensation(
        symmetric, empty_operators, Int[]; motion_surface=empty_surface, motion_force=empty_force,
    )
    @test empty_mumps.backend == :cpu_umfpack
    @test empty_mumps.fem_solver_requested == :mumps
    @test occursin("Schur set is empty", empty_mumps.fem_solver_fallback_reason)
    @test isempty(empty_store)
    @test size(empty_mumps.schur) == (0, 0)
    @test norm(empty_mumps.motion_solution) > 0
    for field in (:motion_solution, :force_solution, :motion_force_correction)
        @test getproperty(empty_mumps, field) == getproperty(empty_umfpack, field)
    end

    # A double-precision FEM block (BLAB_COUPLED_FEM_FLOAT64) with single-precision operators:
    # MUMPS factors it and demotes the Schur block to the operators' type, like UMFPACK.
    base32, operators32, retained32 = condensed_synthetic_case(Float32)
    base64, _, retained64 = condensed_synthetic_case(Float64)
    @test retained32 == retained64
    widened = SparseMatrixCSC{ComplexF64,Int}(base64 + transpose(base64))
    widened_umfpack = BeatEngineCoupledCondensed._build_condensation(widened, operators32, retained32)
    widened_mumps = BeatEngineCoupledCondensed._build_condensation(widened, operators32, retained32; fem_solver=:mumps)
    @test widened_mumps.backend == :mumps_seq
    @test eltype(widened_mumps.schur) == ComplexF32 && eltype(widened_umfpack.schur) == ComplexF32
    @test relative(widened_umfpack.schur, widened_mumps.schur) < 1e-6
    BeatEngineCoupledCondensed._release_condensation!(widened_mumps)

    # A library that cannot load degrades to UMFPACK and says so.
    hook = BeatEngineCoupledCondensed.BeatEngineMumps.FORCE_UNAVAILABLE
    hook[] = true
    unavailable = try
        BeatEngineCoupledCondensed._build_condensation(symmetric, operators, retained; fem_solver=:mumps)
    finally
        hook[] = false
    end
    @test unavailable.backend == :cpu_umfpack
    @test startswith(unavailable.fem_solver_fallback_reason, "MUMPS unavailable")
    @test unavailable.schur == umfpack.schur
    @test isnothing(umfpack.fem_solver_fallback_reason) && umfpack.fem_solver_requested == :umfpack

    # The environment switch.
    @test withenv(BeatEngineCoupledCondensed._fem_solver_selection, "BLAB_COUPLED_FEM_SOLVER" => nothing) == :umfpack
    @test withenv(BeatEngineCoupledCondensed._fem_solver_selection, "BLAB_COUPLED_FEM_SOLVER" => "MUMPS") == :mumps
    @test_throws "BLAB_COUPLED_FEM_SOLVER" withenv(BeatEngineCoupledCondensed._fem_solver_selection,
        "BLAB_COUPLED_FEM_SOLVER" => "pardiso")

    # Release is idempotent (JOB=-2 once).
    BeatEngineCoupledCondensed.BeatEngineMumps.mumps_release!(solver)
    @test !solver.initialized
end

if get(ENV, "BLAB_RUN_COUPLED_REFERENCE", "0") == "1"
    with_baseline_cpu_coupled() do
        @testset "Condensed MUMPS Schur backend matches UMFPACK and monolithic" begin
            fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
            bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
            interface_map = build_conforming_interface_map(
                fem_mesh, bem_mesh, physical_tag(fem_mesh, 2, "Interface"), 2,
            )
            radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
            options = (quadrature_order=CONDENSED_QUADRATURE_ORDER, singular_order=CONDENSED_SINGULAR_ORDER)
            relative(reference, candidate) = norm(ComplexF64.(candidate) .- ComplexF64.(reference)) /
                                             max(norm(ComplexF64.(reference)), eps(Float64))
            transducer = ElectrodynamicTransducer{Float64}(
                "component:test", [radiator_tag], [1.0], [1], [-1.0], SVector(0.0, 0.0, 1.0),
                2.0, 1, 6.0, 0.0005, 7.0, 0.015, 0.0005, 1.0,
            )
            velocity = (kind=:normal_velocity, radiator_tag=radiator_tag, transducer_index=0, amplitude=ComplexF64(0.3, -0.2))
            voltage = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))
            voltage_b = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(-0.4, 1.3))
            fields = (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)
            mumps = "BLAB_COUPLED_FEM_SOLVER" => "mumps"
            umfpack = "BLAB_COUPLED_FEM_SOLVER" => "umfpack"
            tc = "BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1"
            scenarios = (
                (label="plain", switches=(), transducers=ElectrodynamicTransducer{Float64}[], excitations=[velocity]),
                (label="retained transducer", switches=(), transducers=[transducer], excitations=[voltage, velocity]),
                (label="transducer condensation", switches=(tc,), transducers=[transducer], excitations=[voltage, velocity]),
                (label="tc + pressure elimination", switches=(tc, "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION" => "1"),
                    transducers=[transducer], excitations=[voltage, velocity, voltage_b]),
                (label="tc + flux elimination", switches=(tc, "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1"),
                    transducers=[transducer], excitations=[voltage, velocity, voltage_b]),
                (label="plain + flux elimination", switches=("BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1",),
                    transducers=ElectrodynamicTransducer{Float64}[], excitations=[velocity]),
                (label="tc + flux + dense f64 + overlap", switches=(tc, "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1",
                        "BLAB_COUPLED_DENSE_FLOAT64" => "1", "BLAB_COUPLED_STAGE_OVERLAP" => "on"),
                    transducers=[transducer], excitations=[voltage, velocity]),
                (label="tc + flux specialized + dense f64 + overlap", switches=(tc, "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1",
                        "BLAB_COUPLED_DENSE_FLOAT64" => "1", "BLAB_COUPLED_STAGE_OVERLAP" => "on",
                        "BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "cholmod", "BLAB_COUPLED_INTERFACE_BLOCKS" => "1",
                        "BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "1"),
                    transducers=[transducer], excitations=[voltage, velocity, voltage_b]),
                (label="plain + flux blocks, mass in FEM stage", switches=("BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1",
                        "BLAB_COUPLED_INTERFACE_BLOCKS" => "1", "BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "1"),
                    transducers=ElectrodynamicTransducer{Float64}[], excitations=[velocity]),
            )
            for scenario in scenarios
                @testset "$(scenario.label)" begin
                    transducer_operators = assemble_transducer_operators(fem_mesh, bem_mesh, scenario.transducers)
                    retained = sort(unique(vcat(
                        interface_map.fem_vertex_indices,
                        isempty(scenario.transducers) ? Int[] : findnz(transducer_operators.fem_surface)[1],
                    )))
                    cache_for(switches...) = withenv(switches...) do
                        prepare_condensed_coupled_cache(fem_mesh, bem_mesh, interface_map; options..., retained_fem_vertices=retained)
                    end
                    umfpack_cache = cache_for(umfpack)
                    mumps_cache = cache_for(mumps)
                    try
                        # Three frequencies through one cache: the first analyses, the rest refactor.
                        for (index, frequency) in enumerate((600.0, 900.0, 350.0))
                            build(cache, switch) = withenv(switch, scenario.switches...) do
                                build_condensed_coupled_system(
                                    fem_mesh, bem_mesh, interface_map, frequency, 343.0, 1.21;
                                    options..., cache=cache, transducers=scenario.transducers,
                                )
                            end
                            reference_system = index == 1 ? build_coupled_system(
                                fem_mesh, bem_mesh, interface_map, frequency, 343.0, 1.21;
                                options..., validation_diagnostics=false, bem_backend=:cpu,
                                transducers=scenario.transducers,
                            ) : nothing
                            baseline = build(umfpack_cache, umfpack)
                            candidate = build(mumps_cache, mumps)
                            try
                                @test candidate.condensation.backend == :mumps_seq
                                @test isnothing(candidate.condensation.fem_solver_fallback_reason)
                                @test candidate.condensation.analysis_reused == (index > 1)
                                @test candidate.condensation.mumps_solver === mumps_cache.mumps_store[:solver]
                                @test candidate.solved_system_order == baseline.solved_system_order
                                @test relative(baseline.condensation.schur, candidate.condensation.schur) < 1e-10
                                baselines = solve_condensed_coupled_excitations(baseline, scenario.excitations)
                                candidates = solve_condensed_coupled_excitations(candidate, scenario.excitations)
                                references = isnothing(reference_system) ? baselines :
                                             solve_coupled_excitations(reference_system, scenario.excitations)
                                for (reference, base_solution, solution) in zip(references, baselines, candidates)
                                    for field in fields
                                        getproperty(solution, field) isa AbstractArray || continue
                                        isempty(getproperty(solution, field)) && continue
                                        @test relative(getproperty(base_solution, field), getproperty(solution, field)) < 1e-9
                                        @test relative(getproperty(reference, field), getproperty(solution, field)) < 1e-9
                                    end
                                    @test solution.fem_interior_residual < 1e-10
                                    @test solution.fem_rhs_condensation_s >= 0 && solution.fem_reconstruction_s >= 0
                                end
                            finally
                                isnothing(reference_system) || release_coupled_system!(reference_system)
                                release_condensed_coupled_system!(baseline)
                                release_condensed_coupled_system!(candidate)
                            end
                        end
                        @test mumps_cache.mumps_store[:solver].analysis_count == 1
                        @test mumps_cache.mumps_store[:solver].initialized
                    finally
                        solver = get(mumps_cache.mumps_store, :solver, nothing)
                        release_condensed_coupled_cache!(umfpack_cache)
                        release_condensed_coupled_cache!(mumps_cache)
                        @test isempty(mumps_cache.mumps_store)
                        @test !isnothing(solver) && !solver.initialized
                    end
                end
            end

            # Single-precision operators with the double-precision dense LU: MUMPS and UMFPACK factor
            # the same promoted values, so they agree far below the Float32 operator error.
            fem_mesh32 = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
            bem_mesh32 = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), Float32(0.001))
            interface_map32 = build_conforming_interface_map(
                fem_mesh32, bem_mesh32, physical_tag(fem_mesh32, 2, "Interface"), 2,
            )
            transducer32 = ElectrodynamicTransducer{Float32}(
                "component:test", [physical_tag(fem_mesh32, 2, "Radiator")], Float32[1], [1], Float32[-1],
                SVector(0f0, 0f0, 1f0), 2f0, 1, 6f0, 0.0005f0, 7f0, 0.015f0, 0.0005f0, 1f0,
            )
            best = (tc, "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1", "BLAB_COUPLED_DENSE_FLOAT64" => "1")
            build32(switch) = withenv(switch, best...) do
                build_condensed_coupled_system(
                    fem_mesh32, bem_mesh32, interface_map32, 700f0, 343f0, 1.21f0;
                    options..., transducers=[transducer32],
                )
            end
            baseline32 = build32(umfpack)
            candidate32 = build32(mumps)
            try
                @test candidate32.condensation.backend == :mumps_seq
                @test candidate32.condensation.mumps_owned
                excitations32 = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF32(1, 0))]
                base_solution = only(solve_condensed_coupled_excitations(baseline32, excitations32))
                solution = only(solve_condensed_coupled_excitations(candidate32, excitations32))
                @test eltype(solution.bem_pressure) == ComplexF32
                for field in fields
                    @test relative(getproperty(base_solution, field), getproperty(solution, field)) < 1e-5
                end
            finally
                release_condensed_coupled_system!(baseline32)
                release_condensed_coupled_system!(candidate32)
            end
            @test !candidate32.condensation.mumps_solver.initialized
        end
    end
end
