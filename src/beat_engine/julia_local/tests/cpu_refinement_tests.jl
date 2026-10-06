using Test, LinearAlgebra, JSON
if !isdefined(Main, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore

@testset "opt-in CPU LU refinement" begin
    core = BeatEngineCore
    previous_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        n = 24
        v = ComplexF64[complex(cos(j*.3),sin(j*.3)) for j in 1:n]
        v ./= norm(v)
        A = ComplexF32.(Matrix{ComplexF64}(I,n,n) - (1-2e-5)*(v*v'))
        b = ComplexF32[complex(sin(i*.2+j),cos(i*.4-j)) for i in 1:n,j in 1:2]
        b = hcat(b,zeros(ComplexF32,n))
        saved_A,saved_b = copy(A),copy(b)
        reference = ComplexF64.(A) \ ComplexF64.(b)
        initial = A \ b
        refined,report = core.beat_refine_cpu_lu(A,b)
        @test eltype(refined) === ComplexF32
        @test size(refined) == size(b)
        @test A == saved_A && b == saved_b
        @test norm(refined-reference)/norm(reference) < 2e-6
        @test norm(refined-reference) < norm(initial-reference)/20
        @test refined[:,3] == zeros(ComplexF32,n)
        @test 0 <= report.steps <= 3
        independently_returned = [norm(ComplexF64.(b[:,j]) - ComplexF64.(A)*ComplexF64.(refined[:,j])) /
            norm(ComplexF64.(b[:,j])) for j in 1:2]
        @test report.returned_relative_residuals[1:2] ≈ independently_returned
        @test report.returned_relative_residuals[3] == 0
        @test report.history[end] == report.working_relative_residuals
        @test all(all(next .<= previous) for (previous,next) in zip(report.history,report.history[2:end]))
        for j in 1:2
            separate,_ = core.beat_refine_cpu_lu(A,b[:,j])
            @test separate ≈ refined[:,j] rtol=2e-6
        end
        # Reuse single-precision factors against a genuinely more precise operator.
        A64 = ComplexF64.(A) + 1e-9*(v*v')
        xtrue,rtrue = core.beat_refine_cpu_lu(A,b;residual_matrix=A64)
        @test norm(xtrue-A64\ComplexF64.(b))/norm(reference) < 2e-6
        @test all(isfinite,rtrue.returned_relative_residuals)

        system = (;matrix=A,rhs=b)
        withenv("BLAB_BEAT_CPU_LU_REFINEMENT_STEPS"=>nothing) do
            native,native_report = core.beat_solve_dense_system(A,b;method=:lu)
            unchanged,unchanged_report = core.solve_burton_miller_neumann_system_cpu_with_report(system;method=:lu)
            @test reinterpret(UInt8,vec(unchanged)) == reinterpret(UInt8,vec(native))
            @test !hasproperty(unchanged_report,:refinement)
            @test core.beat_dense_solve_diagnostics(unchanged_report) == core.beat_dense_solve_diagnostics(native_report)
        end
        # Both public wrappers keep the native n×drives contract for vectors too.
        for rhs in (b[:,1], b[:,1:1], b)
            shape_system = (;matrix=A,rhs)
            native,_ = core.beat_solve_dense_system(A,rhs;method=:lu)
            disabled,disabled_report = core.solve_burton_miller_neumann_system_cpu_with_report(
                shape_system;method=:lu,refinement_steps=0)
            enabled,_ = core.solve_burton_miller_neumann_system_cpu_with_report(
                shape_system;method=:lu,refinement_steps=3)
            @test size(enabled) == size(disabled) == size(native)
            @test reinterpret(UInt8,vec(disabled)) == reinterpret(UInt8,vec(native))
            @test !hasproperty(disabled_report,:refinement)
            @test isempty(core.beat_dense_refinement_diagnostics(disabled_report))
            @test size(core.solve_burton_miller_neumann_system_cpu(
                shape_system;method=:lu,refinement_steps=3)) == size(native)
        end
        withenv("BLAB_BEAT_CPU_LU_REFINEMENT_STEPS"=>"3") do
            actual,actual_report = core.solve_burton_miller_neumann_system_cpu_with_report(system;method=:lu)
            @test actual == refined
            @test hasproperty(actual_report,:refinement)
            @test occursin("Float64 residual refinement",core.describe_dense_solve(actual_report))
            @test core.beat_dense_solve_diagnostics(actual_report)["dense_solve_refinement_operator"] == "rounded_float32"
            @test_throws ArgumentError core.solve_burton_miller_neumann_system_cpu_with_report(system;method=:gmres)
            double_system = (;matrix=ComplexF64.(A),rhs=ComplexF64.(b))
            double_native,_ = core.beat_solve_dense_system(double_system.matrix,double_system.rhs;method=:lu)
            double_actual,double_report = core.solve_burton_miller_neumann_system_cpu_with_report(double_system;method=:lu)
            @test double_actual == double_native && !hasproperty(double_report,:refinement)
        end
        for value in ("bad","-1","4")
            withenv("BLAB_BEAT_CPU_LU_REFINEMENT_STEPS"=>value) do
                @test_throws ArgumentError core.beat_cpu_lu_refinement_steps()
            end
        end
        for steps in (0,4)
            @test_throws ArgumentError core.beat_refine_cpu_lu(A,b;max_steps=steps)
        end
        @test_throws ArgumentError core.beat_refine_cpu_lu(A,b;rtol=NaN)
        @test_throws DimensionMismatch core.beat_refine_cpu_lu(A,b[1:end-1,:])
        @test_throws DimensionMismatch core.beat_refine_cpu_lu(A,b;residual_matrix=A[1:end-1,:])
        identity = Matrix{ComplexF32}(I,3,3)
        exact = ComplexF32[1+2im,3-4im,0]
        x,r = core.beat_refine_cpu_lu(identity,exact)
        @test x == exact && r.steps == 0 && r.status === :converged
        x,r = core.beat_refine_cpu_lu(identity,exact;residual_matrix=zeros(ComplexF64,3,3))
        @test x == exact && r.status === :stagnated && r.steps == 1
        @test r.history == [ones(1),ones(1)]
        x,r = core.beat_refine_cpu_lu(identity,zeros(ComplexF32,3,2))
        @test iszero(x) && r.steps == 0 && r.status === :converged
        @test r.history == [zeros(2)]
        @test r.working_relative_residuals == r.returned_relative_residuals == zeros(2)

        # The first column's corrections worsen its residual and must be rejected,
        # while the second column improves each step using the same LU factors.
        I2 = Matrix{ComplexF32}(I,2,2)
        residual_operator = ComplexF64[-1 0; 0 0.5]
        x,r = core.beat_refine_cpu_lu(I2,I2;residual_matrix=residual_operator)
        @test x == ComplexF32[1 0; 0 1.875]
        @test r.status === r.working_status === :step_limit
        @test r.steps == 3
        @test r.history == [[2.,0.5],[2.,0.25],[2.,0.125],[2.,0.0625]]
        @test r.working_relative_residuals == r.returned_relative_residuals == [2.,0.0625]
        @test r.returned_relative_residuals ≈ [norm(I2[:,j]-residual_operator*x[:,j]) for j in 1:2]
        for j in 1:2
            separate,_ = core.beat_refine_cpu_lu(I2,I2[:,j];residual_matrix=residual_operator)
            @test separate == x[:,j]
        end

        # Working convergence on the last correction does not imply convergence
        # after rounding the pressure back to the public ComplexF32 type.
        triangular = ComplexF32[1 3; 0 7]
        ones_rhs = ones(ComplexF32,2)
        x,r = core.beat_refine_cpu_lu(triangular,ones_rhs;max_steps=1)
        @test r.steps == 1 && r.working_status === :converged
        @test r.status === :rounding_limited && r.rtol == 1e-10
        @test only(r.working_relative_residuals) <= r.rtol < only(r.returned_relative_residuals)
        @test only(r.returned_relative_residuals) ≈ norm(ComplexF64.(ones_rhs)-ComplexF64.(triangular)*x)/sqrt(2)
        _,loose = core.beat_refine_cpu_lu(triangular,ones_rhs;max_steps=1,rtol=1e-6)
        @test loose.status === :converged && loose.steps == 0

        # Overflow in an initial Float32 solve must still produce JSON-safe
        # diagnostics (JSON rejects NaN and Inf by default).
        tiny = reshape(ComplexF32[floatmin(Float32)],1,1)
        huge_rhs = ComplexF32[floatmax(Float32)]
        _,exceptional = core.solve_burton_miller_neumann_system_cpu_with_report(
            (;matrix=tiny,rhs=huge_rhs);method=:lu,refinement_steps=1)
        @test exceptional.refinement.status === :nonfinite
        diagnostics = JSON.parse(JSON.json(core.beat_dense_solve_diagnostics(exceptional)))
        @test diagnostics["dense_solve_refinement_working_residuals"] == [nothing]
        @test diagnostics["dense_solve_refinement_returned_residuals"] == [nothing]
        @test diagnostics["dense_solve_refinement_tolerance"] == 1e-10
    finally
        BLAS.set_num_threads(previous_blas_threads)
    end
end

include(joinpath(@__DIR__, "cpu_refinement_driver_tests.jl"))
