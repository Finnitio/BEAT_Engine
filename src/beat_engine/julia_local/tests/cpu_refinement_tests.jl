using Test, LinearAlgebra
if !isdefined(Main, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore

@testset "opt-in CPU LU refinement" begin
    core = BeatEngineCore
    previous_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
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
        @test unchanged == native
        @test !hasproperty(unchanged_report,:refinement)
        @test core.beat_dense_solve_diagnostics(unchanged_report) == core.beat_dense_solve_diagnostics(native_report)
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
    @test x == exact && r.status === :stagnated
    BLAS.set_num_threads(previous_blas_threads)
end
