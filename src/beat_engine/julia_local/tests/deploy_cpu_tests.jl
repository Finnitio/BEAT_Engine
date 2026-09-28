using Test, LinearAlgebra
include(joinpath(@__DIR__, "..", "deploy_cpu.jl"))

@testset "Deploy CPU GMRES complex solve and warm start" begin
    for T in (Float32, Float64)
        A = Complex{T}[4+im 1-im 0; 0.5im 3-im 1; 1 0 2+im]
        b = Complex{T}[1+im, 2-im, 3]
        tolerance = T == Float32 ? 1f-5 : 1e-12
        exact = A \ b
        solution, iterations, residual, history, applications, initial = deploy_cpu_gmres(
            x -> A*x, b; tolerance=tolerance, max_iterations=3,
        )
        @test isapprox(solution, exact; rtol=tolerance)
        @test norm(A*solution-b)/norm(b) < tolerance
        @test residual < tolerance
        @test applications == iterations == length(history)
        warm = deploy_cpu_gmres(x -> A*x, b; tolerance=tolerance, max_iterations=3, initial_guess=2exact)
        @test isapprox(warm[1], exact; rtol=tolerance)
        @test warm[2] == 0
        @test warm[5] == 1
        cold = deploy_cpu_gmres(x -> A*x, zero(b); tolerance=tolerance, max_iterations=3)
        @test iszero(cold[1])
        @test cold[2] == 0
        limited = deploy_cpu_gmres(x -> A*x, b; tolerance=tolerance, max_iterations=1)
        @test limited[2] == 1
        @test limited[3] > tolerance
        @test_throws ErrorException deploy_cpu_gmres(x -> A*x, b; tolerance=tolerance, max_iterations=0)
    end
end
