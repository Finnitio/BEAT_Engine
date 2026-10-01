using Test
include(joinpath(@__DIR__, "..", "exterior_rhs_policy.jl"))
@testset "Exterior CUDA right-hand-side policy" begin
    gib = 1024^3
    matrix, operator = 4gib, 8gib
    @test exterior_use_rhs_operator("auto", 60, matrix, operator, 32gib)
    @test exterior_use_rhs_operator("auto", 2, matrix, operator, 32gib)
    # One excitation has no repeated integration to save.
    @test !exterior_use_rhs_operator("auto", 1, matrix, operator, 32gib)
    @test !exterior_use_rhs_operator("auto", 0, matrix, operator, 32gib)
    @test !exterior_use_rhs_operator("matrix_free", 60, matrix, operator, 32gib)
    @test exterior_use_rhs_operator("cached_operator", 1, matrix, operator, 32gib)
    @test exterior_use_rhs_operator(" Cached_Operator ", 2, matrix, operator, 32gib)
    # 4 + 8 GiB + 512 MiB = 12.5 GiB.
    @test exterior_use_rhs_operator("auto", 60, matrix, operator, 13gib)
    @test !exterior_use_rhs_operator("auto", 60, matrix, operator, 12gib)
    @test !exterior_use_rhs_operator("cached_operator", 60, matrix, operator, 12gib)
    @test !exterior_use_rhs_operator("auto", 60, matrix, 0, 32gib)
    @test_throws ErrorException exterior_use_rhs_operator("fast", 60, matrix, operator, 32gib)
    @test exterior_rhs_mode("MATRIX_FREE") == "matrix_free"
end
