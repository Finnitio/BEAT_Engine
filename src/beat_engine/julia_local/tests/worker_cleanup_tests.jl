using Test
include(joinpath(@__DIR__, "..", "src", "BeatEngineWorkerCleanup.jl"))
using .BeatEngineWorkerCleanup

@testset "Opt-in worker cleanup" begin
    legacy = cleanup_options(Dict())
    @test legacy.policy == "aggressive"
    @test cleanup_reason(legacy, 1; free_fraction=0.9) == "aggressive"
    opt = cleanup_options(Dict("bem_backend" => "cuda", "worker_cleanup" => Dict("policy" => "cuda_reuse")))
    @test cleanup_reason(opt, 1; free_fraction=0.9) == "reuse"
    @test cleanup_reason(opt, 7; free_fraction=0.9) == "reuse"
    @test cleanup_reason(opt, 8; free_fraction=0.9) == "interval"
    @test cleanup_reason(opt, 1; free_fraction=0.2) == "memory_pressure"
    @test cleanup_reason(opt, 1; free_fraction=0.1) == "memory_pressure"
    @test cleanup_reason(opt, 1) == "memory_unknown"
    @test cleanup_reason(opt, 1; free_fraction=NaN) == "memory_unknown"
    @test cleanup_reason(opt, 1; cancelled=true, free_fraction=0.9) == "cancelled"
    @test_throws ErrorException cleanup_options(Dict("worker_cleanup" => Dict("policy" => "cuda_reuse")))
    for value in (0, -1, 1.5, true, "8", 1025)
        @test_throws ErrorException cleanup_options(Dict("worker_cleanup" => Dict("max_requests" => value)))
    end
    for value in (0, 1, Inf, NaN, true, "0.2")
        @test_throws ErrorException cleanup_options(Dict("worker_cleanup" => Dict("min_free_fraction" => value)))
    end
    @test_throws ErrorException cleanup_options(Dict("worker_cleanup" => Dict("policy" => "typo")))
    @test_throws ErrorException cleanup_options(Dict("worker_cleanup" => Dict("unexpected" => 1)))
end

include(joinpath(@__DIR__, "worker_completion_tests.jl"))
