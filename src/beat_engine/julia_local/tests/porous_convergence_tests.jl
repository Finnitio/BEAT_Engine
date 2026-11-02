# Slow, opt-in: runtests.jl includes it only with BEAT_RUN_POROUS_CONVERGENCE=1; running the
# file directly always runs it. Checks the documented porous mesh-resolution
# thresholds (beat_contract/porous-mesh-resolution.json, "Mesh resolution" in
# docs/Porous Materials.md) against the filled-tube convergence study. Regenerate the table
# with scripts/porous_convergence_study.jl.
module PorousConvergenceTests
using Test, JSON
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))
include(joinpath(@__DIR__, "porous_tube_helpers.jl"))

const RESOLUTION = JSON.parsefile(joinpath(ENGINE_DIR, "..", "beat_contract", "porous-mesh-resolution.json"))

@testset "porous mesh-resolution threshold is the tightest that meets the target error" begin
    target = RESOLUTION["target_relative_impedance_error"]
    for (order, threshold) in RESOLUTION["min_elements_per_wavelength"]
        order = parse(Int, order)
        worst = Dict(name => porous_study_worst_error(material, order, threshold)
                     for (name, material) in POROUS_STUDY_MATERIALS)
        coarser = Dict(name => porous_study_worst_error(material, order, threshold - 1)
                       for (name, material) in POROUS_STUDY_MATERIALS)
        @info "P$order threshold $threshold" worst coarser
        @test all(<=(target), values(worst))
        @test any(>(target), values(coarser))
    end
end
end # module
