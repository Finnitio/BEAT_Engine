"""
Filled-tube convergence study for porous regions: worst relative input-impedance error
over elements per porous wavelength (mean tetrahedron edge, wavelength 2 pi / Re(k_eq) at
f_max) for P1 and P2 tetrahedra, soft foam and glass wool. Prints the Markdown table and
the derived thresholds for "Mesh resolution" in docs/Porous Materials.md.

    julia --threads=2 --project=src/beat_engine/julia_local \\
        src/beat_engine/julia_local/scripts/porous_convergence_study.jl

beat_contract/porous-mesh-resolution.json records the thresholds of its target error;
tests/porous_convergence_tests.jl checks that they still hold and are the tightest.
"""

const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))
include(joinpath(ENGINE_DIR, "tests", "porous_tube_helpers.jl"))
using Printf

const ELEMENTS_PER_WAVELENGTH = Dict(1 => [4, 6, 8, 9, 10, 11, 12, 14, 16, 20, 24], 2 => [2, 3, 4, 5, 6, 8])
const TARGETS = (0.01, 0.02, 0.05)

# Top-level loops: inferring one function around the whole sweep stalls the compiler.
errors = Dict{Tuple{Int,Int},Vector{Float64}}()
for order in (1, 2), n in ELEMENTS_PER_WAVELENGTH[order]
    errors[(order, n)] = [porous_study_worst_error(material, order, n) for material in last.(POROUS_STUDY_MATERIALS)]
end
println("| Order | N | ", join(first.(POROUS_STUDY_MATERIALS), " | "), " |")
println("|---|---|", repeat("---|", length(POROUS_STUDY_MATERIALS)))
for order in (1, 2), n in ELEMENTS_PER_WAVELENGTH[order]
    println("| P$order | $n | ", join([@sprintf("%.1e", e) for e in errors[(order, n)]], " | "), " |")
end
function threshold(order, target)
    index = findfirst(n -> maximum(errors[(order, n)]) <= target, ELEMENTS_PER_WAVELENGTH[order])
    return isnothing(index) ? "> $(last(ELEMENTS_PER_WAVELENGTH[order]))" : string(ELEMENTS_PER_WAVELENGTH[order][index])
end
println("\n| Target error | P1 | P2 |\n|---|---|---|")
for target in TARGETS
    println("| $(round(Int, 100target)) % | ", threshold(1, target), " | ", threshold(2, target), " |")
end
