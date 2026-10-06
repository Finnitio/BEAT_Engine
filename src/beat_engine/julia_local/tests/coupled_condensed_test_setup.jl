isdefined(@__MODULE__, :BeatEngineCoupledCondensed) ||
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupled
using .BeatEngineCoupledCondensed
using LinearAlgebra, Random, SparseArrays, StaticArrays

const CONDENSED_FIXTURE_ROOT = joinpath(@__DIR__, "fixtures")
const CONDENSED_QUADRATURE_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "1"))
const CONDENSED_SINGULAR_ORDER = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "1"))

# Keep reduced-vs-unreduced comparisons independent of backend defaults. Tests inside this
# scope opt into individual optimizations with nested `withenv` calls.
function with_baseline_cpu_coupled(f)
    return withenv(f,
        "BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "0",
        "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION" => "0",
        "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "0",
        "BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "lu",
        "BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "0",
        "BLAB_COUPLED_INTERFACE_BLOCKS" => "0",
        "BLAB_COUPLED_DEMAND_RECONSTRUCTION" => "0",
        "BLAB_COUPLED_DENSE_FLOAT64" => "0",
        "BLAB_COUPLED_DENSE_REFINEMENT" => "0",
        "BLAB_COUPLED_FEM_FLOAT64" => "0",
        "BLAB_COUPLED_FEM_SOLVER" => "umfpack",
        "BLAB_COUPLED_STAGE_OVERLAP" => "off",
    )
end

function condensed_synthetic_case(
    ::Type{T};
    vertex_count::Int=60,
    retained_count::Int=10,
    density::Float64=0.08,
    interior_load::Bool=false,
) where {T<:AbstractFloat}
    Random.seed!(20260814)
    retained = sort(randperm(vertex_count)[1:retained_count])
    # Diagonally dominant so the interior block is safely invertible and the manufactured
    # solution isolates the condensation algebra rather than conditioning.
    system = SparseMatrixCSC{Complex{T},Int}(
        sprand(Complex{T}, vertex_count, vertex_count, density) +
        Complex{T}(vertex_count / 4) * I,
    )
    load_rows = interior_load ? vcat(retained[2:end], first(setdiff(1:vertex_count, retained))) : retained
    fem_load = sparse(load_rows, 1:retained_count, ones(T, retained_count), vertex_count, retained_count)
    operators = InterfaceOperators(
        fem_load,
        spzeros(T, 4, retained_count),
        spzeros(T, retained_count, vertex_count),
        spzeros(T, retained_count, 4),
    )
    return system, operators, retained
end
