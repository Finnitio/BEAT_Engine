# Whether a Metal sweep overlaps the next frequency's GPU assembly with this
# frequency's host solve, and how far ahead it assembles.
#
# Per frequency a sequential sweep costs A + S + F: the GPU assembly, the host
# dense solve, and the GPU field evaluation. Overlapped, the GPU assembles the
# next frequency while the host solves this one, so a frequency costs
# F + max(A + c, (1 + kappa) S), where c is the time the assembly loses to
# sharing the machine with the solve and kappa the fraction the solve slows
# down while the GPU streams memory. The field waits behind the next assembly
# either way, so F cancels, and the saving per frequency is
#
#     min(S - c, A - kappa S).
#
# It is positive when both sides have work to hide and the slowdowns do not eat
# it. Both move with the hardware -- a larger GPU shrinks A, faster BLAS shrinks
# S -- which is why the choice is not a dof threshold. S comes from the adaptive
# dense-solve model in BeatEngineDenseSolve.jl, already calibrated per machine;
# A, c and kappa from the constants below. All are environment overrides, and
# scripts/calibrate_metal_sweep_overlap.jl measures them.
#
# The defaults are calibrated on an Apple M1 Pro (16-core GPU, 4 Julia threads,
# 7 BLAS threads), with c and kappa rounded up. There, per frequency through
# the pipeline itself:
#
#   mesh              P1 dofs   A       S LU / GMRES    saving LU / GMRES   model
#   sample.msh          1,390   69 ms    22 / 20 ms      27 / 24 ms         19 / 17
#   sample_detailed     3,502  355 ms   269 / 140 ms    244 / 171 ms       266 / 137
#
# The dof threshold it replaces (1,900, fitted on an M1 Max) kept sample.msh
# sequential, which costs 0.12-0.25 s of a 1.3-1.7 s sweep on this machine.

const METAL_ASSEMBLY_DOF2_SECONDS_ENV = "BLAB_METAL_ASSEMBLY_DOF2_SECONDS"
const METAL_ASSEMBLY_FIXED_SECONDS_ENV = "BLAB_METAL_ASSEMBLY_FIXED_SECONDS"
const METAL_OVERLAP_COST_SECONDS_ENV = "BLAB_METAL_OVERLAP_COST_SECONDS"
const METAL_OVERLAP_HOST_SLOWDOWN_ENV = "BLAB_METAL_OVERLAP_HOST_SLOWDOWN"
const METAL_PIPELINE_ENV = "BLAB_METAL_PIPELINE"
const METAL_PIPELINE_DEPTH_ENV = "BLAB_METAL_PIPELINE_DEPTH"

"""Fused Metal assembly seconds per squared P1 dof, per symmetry copy (M1 Pro)."""
const METAL_ASSEMBLY_DOF2_SECONDS_DEFAULT = 2.77e-8

"""Fused Metal assembly seconds that do not scale with the mesh (M1 Pro)."""
const METAL_ASSEMBLY_FIXED_SECONDS_DEFAULT = 0.016

"""Seconds per frequency the assembly loses to running beside the solve (M1 Pro)."""
const METAL_OVERLAP_COST_SECONDS_DEFAULT = 0.003

"""Fraction the host solve slows while the GPU assembles (M1 Pro)."""
const METAL_OVERLAP_HOST_SLOWDOWN_DEFAULT = 0.1

function _overlap_env_float(name::AbstractString, default::Float64)
    text = strip(get(ENV, name, ""))
    isempty(text) && return default
    value = tryparse(Float64, text)
    value === nothing && error("$name must be a number; got $(repr(text)).")
    value >= 0 || error("$name must not be negative; got $value.")
    return value
end

"""
    metal_fused_assembly_seconds(dof_count, copies=1)

Modelled seconds for one fused Metal Burton-Miller assembly: every element
pair once per symmetry copy (`symmetry_reduction_factor`), plus a fixed part.
"""
function metal_fused_assembly_seconds(dof_count::Integer, copies::Integer=1)
    dof_count <= 0 && return 0.0
    per_dof2 = _overlap_env_float(METAL_ASSEMBLY_DOF2_SECONDS_ENV, METAL_ASSEMBLY_DOF2_SECONDS_DEFAULT)
    fixed = _overlap_env_float(METAL_ASSEMBLY_FIXED_SECONDS_ENV, METAL_ASSEMBLY_FIXED_SECONDS_DEFAULT)
    return max(1, copies) * per_dof2 * float(dof_count)^2 + fixed
end

"""
    sweep_overlap_saving_seconds(assembly_s, solve_s; overlap_cost_s, host_slowdown)

Seconds per frequency that overlapping the assembly with the solve saves, from
their sequential times: `min(solve_s - overlap_cost_s, assembly_s - host_slowdown * solve_s)`.
Negative when the overlap loses.
"""
function sweep_overlap_saving_seconds(
    assembly_s::Real,
    solve_s::Real;
    overlap_cost_s::Real=_overlap_env_float(METAL_OVERLAP_COST_SECONDS_ENV, METAL_OVERLAP_COST_SECONDS_DEFAULT),
    host_slowdown::Real=_overlap_env_float(METAL_OVERLAP_HOST_SLOWDOWN_ENV, METAL_OVERLAP_HOST_SLOWDOWN_DEFAULT),
)
    return min(solve_s - overlap_cost_s, assembly_s - host_slowdown * solve_s)
end

"""
    metal_sweep_overlap_plan(dof_count, drive_count, symmetry_mode; frequency_count, threads, setting)

Whether a Metal sweep overlaps assembly with the solve, and why. `reason` is
`:single_thread` or `:single_frequency` when it cannot help, `:override` when
`BLAB_METAL_PIPELINE` decided (`0` off, anything else on), and `:model` when the
modelled saving did. The modelled times are returned either way, for
diagnostics.
"""
function metal_sweep_overlap_plan(
    dof_count::Integer,
    drive_count::Integer,
    symmetry_mode;
    frequency_count::Integer,
    threads::Integer=Threads.nthreads(),
    setting::AbstractString=get(ENV, METAL_PIPELINE_ENV, ""),
)
    assembly_s = metal_fused_assembly_seconds(dof_count, symmetry_reduction_factor(symmetry_mode))
    solve = beat_dense_solve_plan(dof_count, max(1, drive_count))
    solve_s = solve.method === :gmres ? solve.gmres_model_seconds : solve.lu_model_seconds
    saving_s = sweep_overlap_saving_seconds(assembly_s, solve_s)
    setting = strip(setting)
    enabled, reason = if threads <= 1
        false, :single_thread
    elseif frequency_count <= 1
        false, :single_frequency
    elseif !isempty(setting)
        setting != "0", :override
    else
        saving_s > 0, :model
    end
    return (
        enabled=enabled,
        reason=reason,
        assembly_model_s=assembly_s,
        solve_model_s=solve_s,
        saving_model_s=saving_s,
    )
end

"""
    metal_sweep_assembly_lookahead(dof_count, drive_count, frequency_count, FloatType)

How many frequencies the Metal assembly producer may run ahead of the solve.

Derived from memory, never fixed: one in-flight frequency costs a dense
`dofs x dofs` system plus its right-hand sides, which is 12 MB at 1,200 dofs and
3.2 GB at 20,000, so the same constant cannot be right at both ends.
`BLAB_METAL_PIPELINE_DEPTH` overrides it for measurement.
"""
function metal_sweep_assembly_lookahead(
    dof_count::Integer,
    drive_count::Integer,
    frequency_count::Integer,
    ::Type{T},
) where {T<:AbstractFloat}
    entry_bytes = sizeof(Complex{T})
    system_bytes = entry_bytes * (Int(dof_count)^2 + Int(dof_count) * max(1, Int(drive_count)))
    override = strip(get(ENV, METAL_PIPELINE_DEPTH_ENV, ""))
    if !isempty(override)
        requested = tryparse(Int, override)
        requested === nothing &&
            error("$METAL_PIPELINE_DEPTH_ENV must be a positive integer; got $(repr(override)).")
        return clamp(requested, 1, max(1, Int(frequency_count)))
    end
    return sweep_pipeline_depth(system_bytes, metal_sweep_memory_available(), frequency_count)
end

# Coupled (condensed FEM-BEM) sweeps on Metal pipeline differently from the
# exterior ones above. Within a frequency the host FEM condensation (S) already
# runs beside the GPU assembly of the BEM operators (G) and the host combination
# of those operators into the Burton-Miller blocks (C) that follows it. After
# both, the host assembles, factors and solves the coupled system (L). Per
# frequency:
#
#     sequential   max(G + C + R, S) + L
#     pipelined    max(G + C, max(S, R) + L)
#
# where R is the BEM-stage host work the producer does not take over (motion and
# prescribed-source products, an interface-radiation replay LU), which runs
# before the FEM task is collected and so overlaps it in both schedules. The
# pipeline moves G and C onto a producer that runs one frequency ahead. With
# R = 0 the saving is min(L, G + C - S): zero when the FEM side already binds (a
# large FEM interior behind a small exterior), at most L when the GPU side does. It is the bound docs/Benchmarking.md asks for, with
# the combine counted on the GPU side. All four times are measured in the run
# itself, on the frequencies solved before the pipeline starts, so the decision
# follows the machine and the model instead of a constant. Running the producer
# beside the host stages costs them memory bandwidth -- on an M1 Max about half
# the modelled saving on Multi_region_SAWMOD -- hence the 10% threshold.

const COUPLED_SWEEP_PIPELINE_ENV = "BLAB_COUPLED_SWEEP_PIPELINE"
const COUPLED_SWEEP_PIPELINE_MIN_SAVING_ENV = "BLAB_COUPLED_SWEEP_PIPELINE_MIN_SAVING"

"""Smallest modelled saving, as a fraction of a sequential frequency, that turns the coupled pipeline on."""
const COUPLED_SWEEP_PIPELINE_MIN_SAVING_DEFAULT = 0.10

"""
    coupled_sweep_pipeline_saving_seconds(bem_operator_s, bem_combine_s, fem_task_s, host_rest_s,
                                          retained_s=0) -> (saving_s, sequential_s)

Modelled seconds per frequency that the coupled sweep pipeline saves, and the sequential
per-frequency time it is measured against, from sequential sections: G (`bem_operator_s`), C
(`bem_combine_s`, what the producer takes over), S (`fem_task_s`), L (`host_rest_s`, the host work
after the FEM task is collected) and R (`retained_s`, BEM-stage host work that stays on the host,
before the FEM task is collected): `max(G + C + R, S) + L - max(G + C, max(S, R) + L)`.
"""
function coupled_sweep_pipeline_saving_seconds(
    bem_operator_s::Real,
    bem_combine_s::Real,
    fem_task_s::Real,
    host_rest_s::Real,
    retained_s::Real=0,
)
    G, C, S, L, R = Float64.((bem_operator_s, bem_combine_s, fem_task_s, host_rest_s, retained_s))
    sequential = max(G + C + R, S) + L
    return (saving_s=sequential - max(G + C, max(S, R) + L), sequential_s=sequential)
end

"""
    coupled_sweep_pipeline_plan(timings; remaining_frequencies, bem_backend, in_flight_bytes,
                                available_bytes, threads, setting, min_saving)

Whether a coupled sweep should assemble the remaining frequencies' BEM operators
one frequency ahead, and why. `timings` holds mean `bem_operator_s`,
`bem_combine_s`, `fem_task_s`, `host_rest_s` and optionally `retained_s`: the
driver passes medians over the sequentially solved frequencies after the first. `reason` is `:single_thread`, `:single_frequency`, `:override`
(`BLAB_COUPLED_SWEEP_PIPELINE=on|off`), `:backend` (`auto` pipelines Metal only),
`:memory` (two more operator sets would not fit in half the free working set),
or `:model`. The modelled saving is returned either way, for diagnostics.
"""
function coupled_sweep_pipeline_plan(
    timings;
    remaining_frequencies::Integer,
    bem_backend::Symbol,
    in_flight_bytes::Integer,
    available_bytes::Union{Nothing,Integer}=nothing,
    threads::Integer=Threads.nthreads(),
    setting::AbstractString=get(ENV, COUPLED_SWEEP_PIPELINE_ENV, "auto"),
    min_saving::Real=_overlap_env_float(COUPLED_SWEEP_PIPELINE_MIN_SAVING_ENV, COUPLED_SWEEP_PIPELINE_MIN_SAVING_DEFAULT),
)
    model = coupled_sweep_pipeline_saving_seconds(
        timings.bem_operator_s, timings.bem_combine_s, timings.fem_task_s, timings.host_rest_s,
        hasproperty(timings, :retained_s) ? timings.retained_s : 0.0,
    )
    setting = lowercase(strip(setting))
    setting in ("auto", "on", "off") || error(
        "Unsupported $COUPLED_SWEEP_PIPELINE_ENV value: $(repr(setting)). Expected auto, on, or off.",
    )
    enabled, reason = if threads <= 1
        false, :single_thread
    elseif remaining_frequencies < 2
        false, :single_frequency
    elseif setting != "auto"
        setting == "on", :override
    elseif bem_backend != :metal
        false, :backend
    elseif available_bytes !== nothing && 0.5 * available_bytes < 2 * in_flight_bytes
        false, :memory
    else
        model.saving_s > min_saving * model.sequential_s, :model
    end
    return (
        enabled=enabled,
        reason=reason,
        saving_model_s=model.saving_s,
        sequential_model_s=model.sequential_s,
    )
end
