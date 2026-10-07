module BeatEngineWorkerCleanup

export cleanup_options, cleanup_reason, finish_aggressive_solve!

"""Request-local opt-in. Absence preserves the historical worker behavior."""
function cleanup_options(options)
    value = get(options, "worker_cleanup", Dict())
    value isa AbstractDict || error("worker_cleanup must be an object")
    all(k in ("policy", "max_requests", "min_free_fraction") for k in keys(value)) ||
        error("Unknown worker_cleanup option")
    policy = get(value, "policy", "aggressive")
    policy in ("aggressive", "cuda_reuse") || error("Unknown worker_cleanup policy")
    interval = get(value, "max_requests", 8)
    interval isa Integer && !(interval isa Bool) && 1 <= interval <= 1024 ||
        error("worker_cleanup.max_requests must be an integer in 1:1024")
    fraction = get(value, "min_free_fraction", 0.2)
    fraction isa Real && !(fraction isa Bool) && isfinite(fraction) && 0 < fraction < 1 ||
        error("worker_cleanup.min_free_fraction must be between 0 and 1")
    policy == "cuda_reuse" && get(options, "bem_backend", "cpu") != "cuda" &&
        error("cuda_reuse cleanup requires bem_backend=cuda")
    return (policy=policy, max_requests=Int(interval), min_free_fraction=Float64(fraction))
end

"""Return the reason for full cleanup, or `reuse` after a successful CUDA job.

The count is successful requests since the last full cleanup, including this job.
Unknown device memory is handled conservatively, not as unlimited headroom.
"""
function cleanup_reason(options, count; cancelled=false, free_fraction=nothing)
    cancelled && return "cancelled"
    options.policy == "aggressive" && return "aggressive"
    count >= options.max_requests && return "interval"
    (free_fraction === nothing || !isfinite(free_fraction)) && return "memory_unknown"
    free_fraction <= options.min_free_fraction && return "memory_pressure"
    return "reuse"
end

"""Emit and flush successful completion before reclaiming on the aggressive path."""
function finish_aggressive_solve!(solved_count, emit, reclaim; output=stdout, error_output=stderr)
    emit(Dict{String,Any}("type" => "completed", "solved_count" => solved_count))
    flush(output)
    try
        reclaim()
    catch exception
        # Completion is already visible to the client; never emit another terminal event.
        println(error_output, "Post-solve reclamation failed: ", sprint(showerror, exception))
        flush(error_output)
    end
    return nothing
end

end
