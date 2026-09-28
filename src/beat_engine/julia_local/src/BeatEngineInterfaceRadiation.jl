"""Exterior radiation contributions with the operating coupled flux held fixed."""
module BeatEngineInterfaceRadiation
using LinearAlgebra
export interface_radiation_traces

"""
Recover the complete scattered pressure driven by each interface's spatial flux.
Masking the solved boundary pressure would omit scattering and is not a source
decomposition. Ranges partition the interface flux DOFs. The optional remainder
groups all directly radiating exterior sources, with their original motion.
"""
function interface_radiation_traces(system, solution, ranges; include_other::Bool=false)
    replay = system.interface_radiation_replay
    isnothing(replay) && error("Interface radiation replay was not requested during assembly.")
    T = eltype(solution.bem_pressure)
    rhs = zeros(T, length(solution.bem_pressure), length(ranges))
    flux = zeros(T, length(solution.bem_neumann), length(ranges))
    for (index, indices) in enumerate(ranges)
        values = view(solution.interface_flux, indices)
        rhs[:, index] = -(view(replay.interface_block, :, indices) * values)
        flux[:, index] = system.interface_operators.bem_flux[:, indices] * values
    end
    pressure = replay.factorization \ rhs
    if include_other
        pressure = hcat(pressure, solution.bem_pressure - vec(sum(pressure; dims=2)))
        flux = hcat(flux, solution.bem_neumann - vec(sum(flux; dims=2)))
    end
    scale = max(norm(solution.bem_pressure), norm(pressure), eps(real(T)))
    reconstruction_error = norm(vec(sum(pressure; dims=2)) - solution.bem_pressure) / scale
    return (pressure=pressure, normal_derivative=flux, reconstruction_error=reconstruction_error)
end
end
