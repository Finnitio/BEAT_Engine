# Device-independent policy and projection. Included by the condensed solver;
# also usable on their own for CPU-only checks without loading Metal or FEM.
function resolve_metal_coupled_bem_assembly(requested, unsupported_reason=nothing)
    mode = Symbol(lowercase(strip(String(requested))))
    mode in (:auto, :combined, :operators) || error(
        "BLAB_METAL_COUPLED_BEM_ASSEMBLY must be auto, combined, or operators; got $requested.")
    mode == :operators && return (mode=:operators, fallback_reason=nothing)
    if unsupported_reason !== nothing
        mode == :combined && error("Combined Metal coupled assembly unavailable: $unsupported_reason")
        return (mode=:operators, fallback_reason=String(unsupported_reason))
    end
    return (mode=:combined, fallback_reason=nothing)
end

function project_metal_coupled_host_blocks(combined, bem_flux, motion_flux, prescribed_neumann)
    T = real(eltype(combined.a))
    size(combined.c, 2) == size(bem_flux, 1) == size(motion_flux, 1) == size(prescribed_neumann, 1) ||
        error("Metal combined flux maps must have one row per DP0 dof.")
    return (
        # A must outlive the device buffers. Projection outputs are owned host
        # arrays too; Q remains sparse and no N x F host combine is needed.
        bem_lhs=copy(combined.a),
        bem_interface_block=combined.c * Complex{T}.(bem_flux),
        bem_motion_block=size(motion_flux, 2) == 0 ? nothing : combined.c * motion_flux,
        bem_prescribed_rhs=-(combined.c * prescribed_neumann),
    )
end
