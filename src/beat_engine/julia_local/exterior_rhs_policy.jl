# Right-hand-side assembly for exterior CUDA solves with several excitations.
#
# `matrix_free` integrates each excitation's right-hand side over every element
# pair: excitation 1 rides along the system assembly, and each further one
# repeats the regular, image and singular quadrature. `cached_operator` runs the
# system assembly once with the DP0-to-P1 right-hand-side mapping as its 2-D
# right-hand side and forms every excitation with one GEMM, so it needs one pair
# integration where `matrix_free` needs one per excitation. From two
# excitations on it replaces at least one full integration with a GEMM that is
# negligible beside it; that is a pass count, not a machine constant, so `auto`
# decides on the excitation count and memory alone.
#
# BEAT_EXTERIOR_RHS_MODE forces either path for comparisons: `matrix_free`, or
# `cached_operator` (still subject to the memory check).
const EXTERIOR_RHS_MODES = ("auto", "matrix_free", "cached_operator")

function exterior_rhs_mode(value)
    mode = lowercase(strip(String(value)))
    mode in EXTERIOR_RHS_MODES || error(
        "Unknown exterior right-hand-side mode: $(value). Expected auto, matrix_free or cached_operator.",
    )
    return mode
end

# The system matrix and the mapping are resident together until the mapping has
# been applied; the mapping is released before the factorization. The other
# transients of that pass (correction blocks, the right-hand sides, cuBLAS
# workspace) are ~0.15 GiB on a 44.7k-face mesh, so 512 MiB covers them. An
# estimate that is still too optimistic costs one failed allocation: the
# assembly then falls back to integrating each excitation.
const EXTERIOR_RHS_MEMORY_MARGIN_BYTES = 512 * 1024^2

function exterior_rhs_memory_fits(matrix_bytes, operator_bytes, available_bytes)
    operator_bytes > 0 && matrix_bytes >= 0 &&
        matrix_bytes + operator_bytes + EXTERIOR_RHS_MEMORY_MARGIN_BYTES <= available_bytes
end

function exterior_use_rhs_operator(mode, excitation_count, matrix_bytes, operator_bytes, available_bytes)
    mode = exterior_rhs_mode(mode)
    mode == "matrix_free" && return false
    excitation_count >= 1 || return false
    mode == "auto" && excitation_count < 2 && return false
    return exterior_rhs_memory_fits(matrix_bytes, operator_bytes, available_bytes)
end
