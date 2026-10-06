# Combined coupled assembly. Reuse the exterior pair arithmetic and gather
# tables, retaining the flux coefficient as a matrix instead of applying a
# known q. No exterior launch or solver path is changed.
# Pair buffers hold A and -C (24 floats); final row scaling changes -C to C.

function _metal_coupled_flux_gather_kernel!(
    rhs_operator,
    blocks,
    elements,
    element_positions,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    element_dp0_dofs,
    element_count::Int32,
    chunk_start::Int32,
    chunk_count::Int32,
    pair_stride::Int32,
    p1_count::Int32,
)
    index = Int32(thread_position_in_grid_1d())
    index > p1_count * chunk_count && return nothing
    row = (index - Int32(1)) % p1_count + Int32(1)
    trial_local = (index - Int32(1)) ÷ p1_count + Int32(1)
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    column_base = element_count * (trial_local - Int32(1))
    value_re = zero(eltype(blocks))
    value_im = zero(eltype(blocks))
    @inbounds incident_position = Int32(vertex_offsets[row])
    @inbounds incident_stop = Int32(vertex_offsets[row + Int32(1)]) - Int32(1)
    while incident_position <= incident_stop
        @inbounds test_position = Int32(element_positions[incident_elements[incident_position]])
        @inbounds local_row = Int32(incident_local_indices[incident_position])
        pair = test_position + column_base
        @inbounds value_re += blocks[pair + (local_row + Int32(17)) * pair_stride]
        @inbounds value_im += blocks[pair + (local_row + Int32(20)) * pair_stride]
        incident_position += Int32(1)
    end
    coefficient = Complex(value_re, value_im)
    @inbounds dp0_column = Int32(element_dp0_dofs[trial_index])
    @inbounds rhs_operator[row + (dp0_column - Int32(1)) * p1_count] += coefficient
    return nothing
end

function _launch_metal_coupled_pair_kernels!(
    lhs,
    rhs_operator,
    blocks,
    cache::MetalRegularAssemblyCache,
    k,
    pair_offsets,
    singular_trial_indices,
    skip_mode,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    element_count = length(cache.element_indices)
    element_count == 0 && return nothing
    rule_count = cache.rule_count
    rule_count in (1, 3, 6) || error("Fused Metal assembly expects a 1-, 3-, or 6-point triangle rule; got $(rule_count).")
    tables = _metal_fused_gather_tables(cache)
    chunk_size = tables.chunk_size
    pair_stride = Int32(element_count * chunk_size)
    tile_x, tile_y = _metal_atomic_tile()
    groupsize = _metal_kernel_groupsize()
    p1_count = Int32(cache.p1_dof_count)
    timed = get(ENV, "BLAB_METAL_GATHER_TIMING", "0") == "1"
    packed = _metal_packed_pair_tables_for(cache)
    timed && Metal.synchronize()
    stamp = time()
    for chunk in 1:tables.chunk_count
        chunk_start = (chunk - 1) * chunk_size + 1
        chunk_count = min(chunk_size, element_count - chunk_start + 1)
        # One transform per launch and gather, so each pair block is overwritten (no accumulation).
        Metal.@metal threads=(tile_x, tile_y) groups=(cld(element_count, tile_x), cld(chunk_count, tile_y)) _metal_fused_pair_blocks_kernel!(
            blocks, packed.points4, packed.normals4, cache.areas, packed.curls4, cache.faces, tables.elements,
            cache.rule_points, cache.rule_weights,
            Int32(element_count), Int32(chunk_start), Int32(chunk_count), pair_stride,
            k, inv(k), Int32(cache.face_count), Val(packed.rule), Val(rule_count),
            pair_offsets, singular_trial_indices, skip_mode,
            trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
            Val(false),
        )
        stamp = _metal_gather_stage!("fused_pairs", timed, stamp)
        _metal_launch(
            _metal_coupled_flux_gather_kernel!,
            cache.p1_dof_count * chunk_count,
            rhs_operator,
            blocks,
            tables.elements,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            cache.element_dp0_dofs,
            Int32(element_count),
            Int32(chunk_start),
            Int32(chunk_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("fused_rhs", timed, stamp)
        node_start = tables.chunk_node_offsets[chunk]
        node_count = tables.chunk_node_offsets[chunk + 1] - node_start
        _metal_launch(
            _metal_fused_lhs_gather_kernel!,
            cache.p1_dof_count * node_count,
            lhs,
            blocks,
            tables.element_positions,
            cache.vertex_offsets,
            cache.incident_elements,
            cache.incident_local_indices,
            tables.chunk_nodes,
            tables.inc_offsets,
            tables.inc_packed,
            Int32(node_start),
            Int32(node_count),
            Int32(element_count),
            pair_stride,
            p1_count;
            groupsize=groupsize,
        )
        stamp = _metal_gather_stage!("fused_lhs", timed, stamp)
    end
    return nothing
end

function _launch_metal_coupled_singular_kernels!(
    lhs,
    rhs_operator,
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    k,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1),
)
    pair_count = singular_cache.pair_count
    pair_count == 0 && return nothing
    T = typeof(k)
    sx = T(transform.signs[1])
    sy = T(transform.signs[2])
    sz = T(transform.signs[3])
    csx = T(transform.determinant * transform.signs[1])
    csy = T(transform.determinant * transform.signs[2])
    csz = T(transform.determinant * transform.signs[3])
    part_count = _metal_singular_part_count()
    # Same maps the four-operator path uses: the fused left-hand side lands on
    # the same P1-row/P1-column cells as the double layer and hypersingular.
    gather_tables = lock(() -> _metal_singular_gather_tables(regular_cache, singular_cache, part_count), _metal_packed_cache_lock)
    value_count = pair_count * part_count
    lhs_values = rhs_values = nothing
    try
        lhs_values = Metal.zeros(eltype(lhs), value_count, 9)
        rhs_values = Metal.zeros(eltype(lhs), value_count, 3)
        # The exterior fused path's packed singular blocks, grouped by rule point count.
        tables = _metal_fused_singular_tables_for(regular_cache, singular_cache)
        packed = _metal_packed_pair_tables_for(regular_cache)
        for (point_count, positions) in tables.groups
            group_count = length(positions)
            _metal_launch(
                _metal_fused_singular_packed_kernel!, group_count * part_count,
                lhs_values, rhs_values, positions,
                singular_cache.test_indices, singular_cache.trial_indices, singular_cache.rule_indices,
                singular_cache.jac_scales, singular_cache.normal_products, singular_cache.rule_offsets,
                tables.rule_points4, singular_cache.rule_weights, tables.vertices4, packed.normals4, packed.curls4,
                k, inv(k), Int32(group_count), Int32(pair_count),
                sx, sy, sz, csx, csy, csz,
                Val(point_count), Val(part_count),
            )
        end
        for (destination, values, block_map) in (
            (lhs, lhs_values, gather_tables.p1_p1),
            (rhs_operator, rhs_values, gather_tables.p1_dp0),
        )
            _metal_launch(
                _metal_singular_entry_gather_kernel!,
                block_map.entry_count,
                destination, values,
                block_map.entry_indices, block_map.contrib_offsets, block_map.contrib_values,
                block_map.entry_count, pair_count, part_count,
            )
        end
        Metal.synchronize()
    finally
        Metal.synchronize()
        lhs_values === nothing || Metal.unsafe_free!(lhs_values)
        rhs_values === nothing || Metal.unsafe_free!(rhs_values)
    end
    return nothing
end

# Unsupported diagnostic configurations keep their original operator kernels.
function metal_coupled_combined_support_reason(prepared, ::Type{T}) where {T}
    T === Float32 || return "combined Metal assembly requires Float32"
    _normalized_metal_assembly_mode(nothing) == :native || return "host-staged assembly requested"
    _normalized_metal_regular_kernel_mode() == :pair_gather || return "reference regular kernel mode requested"
    _normalized_metal_singular_mode() == :native || return "host singular mode requested"
    _normalized_metal_singular_writeback() == :gather || return "singular scatter write-back requested"
    cache = prepared.device_cache
    cache isa MetalRegularAssemblyCache || return "no native Metal regular cache"
    cache.rule_count in (1, 3, 6) || return "combined kernels support only 1-, 3-, or 6-point rules"
    prepared.device_singular_cache isa MetalSingularCorrectionCache ||
        return "no native Metal singular cache"
    prepared.symmetry_mode in (:off, :x, :xy) || return "unsupported coupled symmetry mode"
    cache.symmetry_mode == prepared.symmetry_mode ||
        return "regular cache symmetry does not match the requested symmetry"
    return nothing
end

function build_metal_coupled_identity_cache(prepared, ::Type{T}) where {T}
    pp = build_metal_sparse_scatter_cache(sparse(Complex{T}.(prepared.identity_p1_p1)))
    try
        pq = build_metal_sparse_scatter_cache(sparse(Complex{T}.(prepared.identity_p1_dp0)))
        return (p1_p1=pp, p1_dp0=pq)
    catch
        release_metal_sparse_scatter_cache!(pp)
        rethrow()
    end
end

function release_metal_coupled_identity_cache!(cache)
    release_metal_sparse_scatter_cache!(cache.p1_p1)
    release_metal_sparse_scatter_cache!(cache.p1_dp0)
    return nothing
end

"""
    assemble_coupled_burton_miller_metal(mesh, prepared, k; identity_cache=nothing)

Retain A (P1 x P1) and C (P1 x DP0) using the exterior fused pair arithmetic.
The signed wavenumber, image normals/curls, direct and image Duffy rules, and
symmetry weights follow the existing exterior path. Images use separate launches.
Returned device storage belongs to the caller, including on private storage.
"""
function assemble_coupled_burton_miller_metal(
    mesh::BoundaryMesh{T}, prepared, k::T; identity_cache=nothing,
) where {T<:AbstractFloat}
    _require_metal!()
    reason = metal_coupled_combined_support_reason(prepared, T)
    reason === nothing || error("Combined Metal coupled assembly unavailable: $reason")
    cache = prepared.device_cache
    cache.symmetry_mode == prepared.symmetry_mode || error("Metal coupled cache symmetry mismatch.")
    signed_k = outgoing_wavenumber(k)
    isfinite(signed_k) && !iszero(signed_k) || error("Combined Metal assembly needs finite nonzero k.")
    owns_identity = identity_cache === nothing
    owns_identity && (identity_cache = build_metal_coupled_identity_cache(prepared, T))
    a = c = blocks = nothing
    succeeded = false
    try
        storage = metal_operator_storage_mode()
        # Pair buffer for one gather chunk (24 floats per pair); the gather tables no longer own it.
        if !isempty(cache.element_indices)
            tables = _metal_fused_gather_tables(cache)
            blocks = MtlArray{Float32}(undef, _METAL_FUSED_COMPONENTS * length(cache.element_indices) * tables.chunk_size)
        end
        a = Metal.zeros(Complex{T}, prepared.p1.global_dof_count, prepared.p1.global_dof_count; storage=storage)
        c = Metal.zeros(Complex{T}, prepared.p1.global_dof_count, prepared.dp0.global_dof_count; storage=storage)
        empty!(_metal_gather_stage_timing)
        _launch_metal_coupled_pair_kernels!(
            a, c, blocks, cache, signed_k, cache.vertex_offsets, cache.incident_elements, Int32(0),
            one(T), one(T), one(T), one(T), one(T), one(T),
        )
        for (transform, image_cache) in zip(cache.image_transforms, cache.image_singular_caches)
            _launch_metal_coupled_pair_kernels!(
                a, c, blocks, cache, signed_k, image_cache.pair_offsets, image_cache.trial_indices, Int32(1),
                T(transform.signs[1]), T(transform.signs[2]), T(transform.signs[3]),
                T(transform.determinant * transform.signs[1]),
                T(transform.determinant * transform.signs[2]),
                T(transform.determinant * transform.signs[3]),
            )
        end
        # Singular compact blocks carry A and -C too; reuse the P1/DP0 map,
        # with one owner per cell instead of the exterior known-flux row map.
        _launch_metal_coupled_singular_kernels!(a, c, cache, prepared.device_singular_cache, signed_k)
        for (transform, image_cache) in zip(cache.image_transforms, cache.image_singular_caches)
            _launch_metal_coupled_singular_kernels!(a, c, cache, image_cache, signed_k, transform)
        end
        # Only the integral part is row weighted. The cached mass matrices
        # already include the orbit weights. Convert the retained -C to C here.
        if prepared.symmetry_mode == :off
            c .*= -one(T)
        else
            weights = MtlArray(Complex{T}.(p1_symmetry_orbit_weights(mesh, prepared.symmetry_mode)))
            try
                a .*= reshape(weights, :, 1)
                c .= .-c .* reshape(weights, :, 1)
                Metal.synchronize()
            finally
                Metal.unsafe_free!(weights)
            end
        end
        scatter_metal_sparse_to_dense!(a, identity_cache.p1_p1; alpha=Complex{T}(0.5), add=true)
        scatter_metal_sparse_to_dense!(c, identity_cache.p1_dp0;
            alpha=T(0.5) * burton_miller_coupling(k), add=true)
        Metal.synchronize()
        succeeded = true
        return (a=a, c=c, on_gpu=true, gpu_backend=:metal, assembly_mode=:metal_coupled_combined)
    finally
        Metal.synchronize()
        owns_identity && release_metal_coupled_identity_cache!(identity_cache)
        blocks === nothing || Metal.unsafe_free!(blocks)
        if !succeeded
            a === nothing || Metal.unsafe_free!(a)
            c === nothing || Metal.unsafe_free!(c)
        end
    end
end

# Always transfer ownership to the host tuple, matching metal_host_operators.
function metal_host_coupled_burton_miller(combined)
    get(combined, :on_gpu, false) || return combined
    Metal.synchronize()
    a = Metal.is_shared(combined.a) ? unsafe_wrap(Array, combined.a) : Array(combined.a)
    c = Metal.is_shared(combined.c) ? unsafe_wrap(Array, combined.c) : Array(combined.c)
    return (a=a, c=c, on_gpu=false, metal_backing=(a=combined.a, c=combined.c))
end

function release_metal_coupled_burton_miller!(combined)
    backing = get(combined, :metal_backing, combined)
    Metal.unsafe_free!(backing.a)
    Metal.unsafe_free!(backing.c)
    return nothing
end
