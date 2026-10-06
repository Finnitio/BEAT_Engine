"""
Integrate the existing unit-velocity pressure columns into a force/velocity matrix.
All integration arithmetic is Float64, even when the BEM pressures are Float32.
Ideal-source rows count physical radiators, exactly as the legacy self load does.
"""
function exterior_impedance_matrix(mesh, pressures, excitations, components, target_ids, symmetry_mode)
    column_by_component = Dict(excitation.component_id => index for (index, excitation) in enumerate(excitations))
    ids = isempty(target_ids) ? [String(component["id"]) for component in components
        if haskey(column_by_component, String(component["id"]))] : String.(target_ids)
    length(ids) == length(Set(ids)) || error("radiation_impedance_matrix targets must be unique.")
    all(id -> haskey(column_by_component, id), ids) ||
        error("radiation_impedance_matrix targets must be known excited ideal components.")
    matrix = zeros(ComplexF64, length(ids), length(ids))
    # Ideal rows integrate all real copies; future transducer rows pass completion
    # here instead, keeping their orbit count in row_weights.
    copy_count = physical_radiator_count(symmetry_mode)
    for (j, source_id) in enumerate(ids)
        pressure = ComplexF64.(pressures[column_by_component[source_id]])
        for (i, receiver_id) in enumerate(ids)
            excitation = excitations[column_by_component[receiver_id]]
            # Convert weights and axes before multiplication; do not add a second
            # sign to the signed n·axis projection in exterior_motion_factor.
            receiver = (tags=excitation.tags, amplitudes=Float64.(excitation.amplitudes))
            if get(excitation, :motion_axis, nothing) !== nothing
                receiver = merge(receiver, (motion_axis=Float64.(excitation.motion_axis),))
            end
            matrix[i,j] = exterior_component_force(mesh, pressure, receiver, copy_count, Float64)
        end
    end
    weights = ones(Float64, length(ids))
    weighted = Diagonal(weights) * matrix
    scale = maximum(abs, weighted; init=0.0)
    reciprocity = scale == 0.0 ? 0.0 : maximum(abs, weighted - transpose(weighted); init=0.0) / scale
    passivity = isempty(ids) ? nothing : eigmin(Hermitian((weighted + weighted') / 2))
    metadata = Dict{String,Any}(
        "component_ids" => ids,
        "kinds" => fill("ideal_velocity_source", length(ids)),
        "row_weights" => weights,
        "definition" => "force_per_unit_velocity; per-row physical-copy weighting in row_weights",
        "phasor_convention" => phasor_convention(),
        "reciprocity_max_rel" => reciprocity,
        "passivity_min_eig" => passivity,
    )
    return matrix, metadata
end
