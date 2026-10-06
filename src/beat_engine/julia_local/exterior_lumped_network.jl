"""
Integrate the existing unit-velocity pressure columns into a force/velocity matrix.
All integration arithmetic is Float64, even when the BEM pressures are Float32.
Ideal-source rows count physical radiators, exactly as the legacy self load does.
"""
function exterior_impedance_matrix(mesh, pressures, excitations, components, target_ids, symmetry_mode; force_matrix=nothing)
    column_by_component = Dict(excitation.component_id => index for (index, excitation) in enumerate(excitations))
    ids = isempty(target_ids) ? [String(component["id"]) for component in components
        if haskey(column_by_component, String(component["id"]))] : String.(target_ids)
    length(ids) == length(Set(ids)) || error("radiation_impedance_matrix targets must be unique.")
    all(id -> haskey(column_by_component, id), ids) ||
        error("radiation_impedance_matrix targets must be known motion-basis components.")
    matrix = zeros(ComplexF64, length(ids), length(ids))
    # Ideal rows integrate all real copies. Shared force rows for transducers
    # include completion only; orbit stays in row_weights.
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
            matrix[i,j] = force_matrix === nothing ?
                exterior_component_force(mesh, pressure, receiver, copy_count, Float64) :
                sum(force_matrix[:,column_by_component[receiver_id]] .* pressure)
        end
    end
    selected = [excitations[column_by_component[id]] for id in ids]
    weights = Float64[get(e,:orbit,1) for e in selected]
    completions = Float64[get(e,:completion,copy_count) for e in selected]
    areas = Float64[]
    cancelling = Bool[]
    for (index,e) in enumerate(selected)
        amplitudes = Dict(zip(e.tags,e.amplitudes))
        contributions = [Float64(mesh.areas[f]) * get(amplitudes,mesh.physical_tags[f],0.0) *
            exterior_motion_factor(e,mesh.normals[f],Float64) for f in eachindex(mesh.faces)]
        copies = completions[index] * weights[index]
        # For transducers use exactly the shared force coefficients, including
        # completion and the same normalized axis used by the discretisation.
        area = force_matrix === nothing ? copies * sum(contributions) :
            weights[index] * sum(force_matrix[:,column_by_component[ids[index]]])
        push!(areas,area)
        push!(cancelling,abs(area) <= 1e-8 * copies * sum(abs,contributions))
    end
    weighted = Diagonal(weights) * matrix
    scale = maximum(abs, weighted; init=0.0)
    reciprocity = scale == 0.0 ? 0.0 : maximum(abs, weighted - transpose(weighted); init=0.0) / scale
    passivity = isempty(ids) ? nothing : eigmin(Hermitian((weighted + weighted') / 2))
    metadata = Dict{String,Any}(
        "component_ids" => ids,
        "kinds" => [get(e,:kind,"ideal_velocity_source") for e in selected],
        "surface_completion_factors" => completions,
        "physical_driver_orbit_counts" => Int.(weights),
        "effective_volume_area_m2" => areas,
        "effective_volume_area_definition" => "signed integral of motion factor over physical moving surfaces; real symmetry copies included; ground images excluded",
        "effective_volume_area_zero_or_near_cancelling" => cancelling,
        "effective_volume_area_cancellation_relative_tolerance" => 1e-8,
        "row_weights" => weights,
        "definition" => "force_per_unit_velocity; per-row physical-copy weighting in row_weights",
        "phasor_convention" => phasor_convention(),
        "reciprocity_max_rel" => reciprocity,
        "passivity_min_eig" => passivity,
    )
    return matrix, metadata
end

"""Exterior transducers currently require closed, consistently outward-wound solids."""
function validate_exterior_transducer_surface!(mesh)
    isempty(mesh.faces) && error("Exterior transducers require closed BEM surfaces.")
    edge_faces = Dict{Tuple{Int,Int},Vector{Tuple{Int,Int}}}()
    for (index, (a,b,c)) in enumerate(mesh.faces)
        for (u,v) in ((a,b),(b,c),(c,a))
            push!(get!(edge_faces, minmax(u,v), Tuple{Int,Int}[]), (index, u < v ? 1 : -1))
        end
    end
    all(length(entries) == 2 && entries[1][2] == -entries[2][2] for entries in values(edge_faces)) ||
        error("Exterior transducers require closed BEM surfaces with consistent winding; open/two-sided diaphragms are unsupported.")
    neighbours = [Int[] for _ in mesh.faces]
    for entries in values(edge_faces)
        a,b = entries[1][1],entries[2][1]
        push!(neighbours[a],b); push!(neighbours[b],a)
    end
    seen = falses(length(mesh.faces))
    for seed in eachindex(mesh.faces)
        seen[seed] && continue
        stack = [seed]
        seen[seed] = true
        origin = mesh.vertices[first(mesh.faces[seed])]
        volume = 0.0
        while !isempty(stack)
            index = pop!(stack)
            a,b,c = mesh.faces[index]
            volume += dot(mesh.vertices[a]-origin,
                cross(mesh.vertices[b]-origin, mesh.vertices[c]-origin)) / 6
            for next in neighbours[index]
                seen[next] && continue
                seen[next] = true
                push!(stack,next)
            end
        end
        volume > 0 || error("Exterior transducers require outward-wound closed BEM solids with positive volume.")
    end
    return nothing
end

"""All drivers first (compiled order), then requested ideal sources, once per component."""
function exterior_motion_basis(system, requested_ports, boundaries, bem_domain, mesh, region, symmetry)
    symmetry in (:off, :ground) || error("Exterior electrodynamic_transducers support only off and ground symmetry.")
    validate_exterior_transducer_surface!(mesh)
    # Float64 here preserves LEM parameters as well as BEM geometry. No FEM tags
    # are passed: the shared parser cannot resolve a boundary into another region.
    transducers, index_by_id = electrodynamic_transducers_from_wire(
        system["components"], boundaries, Dict{String,Int}(), bem_domain.boundary_tag_by_id,
        nothing, mesh, Float64, symmetry,
    )
    basis = NamedTuple[]
    for (index,t) in enumerate(transducers)
        push!(basis, (component_id=t.id, tags=t.bem_boundary_tags, amplitudes=t.bem_motion_signs,
            motion_axis=t.motion_axis, kind="electrodynamic_transducer", transducer_index=index,
            completion=t.surface_completion_factor, orbit=t.physical_driver_orbit_count))
    end
    ideal_ports = String[]
    ideal_ids = Set{String}()
    for port_id in requested_ports
        port = object_by_id(system["excitation_ports"], port_id, "excitation port")
        id = String(port["component_id"])
        if haskey(index_by_id,id)
            port["kind"] == "voltage" || error("Exterior transducer excitation ports must be voltage ports.")
        else
            id in ideal_ids && continue
            push!(ideal_ports,port_id); push!(ideal_ids,id)
        end
    end
    ideals = exterior_excitations(ideal_ports, (ports=system["excitation_ports"],items=system["components"]),
        boundaries,bem_domain.boundary_tag_by_id,region,symmetry,Float64)
    append!(basis,ideals)
    operators = assemble_bem_transducer_operators(mesh,transducers)
    force = exterior_basis_force_matrix(mesh,basis,operators,symmetry)
    return (; basis, transducers, operators, force)
end

function exterior_basis_force_matrix(mesh, basis, operators, symmetry)
    force = zeros(Float64,length(mesh.vertices),length(basis))
    for (column,e) in enumerate(basis)
        if get(e,:kind,"ideal_velocity_source") == "electrodynamic_transducer"
            force[:,column] = operators.bem_force[:,e.transducer_index]
        else
            amplitudes = Dict(zip(e.tags,e.amplitudes))
            for (index,face) in enumerate(mesh.faces)
                coefficient = get(amplitudes,mesh.physical_tags[index],0.0) *
                    exterior_motion_factor(e,mesh.normals[index],Float64)
                nodal_area = physical_radiator_count(symmetry) * coefficient * mesh.areas[index] / 3
                for vertex in face
                    force[vertex,column] += nodal_area
                end
            end
        end
    end
    return force
end

function exterior_basis_neumann(mesh, e, density, omega, operators)
    get(e,:kind,"ideal_velocity_source") == "electrodynamic_transducer" ||
        return exterior_neumann(mesh,e,density,omega)
    return neumann_scale(density,omega) .* Vector(operators.bem_normal_velocity[:,e.transducer_index])
end

"""Eliminate the BEM into per-copy force rows; undriven coils are shorted (V=0)."""
function solve_exterior_lumped_network(z, transducers, basis, ports, requested_ports,
    omega, density, sound_speed, reference_voltage)
    isfinite(reference_voltage) && reference_voltage > 0 || error("transducer_reference_voltage_v must be finite and positive.")
    d,n,c = length(transducers),length(basis),length(requested_ports)
    zm = ComplexF64[mechanical_impedance(t,omega,density,sound_speed) for t in transducers]
    ze = ComplexF64[electrical_impedance(t,omega) for t in transducers]
    bl = Float64[t.bl_n_per_a for t in transducers]
    network = [Matrix{ComplexF64}(z[1:d,1:d]) + Diagonal(zm) -Diagonal(bl);
               Diagonal(bl) Diagonal(ze)]
    rhs = zeros(ComplexF64,2d,c)
    u = zeros(ComplexF64,n,c)
    index_by_id = Dict(e.component_id => i for (i,e) in enumerate(basis))
    for (column,port_id) in enumerate(requested_ports)
        port = object_by_id(ports,port_id,"excitation port")
        index = index_by_id[String(port["component_id"])]
        if index <= d
            port["kind"] == "voltage" || error("Exterior transducer excitation ports must be voltage ports.")
            rhs[d+index,column] = reference_voltage
        else
            port["kind"] == "normal_velocity" || error("Exterior ideal excitation ports must be normal-velocity ports.")
            u[index,column] = 1
            rhs[1:d,column] = -z[1:d,index]
        end
    end
    solution = network \ rhs
    u[1:d,:] = solution[1:d,:]
    return (; velocity=u, current=solution[d+1:2d,:],
        residual_max_abs=maximum(abs,network*solution-rhs;init=0.0))
end

function exterior_transducer_metadata(transducers)
    Dict{String,Any}(
        "component_ids" => [t.id for t in transducers],
        "surface_completion_factors" => [t.surface_completion_factor for t in transducers],
        "physical_driver_orbit_counts" => [t.physical_driver_orbit_count for t in transducers],
    )
end
