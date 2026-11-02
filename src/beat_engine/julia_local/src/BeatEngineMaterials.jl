# Region materials: air and rigid-frame equivalent-fluid porous media.
# Included in BeatEngineCoupled after BeatEngineThermoviscous.jl (shares THERMOVISCOUS_AIR).
#
# Reference: J.-F. Allard and N. Atalla, Propagation of Sound in Porous Media, 2nd ed.
# (Wiley, 2009). Johnson density Eqs. 5.50/5.52, simplified Lafarge bulk modulus
# Eqs. 5.51/5.53, JCA as JCAL with k0' = phi Lambda'^2 / 8 (Eqs. 5.54/5.55), and the
# equivalent fluid rho_eq = rho/phi, K_eq = K/phi (Eqs. 5.43/5.44), and the limp frame
# (Panneton approximation, Eq. 11.55) applied to rho_eq with K_eq unchanged. The book's
# e^{+j omega t} is written here as j omega = time_derivative(omega), so every formula
# follows the active phasor convention.
#
# Weak form. The air operator is K - k^2 M with Neumann data neumann_scale(rho0, omega) v.
# A porous region is assembled as (rho0/rho_eq) K_r - (omega^2 rho0/K_eq) M_r, i.e. its
# equivalent-fluid form (1/rho_eq) grad p . grad q - (omega^2/K_eq) p q multiplied by rho0.
# Its natural boundary term is then rho0 (1/rho_eq) dp/dn = neumann_scale(rho0, omega) v,
# so moving-boundary loads and transducer couplings are unchanged (this is where "use
# rho_eq where coupling uses rho0" enters), and volume flux is continuous between regions
# sharing nodes.

"""
Normalized region material. `model` is `:air`, `:jca` or `:jcal`; `frame` is `:none` for
air and `:rigid` or `:limp` for porous models. Porous parameters are SI;
`static_thermal_permeability_m2` is phi Lambda'^2 / 8 for JCA and the measured k0' for
JCAL; `frame_density_kg_per_m3` (rho1) is set only for a limp frame. Unused fields are NaN.
"""
struct RegionMaterial{T<:AbstractFloat}
    model::Symbol
    frame::Symbol
    bulk_loss_factor::T
    porosity::T
    flow_resistivity_pa_s_per_m2::T
    tortuosity::T
    viscous_characteristic_length_m::T
    thermal_characteristic_length_m::T
    static_thermal_permeability_m2::T
    frame_density_kg_per_m3::T
end

const POROUS_MATERIAL_KEYS = (
    "porosity",
    "flow_resistivity_pa_s_per_m2",
    "tortuosity",
    "viscous_characteristic_length_m",
    "thermal_characteristic_length_m",
)

is_porous(material::RegionMaterial) = material.model != :air

function _material_number(loss_model, key)
    value = get(loss_model, key, nothing)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        error("Region material parameter $key must be a finite number.")
    return Float64(value)
end

"""
    parse_region_material(loss_model, T) -> RegionMaterial{T}

Read a contract-validated region `loss_model`. Checks are repeated so that direct
callers cannot build an inadmissible material.
"""
function parse_region_material(loss_model, ::Type{T}) where {T<:AbstractFloat}
    model = String(get(loss_model, "model", "air"))
    nan = T(NaN)
    if model == "air"
        allowed = ("model", "bulk_loss_factor")
        all(key -> key in allowed, keys(loss_model)) ||
            error("Unsupported air material keys: $(join(sort([k for k in keys(loss_model) if !(k in allowed)]), ", ")).")
        factor = T(haskey(loss_model, "bulk_loss_factor") ? _material_number(loss_model, "bulk_loss_factor") : 0.0)
        zero(T) <= factor <= one(T) ||
            error("Bounded-region FEM bulk loss factor must be finite and between 0 and 1.")
        return RegionMaterial{T}(:air, :none, factor, nan, nan, nan, nan, nan, nan, nan)
    end
    model in ("jca", "jcal") || error("Unsupported region material model: $(repr(model)).")
    frame = String(get(loss_model, "frame", ""))
    frame in ("rigid", "limp") || error("Unsupported porous frame: $(repr(frame)).")
    allowed = ("model", "frame", POROUS_MATERIAL_KEYS...,
        (model == "jcal" ? ("static_thermal_permeability_m2",) : ())...,
        (frame == "limp" ? ("frame_density_kg_per_m3",) : ())...)
    all(key -> key in allowed, keys(loss_model)) ||
        error("Unsupported $model material keys: $(join(sort([k for k in keys(loss_model) if !(k in allowed)]), ", ")).")
    porosity, sigma, tortuosity, viscous, thermal = (_material_number(loss_model, key) for key in POROUS_MATERIAL_KEYS)
    0 < porosity <= 1 || error("Porosity must be in (0, 1].")
    sigma > 0 || error("Flow resistivity must be positive.")
    tortuosity >= 1 || error("Tortuosity must be at least 1.")
    viscous > 0 && thermal > 0 || error("Characteristic lengths must be positive.")
    # Lambda' < Lambda is unusual but admissible; warning about it is the client's job.
    k0_thermal = model == "jcal" ? _material_number(loss_model, "static_thermal_permeability_m2") :
                 porosity * thermal^2 / 8
    k0_thermal > 0 || error("Static thermal permeability must be positive.")
    frame_density = frame == "limp" ? _material_number(loss_model, "frame_density_kg_per_m3") : NaN
    frame == "rigid" || frame_density > 0 || error("Frame density must be positive.")
    return RegionMaterial{T}(Symbol(model), Symbol(frame), nan, T(porosity), T(sigma), T(tortuosity),
        T(viscous), T(thermal), T(k0_thermal), T(frame_density))
end

"""Normalized wire form (inverse of `parse_region_material`) for diagnostics and provenance."""
function region_material_wire(material::RegionMaterial)
    material.model == :air && return Dict{String,Any}(
        "model" => "air", "bulk_loss_factor" => Float64(material.bulk_loss_factor))
    keys = (POROUS_MATERIAL_KEYS...,
        (material.model == :jcal ? ("static_thermal_permeability_m2",) : ())...,
        (material.frame == :limp ? ("frame_density_kg_per_m3",) : ())...)
    return Dict{String,Any}(
        "model" => String(material.model),
        "frame" => String(material.frame),
        (key => Float64(getfield(material, Symbol(key))) for key in keys)...,
    )
end

"""Air constants of the porous model: the thermoviscous air plus P0 = rho0 c0^2 / gamma."""
function porous_air_constants(density, sound_speed)
    air = THERMOVISCOUS_AIR
    gamma = air.heat_capacity_ratio
    return (
        viscosity=air.dynamic_viscosity_pa_s,
        prandtl=air.dynamic_viscosity_pa_s * air.specific_heat_j_per_kg_k / air.thermal_conductivity_w_per_m_k,
        gamma=gamma,
        static_pressure_pa=Float64(density) * Float64(sound_speed)^2 / gamma,
    )
end

"""
    jcal_equivalent_fluid(omega, rho0, air; porosity, flow_resistivity, tortuosity,
                          viscous_length, thermal_length, static_thermal_permeability)

The single JCAL evaluation path, in Float64 and the active phasor convention. Returns
the equivalent-fluid `(density, bulk_modulus)` = (rho/phi, K/phi).
"""
function jcal_equivalent_fluid(omega, rho0, air; porosity, flow_resistivity, tortuosity,
        viscous_length, thermal_length, static_thermal_permeability)
    j_omega = time_derivative(Float64(omega))
    phi, sigma, alpha = porosity, flow_resistivity, tortuosity
    eta = air.viscosity
    # Johnson: rho = rho0 [alpha + nu phi/(j omega q0) G], q0 = eta/sigma (Eqs. 5.50, 5.52).
    viscous_shape = sqrt(1 + 4 * alpha^2 * eta * rho0 * j_omega / (sigma^2 * viscous_length^2 * phi^2))
    density = rho0 * alpha + sigma * phi * viscous_shape / j_omega
    # Simplified Lafarge with q0' = k0' (Eqs. 5.51, 5.53); nu' = eta / (rho0 Pr).
    nu_thermal = eta / (rho0 * air.prandtl)
    k0 = static_thermal_permeability
    thermal_shape = sqrt(1 + (2k0 / (phi * thermal_length))^2 * j_omega / nu_thermal)
    thermal_term = 1 + nu_thermal * phi * thermal_shape / (j_omega * k0)
    gamma = air.gamma
    bulk_modulus = gamma * air.static_pressure_pa / (gamma - (gamma - 1) / thermal_term)
    return (density=density / phi, bulk_modulus=bulk_modulus / phi)
end

"""
    equivalent_fluid_properties(material, frequency_hz, density, sound_speed)

Complex equivalent-fluid density and bulk modulus of a region at one frequency. Air
returns the region's lossless rho0 and rho0 c0^2 (its bulk loss factor is applied by
the operator coefficients, as in the air-only path).
"""
function equivalent_fluid_properties(material::RegionMaterial, frequency_hz, density, sound_speed)
    rho0, c0 = Float64(density), Float64(sound_speed)
    isfinite(frequency_hz) && frequency_hz > 0 && rho0 > 0 && c0 > 0 ||
        error("Region materials require positive finite frequency, density and sound speed.")
    material.model == :air && return (density=complex(rho0), bulk_modulus=complex(rho0 * c0^2))
    rigid = jcal_equivalent_fluid(2pi * Float64(frequency_hz), rho0, porous_air_constants(rho0, c0);
        porosity=Float64(material.porosity),
        flow_resistivity=Float64(material.flow_resistivity_pa_s_per_m2),
        tortuosity=Float64(material.tortuosity),
        viscous_length=Float64(material.viscous_characteristic_length_m),
        thermal_length=Float64(material.thermal_characteristic_length_m),
        static_thermal_permeability=Float64(material.static_thermal_permeability_m2))
    material.frame == :limp || return rigid
    return (density=limp_frame_density(rigid.density, Float64(material.frame_density_kg_per_m3),
                Float64(material.porosity), rho0),
            bulk_modulus=rigid.bulk_modulus)
end

"""
    limp_frame_density(rigid_density, frame_density, porosity, rho0)

Panneton's limp-frame density (Allard & Atalla Eq. 11.55) from the rigid-frame
equivalent density rho_eq: (rho_t rho_eq - rho0^2) / (rho_t + rho_eq - 2 rho0) with the
total apparent density rho_t = rho1 + phi rho0. It tends to rho_t as omega -> 0 and to
rho_eq as rho1 -> infinity.
"""
function limp_frame_density(rigid_density, frame_density, porosity, rho0)
    total = frame_density + porosity * rho0
    return (total * rigid_density - rho0^2) / (total + rigid_density - 2rho0)
end

"""
    region_operator_coefficients(material, omega, density, sound_speed)

Per-frequency `(stiffness, mass)` coefficients of one region in the rho0-normalized
operator a K_r + b M_r. Air reproduces the air-only operator K - k^2 M - i sign eta k^2 M;
porous regions use a = rho0/rho_eq and b = -omega^2 rho0/K_eq.
"""
function region_operator_coefficients(material::RegionMaterial, omega::T, density, sound_speed) where {T<:AbstractFloat}
    if material.model == :air
        squared_wavenumber = (omega / T(sound_speed))^2
        return (stiffness=one(Complex{T}),
                mass=-squared_wavenumber - Complex{T}(0, propagation_sign() * T(material.bulk_loss_factor) * squared_wavenumber))
    end
    fluid = equivalent_fluid_properties(material, Float64(omega) / 2pi, density, sound_speed)
    rho0 = Float64(density)
    return (stiffness=Complex{T}(rho0 / fluid.density),
            mass=Complex{T}(-Float64(omega)^2 * rho0 / fluid.bulk_modulus))
end

"""
Per-region stiffness and mass values on one shared sparsity pattern. Forming
A(omega) = sum_r a_r K_r + b_r M_r only rewrites `nzval`, so `colptr`/`rowval` are those
of the air operator for every frequency (symbolic analyses and the Float64 condensed
path's identical-pattern requirement stay valid).
"""
struct RegionFemOperator{T<:AbstractFloat}
    order::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    stiffness_values::Vector{Vector{T}}
    mass_values::Vector{Vector{T}}
    materials::Vector{RegionMaterial{Float64}}
end

function _structural_union(a::SparseMatrixCSC, b::SparseMatrixCSC)
    size(a) == size(b) || error("Region FEM operators must have matching shapes.")
    ia, ja, _ = findnz(a)
    ib, jb, _ = findnz(b)
    return sparse(vcat(ia, ib), vcat(ja, jb), ones(Int8, length(ia) + length(ib)), size(a)...)
end

function _values_on_pattern(pattern::SparseMatrixCSC, matrix::SparseMatrixCSC{T}) where {T}
    values = zeros(T, nnz(pattern))
    for column in 1:size(matrix, 2)
        slot = pattern.colptr[column]
        last_slot = pattern.colptr[column + 1] - 1
        for index in nzrange(matrix, column)
            row = matrix.rowval[index]
            while slot <= last_slot && pattern.rowval[slot] < row
                slot += 1
            end
            slot <= last_slot && pattern.rowval[slot] == row ||
                error("Region FEM operator entry ($row, $column) lies outside the shared pattern.")
            values[slot] = matrix.nzval[index]
        end
    end
    return values
end

"""
    prepare_region_fem_operator(mesh, tetrahedron_ranges, materials, stiffness, mass;
                                consistent_mass_weight)

Assemble each region's stiffness and (blended) mass over its own tetrahedra and store
them on the pattern of the global `stiffness`/`mass` pair.
"""
function prepare_region_fem_operator(mesh::VolumeMesh{T}, tetrahedron_ranges, materials,
        stiffness::SparseMatrixCSC{T}, mass::SparseMatrixCSC{T}; consistent_mass_weight=one(T)) where {T}
    length(tetrahedron_ranges) == length(materials) || error("Each FEM region needs one material.")
    pattern = _structural_union(stiffness, mass)
    quadratic = is_quadratic(mesh)
    stiffness_values, mass_values = Vector{T}[], Vector{T}[]
    for range in tetrahedron_ranges
        region_mesh = VolumeMesh(mesh.vertices, mesh.tetrahedra[range], mesh.tetra_physical_tags[range],
            mesh.boundary_faces, mesh.boundary_physical_tags, mesh.physical_names,
            quadratic ? mesh.quadratic_tetrahedra[range] : mesh.quadratic_tetrahedra,
            mesh.quadratic_boundary_faces)
        region_stiffness, region_mass = assemble_fem_matrices(region_mesh)
        push!(stiffness_values, _values_on_pattern(pattern, region_stiffness))
        push!(mass_values, _values_on_pattern(pattern,
            blend_fem_mass_matrix(region_mass, T(consistent_mass_weight))))
    end
    return RegionFemOperator{T}(size(pattern, 1), pattern.colptr, pattern.rowval,
        stiffness_values, mass_values, collect(RegionMaterial{Float64}, materials))
end

"""Form A(omega) = sum_r a_r K_r + b_r M_r on the shared pattern."""
function assemble_region_fem_operator(operator::RegionFemOperator{T}, omega, density, sound_speed) where {T}
    values = zeros(Complex{T}, length(operator.rowval))
    for (material, stiffness, mass) in zip(operator.materials, operator.stiffness_values, operator.mass_values)
        coefficients = region_operator_coefficients(material, T(omega), density, sound_speed)
        @. values += coefficients.stiffness * stiffness + coefficients.mass * mass
    end
    return SparseMatrixCSC(operator.order, operator.order, copy(operator.colptr), copy(operator.rowval), values)
end

"""Per-region material diagnostics (normalized material and, if porous, rho_eq and K_eq)."""
function region_material_diagnostics(region_ids, materials, frequency_hz, density, sound_speed)
    return [begin
        entry = Dict{String,Any}("region_id" => String(id), "material" => region_material_wire(material))
        if is_porous(material)
            fluid = equivalent_fluid_properties(material, frequency_hz, density, sound_speed)
            entry["equivalent_density_kg_per_m3"] = [real(fluid.density), imag(fluid.density)]
            entry["equivalent_bulk_modulus_pa"] = [real(fluid.bulk_modulus), imag(fluid.bulk_modulus)]
        end
        entry
    end for (id, material) in zip(region_ids, materials)]
end
