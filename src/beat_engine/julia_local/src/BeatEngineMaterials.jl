# Region materials: air and rigid-frame equivalent-fluid porous media.
# Included in BeatEngineCoupled after BeatEngineThermoviscous.jl (shares THERMOVISCOUS_AIR).
#
# Reference: J.-F. Allard and N. Atalla, Propagation of Sound in Porous Media, 2nd ed.
# (Wiley, 2009). Johnson density Eqs. 5.50/5.52, simplified Lafarge bulk modulus
# Eqs. 5.51/5.53, JCA as JCAL with k0' = phi Lambda'^2 / 8 (Eqs. 5.54/5.55), and the
# equivalent fluid rho_eq = rho/phi, K_eq = K/phi (Eqs. 5.43/5.44). The book's
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
Normalized region material. `model` is `:air` or `:jca` (`:jcal` follows); `frame` is
`:none` for air and `:rigid` for porous models (`:limp` follows). Porous parameters are
SI; `static_thermal_permeability_m2` is phi Lambda'^2 / 8 for JCA. Unused fields are NaN.
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
        return RegionMaterial{T}(:air, :none, factor, nan, nan, nan, nan, nan, nan)
    end
    model == "jca" || error("Unsupported region material model: $(repr(model)).")
    frame = String(get(loss_model, "frame", ""))
    frame == "rigid" || error("Unsupported porous frame: $(repr(frame)).")
    allowed = ("model", "frame", POROUS_MATERIAL_KEYS...)
    all(key -> key in allowed, keys(loss_model)) ||
        error("Unsupported $model material keys: $(join(sort([k for k in keys(loss_model) if !(k in allowed)]), ", ")).")
    porosity, sigma, tortuosity, viscous, thermal = (_material_number(loss_model, key) for key in POROUS_MATERIAL_KEYS)
    0 < porosity <= 1 || error("Porosity must be in (0, 1].")
    sigma > 0 || error("Flow resistivity must be positive.")
    tortuosity >= 1 || error("Tortuosity must be at least 1.")
    viscous > 0 && thermal > 0 || error("Characteristic lengths must be positive.")
    k0_thermal = porosity * thermal^2 / 8
    return RegionMaterial{T}(:jca, Symbol(frame), nan, T(porosity), T(sigma), T(tortuosity),
        T(viscous), T(thermal), T(k0_thermal))
end

"""Normalized wire form (inverse of `parse_region_material`) for diagnostics and provenance."""
function region_material_wire(material::RegionMaterial)
    material.model == :air && return Dict{String,Any}(
        "model" => "air", "bulk_loss_factor" => Float64(material.bulk_loss_factor))
    return Dict{String,Any}(
        "model" => String(material.model),
        "frame" => String(material.frame),
        (key => Float64(getfield(material, Symbol(key))) for key in POROUS_MATERIAL_KEYS)...,
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
    return jcal_equivalent_fluid(2pi * Float64(frequency_hz), rho0, porous_air_constants(rho0, c0);
        porosity=Float64(material.porosity),
        flow_resistivity=Float64(material.flow_resistivity_pa_s_per_m2),
        tortuosity=Float64(material.tortuosity),
        viscous_length=Float64(material.viscous_characteristic_length_m),
        thermal_length=Float64(material.thermal_characteristic_length_m),
        static_thermal_permeability=Float64(material.static_thermal_permeability_m2))
end
