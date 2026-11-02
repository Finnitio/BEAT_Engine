# Shared analytical reference for porous tests: the air constants, the soft fibrous preset and
# the JCA/JCAL equivalent fluid written in book form, independently of the engine. Included
# by the porous, region-seam and porous-coupled test modules and by porous_tube_helpers.jl.
const RHO0, C0 = 1.21, 343.0
# Soft fibrous material, Allard & Atalla 2nd ed. Table 11.2, rigid frame.
const SOFT_FIBROUS = Dict{String,Any}(
    "model" => "jca", "frame" => "rigid", "porosity" => 0.98,
    "flow_resistivity_pa_s_per_m2" => 25000.0, "tortuosity" => 1.02,
    "viscous_characteristic_length_m" => 90e-6, "thermal_characteristic_length_m" => 180e-6)

"""
Analytical equivalent fluid (book form, e^{+j omega t}): Johnson density and simplified
Lafarge bulk modulus, Allard & Atalla Eqs. 5.50-5.53, with q0' = k0' for JCAL and
q0' = phi Lambda'^2/8 for JCA (Eqs. 5.54/5.55); a limp frame applies Eq. 11.55.
"""
function reference_jca(parameters, frequency; rho0=RHO0, c0=C0)
    j, omega = im, 2pi * frequency
    phi = parameters["porosity"]; sigma = parameters["flow_resistivity_pa_s_per_m2"]
    alpha = parameters["tortuosity"]
    lambda_v = parameters["viscous_characteristic_length_m"]
    lambda_t = parameters["thermal_characteristic_length_m"]
    eta, gamma = 1.84e-5, 1.4
    prandtl = eta * 1005.0 / 0.0257
    p0 = rho0 * c0^2 / gamma
    rho = rho0 * alpha * (1 + sigma * phi / (j * omega * rho0 * alpha) *
          sqrt(1 + 4j * alpha^2 * eta * rho0 * omega / (sigma^2 * lambda_v^2 * phi^2)))
    q0_thermal = get(parameters, "static_thermal_permeability_m2", phi * lambda_t^2 / 8)
    nu_thermal = eta / (rho0 * prandtl)
    alpha_thermal = 1 + nu_thermal * phi / (j * omega * q0_thermal) *
                    sqrt(1 + 4j * omega * q0_thermal^2 / (nu_thermal * lambda_t^2 * phi^2))
    bulk = gamma * p0 / (gamma - (gamma - 1) / alpha_thermal)
    rho_eq = rho / phi
    if parameters["frame"] == "limp"
        rho_t = parameters["frame_density_kg_per_m3"] + phi * rho0
        rho_eq = (rho_t * rho_eq - rho0^2) / (rho_t + rho_eq - 2rho0)
    end
    return (density=rho_eq, bulk_modulus=bulk / phi)
end
