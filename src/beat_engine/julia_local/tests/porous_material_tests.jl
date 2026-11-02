using Test, LinearAlgebra, SparseArrays, StaticArrays
if !isdefined(@__MODULE__, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore
if !isdefined(@__MODULE__, :BeatEngineCoupled)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
end
using .BeatEngineCoupled

# Soft fibrous material, Allard & Atalla 2nd ed. Table 11.2 (rigid frame here).
const POROUS_TEST_JCA = Dict{String,Any}(
    "model" => "jca", "frame" => "rigid", "porosity" => 0.98,
    "flow_resistivity_pa_s_per_m2" => 25000.0, "tortuosity" => 1.02,
    "viscous_characteristic_length_m" => 90e-6, "thermal_characteristic_length_m" => 180e-6)
const POROUS_TEST_AIR = (density=1.21, sound_speed=343.0, gamma=1.4)

porous_test_properties(frequency; convention="exp(+i omega t)", material=POROUS_TEST_JCA) =
    BeatEngineCore.with_phasor_convention(convention) do
        equivalent_fluid_properties(parse_region_material(material, Float64), frequency,
            POROUS_TEST_AIR.density, POROUS_TEST_AIR.sound_speed)
    end

@testset "JCA equivalent fluid limits (Allard & Atalla Eqs. 5.50-5.55)" begin
    rho0, c0, gamma = POROUS_TEST_AIR
    phi, sigma, alpha = 0.98, 25000.0, 1.02
    viscous_length, viscosity = 90e-6, 1.84e-5
    # The engine derives P0 from the region's air so that the high-frequency
    # adiabatic limit gamma*P0 equals the air bulk modulus rho0*c0^2.
    p0 = rho0 * c0^2 / gamma

    @testset "high frequency: rho -> alpha_inf rho0 / phi, K -> gamma P0 / phi" begin
        errors = Float64[]
        for frequency in (1e6, 1e8)
            properties = porous_test_properties(frequency)
            relative = abs(properties.density * phi / (alpha * rho0) - 1)
            # Leading-order asymptotic: |rho/(alpha rho0) - 1| ~ sqrt(2) delta_v / Lambda.
            delta_v = sqrt(2viscosity / (rho0 * 2pi * frequency))
            @test relative ≈ sqrt(2) * delta_v / viscous_length rtol=0.02
            push!(errors, relative)
            @test abs(properties.bulk_modulus * phi / (gamma * p0) - 1) < 0.02
        end
        @test errors[2] < errors[1] / 5
        @test porous_test_properties(1e8).bulk_modulus ≈ gamma * p0 / phi rtol=2e-3
    end

    @testset "low frequency: Im rho -> -sigma/omega (book convention), K -> P0 / phi" begin
        frequency = 0.01
        omega = 2pi * frequency
        properties = porous_test_properties(frequency)
        # rho/phi = alpha rho0/phi + sigma/(j omega) G, G -> 1 (viscous, isothermal regime).
        @test imag(properties.density) ≈ -sigma / omega rtol=1e-4
        @test properties.bulk_modulus ≈ p0 / phi rtol=1e-4
    end

    @testset "losses are passive and conjugate between phasor conventions" begin
        for frequency in (20.0, 500.0, 5000.0)
            positive = porous_test_properties(frequency)
            negative = porous_test_properties(frequency; convention="exp(-i omega t)")
            # e^{+j omega t}: Im rho < 0 (viscous drag), Im K > 0 (thermal lag), and the
            # wavenumber omega sqrt(rho/K) decays along exp(-j k x).
            @test imag(positive.density) < 0 && imag(positive.bulk_modulus) > 0
            @test imag(sqrt(positive.density / positive.bulk_modulus)) < 0
            @test negative.density ≈ conj(positive.density) rtol=1e-14
            @test negative.bulk_modulus ≈ conj(positive.bulk_modulus) rtol=1e-14
        end
    end

    @testset "a nearly free, fully open material is the region's air" begin
        open = merge(POROUS_TEST_JCA, Dict{String,Any}("porosity" => 1.0, "tortuosity" => 1.0,
            "flow_resistivity_pa_s_per_m2" => 1e-6, "viscous_characteristic_length_m" => 1000.0,
            "thermal_characteristic_length_m" => 1000.0))
        properties = porous_test_properties(1000.0; material=open)
        @test properties.density ≈ rho0 rtol=1e-6
        @test properties.bulk_modulus ≈ rho0 * c0^2 rtol=1e-6
    end

    @testset "legacy and explicit air carry the bulk loss factor and no porous parameters" begin
        legacy = parse_region_material(Dict{String,Any}("bulk_loss_factor" => 0.25), Float64)
        explicit = parse_region_material(Dict{String,Any}("model" => "air", "bulk_loss_factor" => 0.25), Float64)
        @test !is_porous(legacy) && !is_porous(explicit)
        @test region_material_wire(legacy) == region_material_wire(explicit) ==
              Dict{String,Any}("model" => "air", "bulk_loss_factor" => 0.25)
        @test is_porous(parse_region_material(POROUS_TEST_JCA, Float64))
        @test region_material_wire(parse_region_material(POROUS_TEST_JCA, Float64)) == POROUS_TEST_JCA
    end
end
