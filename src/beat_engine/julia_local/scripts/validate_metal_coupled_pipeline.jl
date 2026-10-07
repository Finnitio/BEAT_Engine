#!/usr/bin/env julia
# Gate for the coupled sweep pipeline on Metal (BLAB_COUPLED_SWEEP_PIPELINE).
#
# Assembling a frequency's BEM operators one frequency ahead, on another task,
# may change when work happens, never what is computed. This sweeps the bundled
# coupled fixture through the condensed solver twice -- once building each
# frequency itself, once from operators the sweep pipeline assembled and
# combined ahead -- and requires every solution to be finite and bit-identical (compared as raw
# bits, so signed zeros and NaN payloads count), under a
# prescribed-velocity and a voltage excitation.
#
#   julia --threads=4 --project=src/beat_engine/julia_metal \
#       src/beat_engine/julia_local/scripts/validate_metal_coupled_pipeline.jl
#
#   BLAB_VALIDATE_STEPS  frequencies from 100 Hz to 4 kHz (default 6)

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupledCondensed

using StaticArrays

const FIXTURE_ROOT = normpath(joinpath(@__DIR__, "..", "tests", "fixtures"))

# Bit-identical and finite: raw representations compared, so neither signed zeros (`==`) nor NaN
# payloads (`isequal`) can hide a difference.
same_bits(a::Nothing, b::Nothing) = true
same_bits(a, b) = typeof(a) == typeof(b) && size(a) == size(b) && all(isfinite, a) && all(isfinite, b) &&
                  reinterpret(UInt8, vec(collect(a))) == reinterpret(UInt8, vec(collect(b)))

function main()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    metal.functional() || error("Metal.functional() is false.")
    Threads.nthreads() > 1 || error("The pipeline needs a second Julia thread; start julia with --threads.")

    steps = parse(Int, get(ENV, "BLAB_VALIDATE_STEPS", "6"))
    frequencies = Float32[exp10(x) for x in range(log10(100.0), log10(4000.0); length=steps)]
    quadrature_order = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "1"))
    singular_order = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "1"))
    fem_mesh = load_gmsh41_volume(joinpath(FIXTURE_ROOT, "femvolume.msh"), 0.001f0)
    bem_mesh = load_gmsh22_with_tags(joinpath(FIXTURE_ROOT, "exterior_conforming.msh"), 0.001f0)
    interface_map = build_conforming_interface_map(
        fem_mesh, bem_mesh, physical_tag(fem_mesh, 2, "Interface"), 2,
    )
    radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
    transducer = ElectrodynamicTransducer{Float32}(
        "component:pipeline-validation", [radiator_tag], Float32[1], [1], Float32[-1],
        SVector(0f0, 0f0, 1f0), 2f0, 1, 6f0, 0.0005f0, 7f0, 0.015f0, 0.0005f0, 1f0,
    )
    excitations = [
        (kind=:normal_velocity, fem_boundary_tags=[radiator_tag], fem_boundary_weights=Float32[1],
         bem_source_index=0, transducer_index=0, amplitude=ComplexF32(1, 0)),
        (kind=:voltage, fem_boundary_tags=Int[], fem_boundary_weights=Float32[],
         bem_source_index=0, transducer_index=1, amplitude=ComplexF32(2.83, 0)),
    ]
    options = (
        quadrature_order=quadrature_order,
        singular_order=singular_order,
        bulk_loss_factor_by_vertex=fill(0.01f0, length(fem_mesh.vertices)),
        transducers=[transducer],
        validation_diagnostics=false,
    )
    probe = build_condensed_coupled_system(fem_mesh, bem_mesh, interface_map, frequencies[1], 343f0, 1.21f0; options...)
    retained = probe.retained_fem_vertices
    release_condensed_coupled_system!(probe)
    cache = prepare_condensed_coupled_cache(
        fem_mesh, bem_mesh, interface_map;
        quadrature_order=quadrature_order, singular_order=singular_order,
        retained_fem_vertices=retained,
        bulk_loss_factor_by_vertex=options.bulk_loss_factor_by_vertex,
        bem_backend=:metal,
    )
    sweep(bem_operators) = map(enumerate(frequencies)) do (index, frequency)
        system = build_condensed_coupled_system(
            fem_mesh, bem_mesh, interface_map, frequency, 343f0, 1.21f0;
            options..., cache=cache, bem_operators=bem_operators(index),
        )
        try
            solve_condensed_coupled_excitations(system, excitations)
        finally
            release_condensed_coupled_system!(system)
        end
    end

    pipeline = nothing
    release = produced -> release_condensed_bem_operators!(produced, :metal)
    try
        sequential = sweep(index -> nothing)
        pipeline = start_sweep_assembly_pipeline(
            index -> assemble_condensed_bem_operators(
                bem_mesh, cache, frequencies[index], 343f0; singular_order=singular_order,
            ),
            steps, 1, release,
        )
        pipelined = sweep(index -> (() -> take_sweep_assembly!(pipeline, index)))
        failures = 0
        fields = (:fem_pressure, :bem_pressure, :bem_neumann, :interface_flux, :diaphragm_velocity, :voice_coil_current)
        for (index, frequency) in enumerate(frequencies), excitation in eachindex(excitations), field in fields
            a = getproperty(sequential[index][excitation], field)
            b = getproperty(pipelined[index][excitation], field)
            if !same_bits(a, b)
                failures += 1
                println("  MISMATCH $(frequency) Hz excitation $(excitation) $(field): max |diff| $(maximum(abs, a .- b))")
            end
        end
        println("coupled sweep pipeline: $(steps) frequencies x $(length(excitations)) excitations, " *
                "threads=$(Threads.nthreads()), device=$(metal.device().name)")
        failures == 0 || error("$(failures) pipelined outputs differ from the sequential sweep.")
        println("METAL_COUPLED_PIPELINE_VALIDATION_OK")
    finally
        shutdown_sweep_assembly_pipeline!(pipeline, release)
        release_condensed_coupled_cache!(cache)
    end
end

main()
