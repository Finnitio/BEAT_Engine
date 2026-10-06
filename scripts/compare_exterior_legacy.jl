#!/usr/bin/env julia
# Opt-in compatibility gate; deliberately not included by runtests.jl.
# Run from the repository root with the julia_local project (see contract docs).
using Test, JSON, Base64, LinearAlgebra

module CurrentExterior
include(joinpath(@__DIR__, "..", "src", "beat_engine", "julia_local", "BeatEngineCompiledDriver.jl"))
end

module LegacyExterior
const BASE_REVISION = "4839c7e62d45295489cac919c5e4b8d36b1b0f1f"
const ROOT = normpath(joinpath(@__DIR__, ".."))
const ENGINE = joinpath(ROOT, "src", "beat_engine", "julia_local")
base_source(file) = read(Cmd(["git", "-C", ROOT, "show",
    "$BASE_REVISION:src/beat_engine/julia_local/$file"]), String)
const LEGACY_GROUND_SOURCE = base_source("compiled_ground_contract.jl")
const DRIVER_SOURCE = base_source("BeatEngineCompiledDriver.jl")
# Freeze the old force integrator as well as the driver: using the current helper
# in both modules would hide a common regression in the explicit-count refactor.
const GROUND_INCLUDE = "include(joinpath(@__DIR__, \"compiled_ground_contract.jl\"))"
occursin(GROUND_INCLUDE, DRIVER_SOURCE) || error("Pinned driver ground include not found")
Base.include_string(@__MODULE__, replace(DRIVER_SOURCE, GROUND_INCLUDE =>
    "Base.include_string(@__MODULE__, LEGACY_GROUND_SOURCE, joinpath(@__DIR__, \"compiled_ground_contract.jl\"))"),
    joinpath(ENGINE, "BeatEngineCompiledDriver.jl"))
# Other numerical dependencies are shared and unchanged from the base revision.
end

function captured_results(driver, request)
    mktemp() do _, io
        redirect_stdout(io) do
            driver.solve_request(deepcopy(request); event_mode=true)
        end
        flush(io)
        seekstart(io)
        events = [JSON.parse(line) for line in eachline(io)]
        [event["result"] for event in events if haskey(event, "result")]
    end
end

function fixture_request(precision, convention, profile)
    root = normpath(joinpath(@__DIR__, ".."))
    request = JSON.parsefile(joinpath(root, "src", "beat_engine", "beat_contract", "example-exterior-request.json"))
    system = request["compiled_system"]
    system["meshes"][1]["file"] = joinpath(root, "src", "beat_engine", "julia_local", "test_meshes", "two_tetrahedra.msh")
    system["meshes"][1]["scale_to_m"] = 0.1
    system["boundaries"][1]["group"]["tag"] = 1
    # Retain the fixture bytes. Two ports on its existing moving tag exercise
    # excitation ordering and unequal motion weights without rewriting a mesh.
    second = deepcopy(system["components"][1])
    second["id"] = "component:second"
    second["parameters"] = Dict{String,Any}("boundary_motion_weights" => Dict("boundary:source" => 2.0))
    push!(system["components"], second)
    port = deepcopy(system["excitation_ports"][1])
    port["id"] = "excitation:second"
    port["component_id"] = "component:second"
    push!(system["excitation_ports"], port)
    if profile == "rigid_translation"
        system["contract_version"] = 2
        for component in system["components"]
            component["parameters"]["motion_profile"] = profile
            component["parameters"]["motion_axis"] = [0.0, 0.0, 1.0]
        end
    end
    request["frequencies_hz"] = [200.0, 100.0] # deliberately unsorted
    request["excitation_port_ids"] = ["excitation:second", "excitation:source"]
    request["solver_options"] = Dict("precision" => precision, "bem_backend" => "cpu",
        "phasor_convention" => convention, "quadrature_order" => 2, "singular_order" => 2)
    request["outputs"] = [Dict("id" => q, "quantity" => q, "target_ids" => [],
        "options" => q == "exterior_pressure" ? Dict("points_m" => [[0.0, 0.0, 2.0]]) : Dict())
        for q in ("exterior_pressure", "bem_boundary_pressure", "bem_boundary_neumann", "radiation_impedance")]
    request
end

BLAS.set_num_threads(1)
@testset "exterior legacy byte equivalence against $(LegacyExterior.BASE_REVISION)" begin
    for precision in ("float32", "float64"), convention in ("exp(+i omega t)", "exp(-i omega t)"),
        profile in ("uniform_normal", "rigid_translation")
        @testset "$precision / $convention / $profile" begin
            request = fixture_request(precision, convention, profile)
            legacy = captured_results(LegacyExterior, request)
            current = captured_results(CurrentExterior, request)
            @test length(current) == length(legacy) == 2
            @test [r["freq_hz"] for r in current] == [r["freq_hz"] for r in legacy]
            # Driver emission order is compared directly, independently of any
            # request sorting performed by the legacy implementation.
            for (old, new) in zip(legacy, current)
                @test new["excitation_port_ids"] == old["excitation_port_ids"] == request["excitation_port_ids"]
                # Compare every non-diagnostic result field, preserving quantity
                # order. Run-dependent timings and provenance are excluded with
                # diagnostics; no quantity metadata is excluded.
                @test filter(p -> first(p) != "diagnostics", new) == filter(p -> first(p) != "diagnostics", old)
                @test length(new["quantities"]) == length(old["quantities"]) == 4
                for (old_q, new_q) in zip(old["quantities"], new["quantities"])
                    @test new_q == old_q # entire object, including id, metadata and encoding
                    @test new_q["values"]["content_base64"] == old_q["values"]["content_base64"]
                    @test base64decode(new_q["values"]["content_base64"]) == base64decode(old_q["values"]["content_base64"])
                    @test new_q["values"]["shape"] == old_q["values"]["shape"]
                    @test new_q["values"]["dtype"] == old_q["values"]["dtype"]
                    @test new_q["axes"] == old_q["axes"]
                    @test new_q["unit"] == old_q["unit"]
                    @test get(new_q, "metadata", nothing) == get(old_q, "metadata", nothing)
                end
            end
        end
    end
end
