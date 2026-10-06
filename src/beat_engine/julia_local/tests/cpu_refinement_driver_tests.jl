# Exercise the actual legacy/source driver JSON boundary, not a copied Dict.
module CpuRefinementDriverTests
using Test, JSON
using ..BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))

function capture_source_diagnostics(request)
    mktemp() do _, output
        redirect_stdout(output) do
            solve_request(request)
        end
        flush(output)
        seekstart(output)
        events = JSON.parse.(readlines(output))
        @test last(events)["type"] == "completed"
        only(event["result"]["diagnostics"] for event in events if event["type"] == "result")
    end
end

@testset "source/legacy refinement diagnostics reach result JSON" begin
    mktempdir() do directory
        mesh_path = joinpath(directory,"tetrahedron.msh")
        write(mesh_path,raw"""$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 0.08 0.0 0.0
3 0.0 0.08 0.0
4 0.0 0.0 0.08
$EndNodes
$Elements
4
1 2 2 2 2 1 3 2
2 2 2 2 2 1 2 4
3 2 2 2 2 2 3 4
4 2 2 2 2 3 1 4
$EndElements
""")
        request = Dict("beat_engine_backend"=>"cpu", "frequencies_hz"=>[500.0],
            "config"=>Dict("mesh_file"=>mesh_path, "scale_factor"=>1.0,
                "tag_throat"=>2, "symmetry"=>"off", "min_angle"=>0.0,
                "max_angle"=>90.0, "step_size"=>30.0, "distance"=>2.0,
                "quadrature_order"=>3, "singular_order"=>3, "regular_quadrature_mode"=>"fixed"))
        withenv("BLAB_BEAT_FUSED_BM"=>"1", "BLAB_BEAT_DENSE_SOLVE"=>"lu") do
            for steps in (nothing,"0","1")
                diagnostics = withenv("BLAB_BEAT_CPU_LU_REFINEMENT_STEPS"=>steps) do
                    capture_source_diagnostics(request)
                end
                @test diagnostics["dense_solve_method"] == "lu"
                @test isempty(diagnostics["dense_solve_iterations"])
                @test isempty(diagnostics["dense_solve_relative_residuals"])
                if steps != "1"
                    @test all(!startswith(key,"dense_solve_refinement_") for key in keys(diagnostics))
                    continue
                end
                @test diagnostics["dense_solve_refinement_steps"] in 0:1
                @test diagnostics["dense_solve_refinement_operator"] == "rounded_float32"
                @test diagnostics["dense_solve_refinement_tolerance"] == 1e-10
                @test length(diagnostics["dense_solve_refinement_working_residuals"]) == 1
                returned = diagnostics["dense_solve_refinement_returned_residuals"]
                @test length(returned) == 1 && all(isfinite,returned)
                @test (diagnostics["dense_solve_refinement_status"] == "converged") ==
                    all(<=(diagnostics["dense_solve_refinement_tolerance"]),returned)
                @test diagnostics["dense_solve_refinement_working_status"] in
                    ("converged","stagnated","step_limit")
                @test occursin("Float64 residual refinement",diagnostics["message"])
            end
        end
    end
end

if basename(dirname(Base.active_project())) == "julia_metal"
    import BeatEngineCompiledMetalBundle
    const RefinementBundle = BeatEngineCompiledMetalBundle
else
    import BeatEngineCompiledCpuBundle
    const RefinementBundle = BeatEngineCompiledCpuBundle
end
@testset "compiled refinement diagnostics reach result JSON" begin
    bundle = RefinementBundle
    mktempdir() do directory
        mesh_path = joinpath(directory,"plate.msh")
        write(mesh_path,bundle.workload_plate_mesh())
        request = bundle.JSON.parse(bundle.JSON.json(bundle.representative_workload_request(mesh_path)))
        request["frequencies_hz"] = [1000.0]
        request["outputs"] = [output for output in request["outputs"] if output["id"] == "surface:p"]
        withenv("BLAB_BEAT_DENSE_SOLVE"=>"lu") do
            for steps in ("0","1")
                diagnostics = withenv("BLAB_BEAT_CPU_LU_REFINEMENT_STEPS"=>steps) do
                    mktemp() do _, output
                        redirect_stdout(output) do
                            bundle.solve_request(request;event_mode=true)
                        end
                        flush(output)
                        seekstart(output)
                        events = bundle.JSON.parse.(readlines(output))
                        only(event["result"]["diagnostics"] for event in events if event["type"] == "result")
                    end
                end
                if steps == "0"
                    @test all(!startswith(key,"dense_solve_refinement_") for key in keys(diagnostics))
                    continue
                end
                @test diagnostics["dense_solve_refinement_operator"] == "rounded_float32"
                @test diagnostics["dense_solve_refinement_steps"] in 0:1
                @test diagnostics["dense_solve_refinement_tolerance"] == 1e-10
                working = diagnostics["dense_solve_refinement_working_residuals"]
                returned = diagnostics["dense_solve_refinement_returned_residuals"]
                @test !isempty(working) && length(working) == length(returned)
                @test all(isfinite,working) && all(isfinite,returned)
                @test (diagnostics["dense_solve_refinement_status"] == "converged") ==
                    all(<=(diagnostics["dense_solve_refinement_tolerance"]),returned)
                @test diagnostics["dense_solve_refinement_working_status"] in
                    ("converged","stagnated","step_limit")
            end
        end
    end
end
end
