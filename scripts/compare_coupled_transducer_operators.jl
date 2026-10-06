#!/usr/bin/env julia
# Frozen pre-split source plus packaged coupled fixtures; no baseline rewriting.
using Test, JSON, Base64, LinearAlgebra
module CurrentCoupled
include(joinpath(@__DIR__,"..","src","beat_engine","julia_local","BeatEngineCompiledDriver.jl"))
const ENGINE_DIR = joinpath(@__DIR__,"..","src","beat_engine","julia_local")
include(joinpath(@__DIR__,"..","src","beat_engine","julia_engine","CompiledCoupledWorkload.jl"))
end
module LegacyCoupled
const BASE_REVISION = "10cba4fcdba2773ee5bcbea485df1717bc57b489"
const ROOT = normpath(joinpath(@__DIR__,".."))
const ENGINE_DIR = joinpath(ROOT,"src","beat_engine","julia_local")
base_source(file) = read(Cmd(["git","-C",ROOT,"show","$BASE_REVISION:src/beat_engine/julia_local/$file"]),String)
const OLD_COUPLED_SOURCE = base_source("src/BeatEngineCoupled.jl")
const DRIVER_SOURCE = base_source("BeatEngineCompiledDriver.jl")
const COUPLED_INCLUDE = "include(joinpath(@__DIR__, \"src\", \"BeatEngineCoupled.jl\"))"
occursin(COUPLED_INCLUDE,DRIVER_SOURCE) || error("Pinned coupled include not found")
Base.include_string(@__MODULE__,replace(DRIVER_SOURCE,COUPLED_INCLUDE=>
    "Base.include_string(@__MODULE__, OLD_COUPLED_SOURCE, joinpath(@__DIR__, \"src\", \"BeatEngineCoupled.jl\"))"),
    joinpath(ENGINE_DIR,"BeatEngineCompiledDriver.jl"))
end

function fixture_operators(driver,precision)
    T = precision == "float32" ? Float32 : Float64
    root=joinpath(@__DIR__,"..","src","beat_engine","julia_local","tests","fixtures")
    fem=driver.load_gmsh41_volume(joinpath(root,"femvolume.msh"),T(0.001))
    bem=driver.load_gmsh22_with_tags(joinpath(root,"exterior_conforming.msh"),T(0.001))
    t=driver.ElectrodynamicTransducer{T}("fixture",[2],T[1.7],[1],T[-0.8],
        driver.SVector{3,T}(0,0,1),T(2),1,T(6),T(0.0005),T(7),T(0.015),T(0.0005),T(1))
    driver.assemble_transducer_operators(fem,bem,[t])
end

function captured(driver,request)
    mktemp() do _,io
        redirect_stdout(io) do
            driver.solve_request(deepcopy(request);event_mode=true)
        end
        seekstart(io)
        [e["result"] for e in JSON.parse.(readlines(io)) if haskey(e,"result")]
    end
end

BLAS.set_num_threads(1)
@testset "shared transducer helper: packaged coupled fixture byte equivalence" begin
    for precision in ("float32","float64")
        old=fixture_operators(LegacyCoupled,precision)
        new=fixture_operators(CurrentCoupled,precision)
        for key in keys(old)
            a,b=getproperty(old,key),getproperty(new,key)
            @test a.colptr == b.colptr
            @test a.rowval == b.rowval
            @test reinterpret(UInt8,a.nzval) == reinterpret(UInt8,b.nzval)
        end
        # Reuse the existing full compiled coupled workload graph/fixtures and
        # attach the moving driver to both FEM and BEM, exercising both halves.
        for convention in ("exp(+i omega t)","exp(-i omega t)")
            r=JSON.parse(JSON.json(CurrentCoupled.coupled_workload_request(;tiny=false)))
            r["frequencies_hz"]=[100.]
            r["solver_options"]["precision"]=precision
            r["solver_options"]["phasor_convention"]=convention
            r["solver_options"]["static_condensation"]=false
            s=r["compiled_system"]
            s["boundaries"][5]["kind"]="moving"
            push!(s["components"][1]["boundary_ids"],"boundary:exterior")
            s["components"][1]["parameters"]["boundary_motion_signs"]=Dict("boundary:exterior"=>-1)
            s["components"][1]["parameters"]["boundary_motion_weights"]=Dict("boundary:exterior"=>0.8)
            append!(r["outputs"],[Dict("id"=>q,"quantity"=>q,"target_ids"=>[],"options"=>Dict())
                for q in ("bem_boundary_pressure","bem_boundary_neumann","fem_nodal_pressure")])
            old_results=captured(LegacyCoupled,r);new_results=captured(CurrentCoupled,r)
            @test length(old_results) == length(new_results) == 1
            old_result,new_result=only(old_results),only(new_results)
            @test filter(p->first(p)!="diagnostics",new_result) == filter(p->first(p)!="diagnostics",old_result)
            for (a,b) in zip(old_result["quantities"],new_result["quantities"])
                @test a == b
                @test base64decode(a["values"]["content_base64"]) == base64decode(b["values"]["content_base64"])
            end
        end
    end
end
