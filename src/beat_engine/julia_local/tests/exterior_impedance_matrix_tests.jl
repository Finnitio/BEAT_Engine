# Standalone, and safe to include in runtests.jl via an isolated driver namespace.
module ExteriorImpedanceMatrixTests
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))

include(joinpath(@__DIR__,"exterior_test_helpers.jl"))

@testset "two pulsating sphere mutual load, both phasors" begin
    a,d,rho,c,ka = 0.1,2.0,1.21,343.0,0.05
    omega = ka*c/a
    left = sphere(a,SVector(0.,0.,0.),2)
    right = sphere(a,SVector(d,0.,0.),3)
    mesh = join_meshes(left,right)
    components = [Dict("id"=>id) for id in ("left","right")]
    # Reverse excitation order to ensure columns are mapped by component id.
    excitations = [(component_id="right",tags=[3],amplitudes=[1.]),
                   (component_id="left",tags=[2],amplitudes=[1.])]
    matrices = Matrix{ComplexF64}[]
    for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
        with_phasor_convention(convention) do
            pressures = pressure_columns(mesh,excitations,omega,rho,c)
            z,meta = exterior_impedance_matrix(mesh,pressures,excitations,components,[], :off)
            @test meta["component_ids"] == ["left","right"]
            @test meta["row_weights"] == [1.,1.]
            @test meta["phasor_convention"] == convention
            @test eltype(z) == ComplexF64
            for (i,column) in ((1,2),(2,1))
                @test z[i,i] == exterior_component_impedance(mesh,pressures[column],excitations[column],:off,Float64)
            end
            reference = 4pi*a^2*rho*c*(a/d)*(im*ka)/(1+im*ka)*exp(-im*(omega/c)*(d-a))
            convention == NEGATIVE_TIME_PHASOR && (reference = conj(reference))
            # Taylor averaging the incident Helmholtz wave over a finite receiver
            # first changes its mean at O((ka)^2). Receiver scattering and further
            # exchanges add small-body/separation corrections O((a/d)^2).
            # A conservative coefficient 4 budgets those asymptotic omissions;
            # separately budget faceting/quadrature with 4 times the area deficit.
            # This is an engineering bound, not a rigorous remainder estimate.
            # The ~7% allowance catches sign, conjugation and copy-count errors,
            # but not errors under ~5%: e^{-ikd} instead of e^{-ik(d-a)}, or a
            # missing (1+ika), can pass. This is not a fine formula discriminator.
            approximation_budget = 4*(ka^2 + (a/d)^2)
            discretization_budget = 4*(1-sum(left.areas)/(4pi*a^2))
            tolerance = approximation_budget + discretization_budget
            phase_tolerance_rad = 0.07 # absolute angle, about 4 degrees
            relative_error = abs(z[1,2]/reference-1)
            @info "Mutual pulsating-sphere oracle" convention faces=length(mesh.faces) relative_error tolerance approximation_budget discretization_budget reciprocity=meta["reciprocity_max_rel"] passivity=meta["passivity_min_eig"]
            @test relative_error < tolerance
            @test abs(angle(z[1,2]/reference)) < phase_tolerance_rad
            @test abs(conj(z[1,2])/reference-1) > 5tolerance
            @test meta["reciprocity_max_rel"] < 1e-10
            @test meta["passivity_min_eig"] > 0
            reordered,_ = exterior_impedance_matrix(mesh,pressures,excitations,components,["right","left"],:off)
            @test reordered == z[[2,1],[2,1]]
            subset,_ = exterior_impedance_matrix(mesh,pressures,excitations,components,["right"],:off)
            @test subset == z[2:2,2:2]
            @test_throws "known motion-basis" exterior_impedance_matrix(mesh,pressures,excitations,components,["missing"],:off)
            push!(matrices,z)
        end
    end
    @test matrices[2] ≈ conj.(matrices[1]) rtol=1e-12
end

@testset "asymmetric receiver/source orientation and physical-copy counts" begin
    # Two disjoint faces, different weights and deliberately unequal pressure
    # columns. A transpose cannot pass even though its diagonal is unchanged.
    mesh = BoundaryMesh(SVector{3,Float64}[(0,1,0),(1,1,0),(0,2,0),
        (2,1,0),(3,1,0),(2,2,0)], [(1,2,3),(4,5,6)], [2,3])
    receiver1 = (component_id="first",tags=[2],amplitudes=[2.])
    receiver2 = (component_id="second",tags=[3],amplitudes=[3.])
    source1_pressure = ComplexF64[1,1,1,4,4,4]
    source2_pressure = ComplexF64[7+2im,7+2im,7+2im,5,5,5]
    # Reversed solve order exercises the component-id mapping too.
    excitations = [receiver2,receiver1]
    pressures = [source2_pressure,source1_pressure]
    components = [Dict("id"=>"first"),Dict("id"=>"second")]
    for (symmetry,count) in ((:off,1),(:x,2),(:xy,4),(:ground,1))
        z,meta = exterior_impedance_matrix(mesh,pressures,excitations,components,[],symmetry)
        @test physical_radiator_count(symmetry) == count
        @test z[1,2] == exterior_component_impedance(mesh,source2_pressure,receiver1,symmetry,Float64)
        @test z[1,2] == count*(7+2im) # force on receiver 1 from source 2
        @test z[2,1] == count*6
        @test z[1,2] != z[2,1]
        @test diag(z) == [exterior_component_impedance(mesh,source1_pressure,receiver1,symmetry,Float64),
                         exterior_component_impedance(mesh,source2_pressure,receiver2,symmetry,Float64)]
        @test z == count*ComplexF64[1 7+2im; 6 7.5]
        @test meta["row_weights"] == [1.,1.]
    end
    # Pressure inputs stand for already solved columns including image pressure.
    # Ground integrates one physical radiator regardless of those images.
    @test exterior_component_force(mesh,source2_pressure,receiver1,0.5,Float64) == 0.5*(7+2im)
end

@testset "x images equal physical mirrored pair" begin
    a,rho,c,omega = .1,1.21,343.,100.
    representative = sphere(a,SVector(.5,0.,0.),2;refinements=1)
    mirror = BoundaryMesh([SVector(-v[1],v[2],v[3]) for v in representative.vertices],
        [(f[1],f[3],f[2]) for f in representative.faces],representative.physical_tags)
    full = join_meshes(representative,mirror)
    excitation = (component_id="pair",tags=[2],amplitudes=[1.])
    components = [Dict("id"=>"pair")]
    for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
        with_phasor_convention(convention) do
            errors = Float64[]
            for order in (4,6)
                reduced_p = pressure_columns(representative,[excitation],omega,rho,c;symmetry=:x,order=order)
                full_p = pressure_columns(full,[excitation],omega,rho,c;order=order)
                reduced,meta = exterior_impedance_matrix(representative,reduced_p,[excitation],components,[],:x)
                explicit,_ = exterior_impedance_matrix(full,full_p,[excitation],components,[],:off)
                relative_error = norm(reduced-explicit)/norm(explicit)
                push!(errors,relative_error)
                @info "Mirrored-pair quadrature convergence" convention order relative_error
                # Tensor Gauss/Duffy rules depend on vertex parametrization.
                # Reflection reverses winding; two separately discretized systems
                # agree within quadrature error, rather than algebraic round-off.
                @test relative_error < (order == 4 ? 1e-8 : 1e-10)
                @test reduced[1,1] == exterior_component_impedance(representative,only(reduced_p),excitation,:x,Float64)
                @test meta["row_weights"] == [1.]
                @test meta["passivity_min_eig"] > 0
            end
            @test errors[2] < errors[1]/5
        end
    end
end

function captured_result(request)
    mktemp() do _,io
        redirect_stdout(io) do
            solve_request(request;event_mode=true)
        end
        flush(io); seekstart(io)
        only([JSON.parse(line)["result"] for line in eachline(io)])
    end
end

function fixture_request()
    request = JSON.parsefile(joinpath(@__DIR__,"..","..","beat_contract","example-exterior-request.json"))
    request["compiled_system"]["meshes"][1]["file"] = joinpath(@__DIR__,"..","test_meshes","two_tetrahedra.msh")
    request["compiled_system"]["meshes"][1]["scale_to_m"] = 0.1
    request["compiled_system"]["boundaries"][1]["group"]["tag"] = 1
    request["frequencies_hz"] = [100.]
    request["solver_options"]["quadrature_order"] = 2
    request["solver_options"]["singular_order"] = 2
    request["outputs"] = [Dict("id"=>q,"quantity"=>q,"target_ids"=>[],"options"=>Dict())
        for q in ("bem_boundary_pressure","bem_boundary_neumann","radiation_impedance")]
    request
end

@testset "compiled wire output and signed force integration" begin
    for precision in ("float32","float64"), convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR), profile in ("uniform_normal","rigid_translation")
        request = fixture_request()
        if profile == "rigid_translation"
            request["compiled_system"]["contract_version"] = 2
            request["compiled_system"]["components"][1]["parameters"] = Dict("motion_profile"=>profile,"motion_axis"=>[0.,0.,1.])
        end
        request["solver_options"]["precision"] = precision
        request["solver_options"]["phasor_convention"] = convention
        legacy = captured_result(request)
        push!(request["outputs"],Dict("id"=>"matrix","quantity"=>"radiation_impedance_matrix","target_ids"=>[],"options"=>Dict()))
        result = captured_result(request)
        @test result["quantities"][1:3] == legacy["quantities"]
        matrix = last(result["quantities"])
        @test matrix["values"]["shape"] == [1,1]
        @test matrix["values"]["dtype"] == "complex128"
        @test matrix["axes"] == ["receiver_component","source_component"]
        @test matrix["unit"] == "N*s/m"
        decode(item,T) = only(reinterpret(T,base64decode(item["values"]["content_base64"])))
        tolerance = precision == "float32" ? 5e-6 : 1e-14
        @test decode(matrix,ComplexF64) ≈ decode(result["quantities"][3],precision == "float32" ? ComplexF32 : ComplexF64) rtol=tolerance
    end
    # One negative n·axis, positive weight: one sign in both source and force.
    mesh = BoundaryMesh(SVector{3,Float32}[(0,0,0),(1,0,0),(0,1,0)],[(1,2,3)],[2])
    e = (component_id="signed",tags=[2],amplitudes=Float32[2],motion_axis=SVector{3,Float32}(0,0,-1))
    z,_ = exterior_impedance_matrix(mesh,[fill(ComplexF32(1,2),3)],[e],[Dict("id"=>"signed")],[],:off)
    @test only(z) == -(1+2im)
    empty,meta = exterior_impedance_matrix(mesh,[],[],[],[],:off)
    @test size(empty) == (0,0)
    @test meta["reciprocity_max_rel"] == 0
    @test meta["passivity_min_eig"] === nothing
end
@testset "compiled multi-RHS component and target ordering" begin
    request = fixture_request()
    system = request["compiled_system"]
    original = read(system["meshes"][1]["file"],String)
    # Give the second existing tetrahedron its own physical moving tag.
    tagged = replace(original,r"(?m)^([5-8]) 2 2 1 1" => s"\1 2 2 2 2")
    mktemp() do path,io
        write(io,tagged); flush(io)
        system["meshes"][1]["file"] = path
        boundary = deepcopy(system["boundaries"][1])
        boundary["id"] = "boundary:second"
        boundary["group"]["tag"] = 2
        push!(system["boundaries"],boundary)
        component = deepcopy(system["components"][1])
        component["id"] = "component:second"
        component["boundary_ids"] = ["boundary:second"]
        component["parameters"] = Dict("boundary_motion_weights"=>Dict("boundary:second"=>2.))
        push!(system["components"],component)
        port = deepcopy(system["excitation_ports"][1])
        port["id"] = "excitation:second"
        port["component_id"] = "component:second"
        push!(system["excitation_ports"],port)
        request["excitation_port_ids"] = ["excitation:second","excitation:source"]
        request["solver_options"]["precision"] = "float64"
        for (id,targets) in (("all",[]),("reverse",["component:second","component:source"]),("subset",["component:second"]))
            push!(request["outputs"],Dict("id"=>id,"quantity"=>"radiation_impedance_matrix","target_ids"=>targets,"options"=>Dict()))
        end
        result = captured_result(request)
        items = Dict(item["id"]=>item for item in result["quantities"])
        decode(item) = collect(reinterpret(ComplexF64,base64decode(item["values"]["content_base64"])))
        full = permutedims(reshape(decode(items["all"]),2,2)) # C-order wire
        reversed = permutedims(reshape(decode(items["reverse"]),2,2))
        @test items["all"]["metadata"]["component_ids"] == ["component:source","component:second"]
        @test items["reverse"]["metadata"]["component_ids"] == ["component:second","component:source"]
        @test reversed == full[[2,1],[2,1]]
        @test only(decode(items["subset"])) == full[2,2]
        @test diag(full) == decode(items["radiation_impedance"])
        @test result["diagnostics"]["factorization_count"] == 1
        # The coarse non-spherical Burton-Miller response is not symmetric.
        # Publish its raw residual; do not silently enforce reciprocity. The
        # independent pulsating-sphere oracle above checks physical reciprocity.
        residual = maximum(abs,full-transpose(full))/maximum(abs,full)
        @test items["all"]["metadata"]["reciprocity_max_rel"] ≈ residual rtol=1e-14
        @test residual > 1e-10
        @test abs(full[2,1]) > 0
    end
end

@testset "fail closed independently of negotiation" begin
    for version in (1,2)
        request = fixture_request()
        request["compiled_system"]["contract_version"] = version
        undriven = deepcopy(request["compiled_system"]["components"][1])
        undriven["id"] = "undriven"
        undriven["kind"] = "electrodynamic_transducer"
        push!(request["compiled_system"]["components"],undriven)
        @test_throws "exterior electrodynamic_transducer requires compiled-system contract version 3" solve_request(request)
    end
    for bounded in (false,true)
        request = fixture_request()
        bounded && (request["compiled_system"]["regions"][1]["kind"] = "bounded_air")
        request["compiled_system"]["components"][1]["kind"] = "passive_radiator"
        @test_throws "passive_radiator is not implemented" solve_request(request)
    end
    request = fixture_request()
    unexcited = deepcopy(request["compiled_system"]["components"][1])
    unexcited["id"] = "unexcited"
    push!(request["compiled_system"]["components"],unexcited)
    @test_throws "radiator axis remains compiled component order" solve_request(request)
    request["outputs"] = [Dict("id"=>"matrix","quantity"=>"radiation_impedance_matrix","target_ids"=>[],"options"=>Dict())]
    @test last(captured_result(request)["quantities"])["metadata"]["component_ids"] == ["component:source"]
    request["outputs"][1]["target_ids"] = ["unexcited"]
    @test_throws "targets must be excited" solve_request(request)
end

end # module
