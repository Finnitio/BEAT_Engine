# Standalone CPU gate for the v3 exterior BEM/LEM network.
module ExteriorTransducerTests
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))

function mesh_file(io,mesh)
    println(io,"\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n",length(mesh.vertices))
    for (i,v) in enumerate(mesh.vertices)
        println(io,i," ",join(v," "))
    end
    println(io,"\$EndNodes\n\$Elements\n",length(mesh.faces))
    for (i,f) in enumerate(mesh.faces)
        println(io,i," 2 2 ",mesh.physical_tags[i]," ",mesh.physical_tags[i]," ",join(f," "))
    end
    println(io,"\$EndElements")
    flush(io)
end

function driver_request(file; tags=[2], ideal=false)
    r = JSON.parsefile(joinpath(@__DIR__,"..","..","beat_contract","example-exterior-request.json"))
    s = r["compiled_system"]
    s["contract_version"] = 3
    s["meshes"][1]["file"] = file
    s["meshes"][1]["scale_to_m"] = 1.0
    s["components"] = Any[]; s["boundaries"] = Any[]; s["excitation_ports"] = Any[]
    for (index,tag) in enumerate(tags)
        id,bid,pid = "driver:$index","boundary:$index","port:$index"
        isideal = ideal && index == length(tags)
        push!(s["boundaries"],Dict("id"=>bid,"name"=>bid,"kind"=>"moving","region_id"=>"region:exterior",
            "group"=>Dict("mesh_id"=>"mesh:exterior","dimension"=>2,"tag"=>tag,"name"=>nothing),"parameters"=>Dict()))
        params = isideal ? Dict{String,Any}() : Dict{String,Any}("re_ohm"=>6.,"le_h"=>0.0005,
            "bl_n_per_a"=>7.,"mmd_kg"=>0.015,"cms_m_per_n"=>0.0005,"rms_n_s_per_m"=>1.,"motion_axis"=>[0.,0.,1.])
        push!(s["components"],Dict("id"=>id,"name"=>id,"kind"=>isideal ? "ideal_velocity_source" : "electrodynamic_transducer",
            "boundary_ids"=>[bid],"parameters"=>params))
        push!(s["excitation_ports"],Dict("id"=>pid,"name"=>pid,"component_id"=>id,"kind"=>isideal ? "normal_velocity" : "voltage"))
    end
    r["excitation_port_ids"] = ["port:1"]
    r["frequencies_hz"] = [100.]
    r["solver_options"] = Dict("precision"=>"float64","bem_backend"=>"cpu","quadrature_order"=>4,
        "singular_order"=>4,"regular_quadrature_mode"=>"fixed","transducer_reference_voltage_v"=>2.83)
    r["outputs"] = [Dict("id"=>q,"quantity"=>q,"target_ids"=>[],"options"=> q == "exterior_pressure" ?
        Dict("points_m"=>[[0.,0.,0.15],[0.,0.,2.],[1.,0.,2.]]) : Dict())
        for q in ("diaphragm_velocity","voice_coil_current","radiation_impedance_matrix","exterior_pressure",
                  "bem_boundary_pressure","bem_boundary_neumann")]
    r
end

function captured(r)
    mktemp() do _,io
        redirect_stdout(io) do
            solve_request(deepcopy(r);event_mode=true)
        end
        seekstart(io)
        [e["result"] for e in JSON.parse.(readlines(io)) if haskey(e,"result")]
    end
end

function decoded(item)
    values = collect(reinterpret(ComplexF64,base64decode(item["values"]["content_base64"])))
    shape = Tuple(Int.(item["values"]["shape"]))
    length(shape) == 1 ? values : permutedims(reshape(values,reverse(shape)))
end
quantities(r) = Dict(item["id"]=>item for item in r["quantities"])
relative(a,b) = norm(a-b)/norm(b)

# Independent positive-time dipole solution: p = q(a) h_1^(2)(kr)/(k h_1^(2)'(ka)) cos(theta).
function dipole_pressure(point,a,k,rho,omega,u)
    radius = norm(point); x = k*a; y=k*radius
    h = -exp(-im*y)*(y-im)/y^2
    derivative = exp(-im*x)*(im*x^2+2x-2im)/x^3
    -im*rho*omega*u * h/(k*derivative) * point[3]/radius
end

@testset "LEM-driven oscillating sphere: load, u, i, Zin and near/far dipole" begin
    a,rho,c,V = 0.1,1.21,343.,2.83
    mesh = sphere(a,SVector(0.,0.,0.),2)
    # Planar P1/DP0 faces: budget four times the independently computed area
    # deficit, plus 1% quadrature/interpolation allowance, before seeing results.
    tolerance = 4*(1-sum(mesh.areas)/(4pi*a^2)) + 0.01
    @test tolerance < 0.07
    mktemp() do path,io
        mesh_file(io,mesh)
        request = driver_request(path)
        request["frequencies_hz"] = [0.3*c/(2pi*a),1.1*c/(2pi*a)]
        runs = []
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
            request["solver_options"]["phasor_convention"] = convention
            results = captured(request); push!(runs,results)
            for result in results
                omega = 2pi*result["freq_hz"]; k=omega/c; x=k*a
                z = (4pi*a^2*rho*c/3)*(x^4+im*x*(2+x^2))/(4+x^4)
                zm = 1+im*(omega*0.015-1/(omega*0.0005)); ze=6+im*omega*0.0005
                convention == NEGATIVE_TIME_PHASOR && ((z,zm,ze) = conj.((z,zm,ze)))
                u = (7/ze)*V/(zm+z+49/ze); current=(V-7u)/ze
                items = quantities(result)
                numerical_z = only(decoded(items["radiation_impedance_matrix"]))
                numerical_u = only(decoded(items["diaphragm_velocity"]))
                numerical_i = only(decoded(items["voice_coil_current"]))
                @info "oscillating sphere" convention x tolerance z_error=abs(numerical_z/z-1) u_error=abs(numerical_u/u-1)
                @test abs(numerical_z/z-1) < tolerance
                @test abs(numerical_u/u-1) < tolerance
                @test abs(numerical_i/current-1) < tolerance
                @test abs((V/numerical_i)/(V/current)-1) < tolerance
                expected = [dipole_pressure(SVector(Float64.(p)...),a,k,rho,omega,1.) for p in
                    request["outputs"][4]["options"]["points_m"]]
                convention == NEGATIVE_TIME_PHASOR && (expected=conj.(expected))
                expected *= u
                actual = vec(decoded(items["exterior_pressure"]))
                @info "dipole fields" relative_error=relative(actual,expected)
                @test maximum(abs.(actual./expected .- 1)) < tolerance
                @test relative(actual,conj.(expected)) > 3tolerance
                @test relative(actual,sqrt(2).*expected) > 3tolerance
                @test abs(numerical_z/conj(z)-1) > 3tolerance
                meta = items["radiation_impedance_matrix"]["metadata"]
                @test abs(only(meta["effective_volume_area_m2"])) < 1e-14
                @test meta["effective_volume_area_zero_or_near_cancelling"] == [true]
                @test meta["row_weights"] == [1.]
                @test items["diaphragm_velocity"]["values"]["shape"] == [1,1]
                @test items["voice_coil_current"]["metadata"] == items["diaphragm_velocity"]["metadata"]
                @test result["diagnostics"]["exterior_lumped_network"]["termination"] == "shorted"
            end
        end
        for (positive,negative) in zip(runs...)
            p,n=quantities(positive),quantities(negative)
            for id in keys(p)
                @test decoded(n[id]) ≈ conj.(decoded(p[id])) rtol=1e-12 atol=1e-14
            end
        end
        # No implicit motion/current output, even though they were computed.
        request["frequencies_hz"] = [100.]
        request["outputs"] = [request["outputs"][4]]
        @test [q["quantity"] for q in only(captured(request))["quantities"]] == ["exterior_pressure"]
    end
end

@testset "two spheres: undriven shorted response and exact BEM elimination" begin
    mesh = join_meshes(sphere(0.1,SVector(0.,0.,0.),2;refinements=1),
        sphere(0.1,SVector(0.4,0.,0.),3;refinements=1),
        sphere(0.1,SVector(0.8,0.,0.),4;refinements=1))
    mktemp() do path,io
        mesh_file(io,mesh)
        request=driver_request(path;tags=[2,3,4],ideal=true)
        system=request["compiled_system"]
        # A second voltage port on the same driver must reproduce its column.
        push!(system["excitation_ports"],Dict("id"=>"alias","name"=>"Alias","component_id"=>"driver:1","kind"=>"voltage"))
        request["excitation_port_ids"] = ["port:3","port:1","alias"]
        append!(request["outputs"],[
            Dict("id"=>"self-load","quantity"=>"radiation_impedance","target_ids"=>[],"options"=>Dict()),
            Dict("id"=>"selected-load","quantity"=>"radiation_impedance_matrix",
                 "target_ids"=>["driver:3","driver:2"],"options"=>Dict()),
        ])
        system["components"][1]["parameters"]["re_ohm"] = 6.000000000001
        # One negative sign with unequal weight exercises both shared operators.
        system["components"][2]["parameters"]["boundary_motion_signs"] = Dict("boundary:2"=>-1)
        system["components"][2]["parameters"]["boundary_motion_weights"] = Dict("boundary:2"=>1.7)
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
            with_phasor_convention(convention) do
                region=system["regions"][1]
                domain=aggregate_bem_region(system["meshes"],region,system["boundaries"],Float64)
                data=exterior_motion_basis(system,request["excitation_port_ids"],system["boundaries"],domain,domain.mesh,region,:off)
                basis,ts=data.basis,data.transducers
                @test eltype(ts[1].motion_axis) == Float64
                @test typeof(ts[1].re_ohm) == Float64
                @test ts[1].re_ohm == 6.000000000001
                @test ts[1].re_ohm != Float64(Float32(6.000000000001))
                omega,rho,c = 2pi*100.,1.21,343.
                k=omega/c; mesh=domain.mesh
                p1,dp0=build_p1_space(mesh),build_dp0_space(mesh); rule=triangle_rule(Float64,4)
                ops=assemble_regular_galerkin_operators(mesh,p1,dp0,k,rule;skip_singular=false,singular_order=4,backend=:cpu)
                ipp=assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:p1)
                ipq=assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:dp0)
                A,B=burton_miller_neumann_matrices(ops,ipp,ipq,k)
                Q=hcat([exterior_basis_neumann(mesh,e,rho,omega,data.operators) for e in basis]...)
                P=A \ (B*Q); Z=transpose(data.force)*P
                reduced=solve_exterior_lumped_network(Z,ts,basis,system["excitation_ports"],request["excitation_port_ids"],omega,rho,c,2.83)
                d=length(ts); n=length(mesh.vertices)
                zm=[mechanical_impedance(t,omega,rho,c) for t in ts]
                ze=[electrical_impedance(t,omega) for t in ts]; bl=[t.bl_n_per_a for t in ts]
                # Actual assembled BEM rows, followed by coupled mechanical and
                # electrical rows; ideal velocity is an independently fixed RHS.
                monolithic=[A -B*Q[:,1:d] zeros(ComplexF64,n,d);
                    transpose(data.force[:,1:d]) Diagonal(zm) -Diagonal(bl);
                    zeros(ComplexF64,d,n) Diagonal(bl) Diagonal(ze)]
                rhs=zeros(ComplexF64,n+2d,3)
                rhs[1:n,1]=B*Q[:,3]
                rhs[n+d+1,2:3].=2.83
                exact=monolithic \ rhs
                @test relative(P*reduced.velocity,exact[1:n,:]) < 1e-10
                @test relative(reduced.velocity[1:d,:],exact[n+1:n+d,:]) < 1e-10
                @test relative(reduced.current,exact[n+d+1:end,:]) < 1e-10
                @test reduced.velocity[:,2] == reduced.velocity[:,3]
                @test abs(reduced.velocity[2,2]) > 1e-10
                @test reduced.velocity[2,2] ≈ -Z[2,1]/(zm[2]+Z[2,2]+bl[2]^2/ze[2])*reduced.velocity[1,2] rtol=1e-12
                request["solver_options"]["phasor_convention"]=convention
                items=quantities(only(captured(request)))
                @test decoded(items["diaphragm_velocity"]) ≈ transpose(reduced.velocity[1:d,:]) rtol=1e-10
                @test decoded(items["voice_coil_current"]) ≈ transpose(reduced.current) rtol=1e-10
                @test decoded(items["bem_boundary_pressure"]) ≈ transpose(P*reduced.velocity) rtol=1e-10
                @test decoded(items["bem_boundary_neumann"]) ≈ transpose(Q*reduced.velocity) rtol=1e-10
                @test items["self-load"]["axes"] == ["radiator"]
                @test items["self-load"]["values"]["shape"] == [1]
                @test only(decoded(items["self-load"])) ≈ Z[3,3] rtol=1e-13
                @test decoded(items["selected-load"]) ≈ Z[[3,2],[3,2]] rtol=1e-10
                @test items["selected-load"]["metadata"]["component_ids"] == ["driver:3","driver:2"]
                reordered,_=exterior_impedance_matrix(mesh,collect(eachcol(P)),basis,reverse(system["components"]),[],:off;
                    force_matrix=data.force)
                @test reordered ≈ Z[[3,2,1],[3,2,1]] rtol=1e-14
                meta=items["radiation_impedance_matrix"]["metadata"]
                @test meta["component_ids"] == ["driver:1","driver:2","driver:3"]
                @test meta["kinds"] == ["electrodynamic_transducer","electrodynamic_transducer","ideal_velocity_source"]
                @test meta["effective_volume_area_m2"][3] ≈ sum(mesh.areas[findall(tag -> tag in basis[3].tags,mesh.physical_tags)]) rtol=1e-14
                @test meta["effective_volume_area_zero_or_near_cancelling"] == [true,true,false]
            end
        end
    end
end

@testset "two identical oscillating spheres, one voltage-driven" begin
    mesh=join_meshes(sphere(0.1,SVector(0.,0.,0.),2;refinements=1),
        sphere(0.1,SVector(0.4,0.,0.),3;refinements=1))
    mktemp() do path,io
        mesh_file(io,mesh)
        request=driver_request(path;tags=[2,3])
        for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
            request["solver_options"]["phasor_convention"]=convention
            items=quantities(only(captured(request)))
            z=decoded(items["radiation_impedance_matrix"])
            u=vec(decoded(items["diaphragm_velocity"]))
            current=vec(decoded(items["voice_coil_current"]))
            omega=2pi*100.
            zm=1+im*(omega*0.015-1/(omega*0.0005));ze=6+im*omega*0.0005
            convention == NEGATIVE_TIME_PHASOR && ((zm,ze)=conj.((zm,ze)))
            @test abs(u[2]) > 1e-10
            @test u[2] ≈ -z[2,1]/(zm+z[2,2]+49/ze)*u[1] rtol=1e-12
            @test current[2] ≈ -7u[2]/ze rtol=1e-12
            @test 7u[1]+ze*current[1] ≈ 2.83 rtol=1e-12
            @test items["diaphragm_velocity"]["metadata"]["component_ids"] == ["driver:1","driver:2"]
        end
    end
end

@testset "closure, winding, version, precision and deferred symmetry refusals" begin
    mesh=sphere(0.1,SVector(0.,0.3,0.),2;refinements=0)
    @test validate_exterior_transducer_surface!(mesh) === nothing
    open=BoundaryMesh(mesh.vertices,mesh.faces[2:end],mesh.physical_tags[2:end])
    @test_throws "closed BEM surfaces" validate_exterior_transducer_surface!(open)
    inward=BoundaryMesh(mesh.vertices,[(a,c,b) for (a,b,c) in mesh.faces],mesh.physical_tags)
    @test_throws "outward-wound" validate_exterior_transducer_surface!(inward)
    flipped=copy(mesh.faces); a,b,c=flipped[1]; flipped[1]=(a,c,b)
    @test_throws "consistent winding" validate_exterior_transducer_surface!(BoundaryMesh(mesh.vertices,flipped,mesh.physical_tags))
    mktemp() do path,io
        mesh_file(io,mesh)
        r=driver_request(path)
        for v in (1,2)
            r["compiled_system"]["contract_version"]=v
            @test_throws "requires compiled-system contract version 3" solve_request(r)
        end
        r["compiled_system"]["contract_version"]=3
        r["solver_options"]["precision"]="float32"
        @test_throws "require float64 BEM precision" solve_request(r)
        delete!(r["solver_options"],"precision")
        @test_throws "require float64 BEM precision" solve_request(r)
        r["solver_options"]["precision"]="float64"
        for symmetry in ("x","xy")
            r["solver_options"]["symmetry"]=symmetry
            @test_throws "support only off and ground symmetry" solve_request(r)
        end
        r["solver_options"]["symmetry"]="off"
        r["solver_options"]["bem_backend"]="metal"
        @test_throws "cannot use Metal: float64 BEM is unsupported" solve_request(r)
        @test_throws "cannot use Metal: float64 BEM is unsupported" solve_exterior_request(r,r["compiled_system"],r["compiled_system"]["regions"][1])
        r["solver_options"]["bem_backend"]="cpu"
        for voltage in (0., -1., Inf, NaN)
            r["solver_options"]["transducer_reference_voltage_v"]=voltage
            @test_throws Exception solve_request(r)
            @test_throws "must be finite and positive" solve_exterior_request(r,r["compiled_system"],r["compiled_system"]["regions"][1])
        end
        delete!(r["solver_options"],"transducer_reference_voltage_v")
        r["solver_options"]["symmetry"]=" ground "
        items=quantities(only(captured(r)))
        @test items["radiation_impedance_matrix"]["metadata"]["row_weights"] == [1.]
        @test items["diaphragm_velocity"]["metadata"]["physical_driver_orbit_counts"] == [1]
        # Rigid ground changes the pressure but cannot create physical area.
        r["compiled_system"]["components"][1]["parameters"]["boundary_motion_signs"]=Dict("boundary:1"=>-1)
        result=only(captured(r))
        @test result["diagnostics"]["symmetry"] == "ground"
        @test abs(only(quantities(result)["radiation_impedance_matrix"]["metadata"]["effective_volume_area_m2"])) < 1e-14
    end
end

@testset "ground image equivalence and signed patch area" begin
    a=0.1
    mesh=sphere(a,SVector(0.,0.3,0.),2;refinements=0)
    full=join_meshes(mesh,sphere(a,SVector(0.,-0.3,0.),3;refinements=0))
    real_motion=(component_id="real",tags=[2],amplitudes=[1.],motion_axis=SVector(0.,0.,1.))
    even_motion=(component_id="real",tags=[2,3],amplitudes=[1.,1.],motion_axis=SVector(0.,0.,1.))
    for convention in (POSITIVE_TIME_PHASOR,NEGATIVE_TIME_PHASOR)
        with_phasor_convention(convention) do
            pg=only(pressure_columns(mesh,[real_motion],2pi*100.,1.21,343.;symmetry=:ground))
            pf=only(pressure_columns(full,[even_motion],2pi*100.,1.21,343.))
            zg=exterior_component_force(mesh,pg,real_motion,1,Float64)
            zf=exterior_component_force(full,pf,real_motion,1,Float64)
            # Same faceted physical mesh, two independent assembled domains;
            # 1e-6 budgets finite regular image quadrature, not faceting error.
            @test abs(zg/zf-1) < 1e-6
        end
    end
    tags=[n[3] > 0 ? 2 : 9 for n in mesh.normals]
    patch=BoundaryMesh(mesh.vertices,mesh.faces,tags)
    mktemp() do path,io
        mesh_file(io,patch)
        request=driver_request(path)
        request["compiled_system"]["components"][1]["parameters"]["boundary_motion_signs"]=Dict("boundary:1"=>-1)
        request["compiled_system"]["components"][1]["parameters"]["boundary_motion_weights"]=Dict("boundary:1"=>2.)
        # Orthogonal projection of the upper octahedron is a diamond of area 2a².
        # One negative sign and weight two give S=-4a² in either environment.
        for symmetry in ("off","ground")
            request["solver_options"]["symmetry"]=symmetry
            meta=quantities(only(captured(request)))["radiation_impedance_matrix"]["metadata"]
            @test only(meta["effective_volume_area_m2"]) ≈ -4a^2 rtol=1e-14
            @test meta["effective_volume_area_zero_or_near_cancelling"] == [false]
            @test meta["row_weights"] == [1.]
            @test meta["effective_volume_area_cancellation_ratio"] ≈ [1.]
            @test meta["effective_volume_area_cancellation_relative_tolerance"] == 1e-2
        end
    end
end

@testset "area cancellation ratio and topology limits" begin
    mesh=sphere(0.1,SVector(0.,0.,0.),2;refinements=0)
    tags=[n[3] > 0 ? 2 : 3 for n in mesh.normals]
    patch=BoundaryMesh(mesh.vertices,mesh.faces,tags)
    e=(component_id="near-dipole",tags=[2,3],amplitudes=[1.,0.99],motion_axis=SVector(0.,0.,1.))
    _,meta=exterior_impedance_matrix(patch,[zeros(ComplexF64,length(mesh.vertices))],[e],
        [Dict("id"=>"near-dipole")],String[],:off)
    @test only(meta["effective_volume_area_cancellation_ratio"]) ≈ 0.01/1.99
    @test meta["effective_volume_area_zero_or_near_cancelling"] == [true]
    empty_motion=merge(e,(amplitudes=[0.,0.],))
    _,empty_meta=exterior_impedance_matrix(patch,[zeros(ComplexF64,length(mesh.vertices))],[empty_motion],
        [Dict("id"=>"near-dipole")],String[],:off)
    @test empty_meta["effective_volume_area_cancellation_ratio"] == [0.]
    @test empty_meta["effective_volume_area_zero_or_near_cancelling"] == [true]
    @test validate_exterior_transducer_surface!(join_meshes(mesh,mesh)) === nothing
    nonmanifold=BoundaryMesh(mesh.vertices,vcat(mesh.faces,mesh.faces),vcat(mesh.physical_tags,mesh.physical_tags))
    @test_throws "consistent winding" validate_exterior_transducer_surface!(nonmanifold)
    # Coincident coordinates on independently indexed faces do not weld seams.
    vertices=[mesh.vertices[v] for f in mesh.faces for v in f]
    unwelded=BoundaryMesh(vertices,[(3i-2,3i-1,3i) for i in eachindex(mesh.faces)],mesh.physical_tags)
    @test_throws "closed BEM surfaces" validate_exterior_transducer_surface!(unwelded)
    upper=[i for i in eachindex(mesh.faces) if mesh.normals[i][2] > 0]
    used=sort(unique([v for i in upper for v in mesh.faces[i]]))
    remap=Dict(v=>i for (i,v) in enumerate(used))
    half=BoundaryMesh(mesh.vertices[used],[Tuple(remap[v] for v in mesh.faces[i]) for i in upper],mesh.physical_tags[upper])
    @test validate_exterior_transducer_surface!(half,:ground) === nothing
    @test_throws "closed BEM surfaces" validate_exterior_transducer_surface!(half,:off)
    inward=BoundaryMesh(half.vertices,[(a,c,b) for (a,b,c) in half.faces],half.physical_tags)
    @test_throws "outward-wound" validate_exterior_transducer_surface!(inward,:ground)
    open=BoundaryMesh(half.vertices,half.faces[2:end],half.physical_tags[2:end])
    @test_throws "closed BEM surfaces" validate_exterior_transducer_surface!(open,:ground)
    # Ground closes the half sphere; tangential motion is even under reflection.
    motion=(component_id="half",tags=[2],amplitudes=[1.],motion_axis=SVector(0.,0.,1.))
    # Adjacent image pairs need more quadrature than the separated-body case.
    # Keep the 1e-6 comparison tolerance and converge both discretisations.
    pg=only(pressure_columns(half,[motion],2pi*100.,1.21,343.;symmetry=:ground,order=8))
    pf=only(pressure_columns(mesh,[motion],2pi*100.,1.21,343.;order=8))
    @test pg ≈ pf[used] rtol=1e-6
end

end # module
