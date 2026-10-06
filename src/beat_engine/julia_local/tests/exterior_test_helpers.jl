function sphere(radius, centre, tag; refinements=3)
    vertices = SVector{3,Float64}[(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    faces = [(1,3,5), (3,2,5), (2,4,5), (4,1,5), (3,1,6), (2,3,6), (4,2,6), (1,4,6)]
    for _ in 1:refinements
        midpoints = Dict{Tuple{Int,Int},Int}()
        midpoint(a,b) = get!(midpoints, minmax(a,b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        refined = NTuple{3,Int}[]
        for (a,b,c) in faces
            ab,bc,ca = midpoint(a,b), midpoint(b,c), midpoint(c,a)
            append!(refined, [(a,ab,ca), (ab,b,bc), (ca,bc,c), (ab,bc,ca)])
        end
        faces = refined
    end
    BoundaryMesh([centre + radius*v for v in vertices], faces, fill(tag, length(faces)))
end

function join_meshes(meshes...)
    vertices = SVector{3,Float64}[]
    faces = NTuple{3,Int}[]
    tags = Int[]
    for mesh in meshes
        offset = length(vertices)
        append!(vertices, mesh.vertices)
        append!(faces, [face .+ offset for face in mesh.faces])
        append!(tags, mesh.physical_tags)
    end
    BoundaryMesh(vertices, faces, tags)
end

function pressure_columns(mesh, excitations, omega, rho, c; symmetry=:off, order=4)
    p1,dp0 = build_p1_space(mesh),build_dp0_space(mesh)
    rule = order == 4 ? triangle_rule(Float64,4) : tensor_triangle_rule(Float64,order)
    operators = assemble_regular_galerkin_operators(mesh,p1,dp0,omega/c,rule;
        skip_singular=false,singular_order=order,backend=:cpu,symmetry_mode=symmetry)
    ipp = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:p1;symmetry_mode=symmetry)
    ipq = assemble_l2_identity_matrix(mesh,p1,dp0,rule,:p1,:dp0;symmetry_mode=symmetry)
    system = build_burton_miller_neumann_cpu_system(operators,ipp,ipq,omega/c)
    [solve_burton_miller_neumann_cpu_system(system,
        exterior_neumann(mesh,e,rho,omega),Float64) for e in excitations]
end
