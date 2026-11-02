# Region materials and region seams of the bounded-region FEM mesh (ADR 0005), used by
# `aggregate_fem_domains` and the result provenance in BeatEngineCompiledDriver.jl. Included
# by the driver after `BeatEngineCoupled` (materials, mesh helpers).

"""
Normalized material of every bounded region in request order (`region_material_wire`), for air-only
requests too: `[{"region_id", "material"}]`, empty without bounded regions.
"""
function region_materials_provenance(request)
    regions = get(get(request, "compiled_system", Dict{String,Any}()), "regions", [])
    return [Dict{String,Any}("region_id" => String(region["id"]), "material" => region_material_wire(
                parse_region_material(get(region, "loss_model", Dict{String,Any}()), Float64)))
            for region in regions if String(get(region, "kind", "")) == "bounded_air"]
end

# Corner pairs of the six edge nodes (5:10) of a quadratic tetrahedron (see `p2_shape_functions`).
const QUADRATIC_TETRAHEDRON_EDGES = ((1, 2), (2, 3), (3, 1), (1, 4), (4, 3), (2, 4))

"""Exterior faces (sorted full-mesh vertex triples) of the tetrahedra with `tags`, mapped to their tetrahedron."""
function _selection_exterior_faces(mesh::VolumeMesh, tags)
    owner, count = Dict{NTuple{3,Int},Int}(), Dict{NTuple{3,Int},Int}()
    for (index, tetrahedron) in enumerate(mesh.tetrahedra)
        mesh.tetra_physical_tags[index] in tags || continue
        for face in BeatEngineCoupled._tetrahedron_faces(tetrahedron)
            count[face] = get(count, face, 0) + 1
            owner[face] = index
        end
    end
    return Dict(face => owner[face] for (face, n) in count if n == 1)
end

"""
    find_region_seams(records)

Classify the faces shared by two bounded regions on the same mesh (an exterior tetrahedron
face of both selections, on different tetrahedra). A side is *assigned* when a mesh triangle on
the face carries a tag that the region assigns to a boundary. Unassigned on both sides: a region
seam (interior, shared nodes). Assigned on both sides: a thin rigid partition (today's
duplicated nodes). Assigned on one side only: an error naming both regions.

`seam_links` lists `(i, j, nodes)` per seam: the full-mesh nodes (corners and, for quadratic
tetrahedra, edge nodes) of the seam faces between records `i` and `j`. Only these region-pair
copies are joined (`seam_node_owners`); a third region touching such a node across a partition
keeps its own copy.
"""
function find_region_seams(records)
    seam_faces = [Set{NTuple{3,Int}}() for _ in records]
    seam_links = Tuple{Int,Int,Set{Int}}[]
    seams = NamedTuple[]
    for i in eachindex(records), j in (i + 1):length(records)
        a, b = records[i], records[j]
        a.mesh_id == b.mesh_id || continue
        shared = [face for (face, tetrahedron) in a.exterior_faces
                  if get(b.exterior_faces, face, tetrahedron) != tetrahedron]
        isempty(shared) && continue
        pair_faces = NTuple{3,Int}[]
        for face in shared
            assigned_a, assigned_b = face in a.assigned_faces, face in b.assigned_faces
            if assigned_a != assigned_b
                error("A face shared by bounded regions $(repr(a.id)) and $(repr(b.id)) on mesh " *
                      "$(repr(a.mesh_id)) is assigned only in region $(repr(assigned_a ? a.id : b.id)). " *
                      "Assign it in both regions (a thin rigid partition) or in neither (a region seam).")
            end
            assigned_a || push!(pair_faces, face)
        end
        isempty(pair_faces) && continue
        union!(seam_faces[i], pair_faces)
        union!(seam_faces[j], pair_faces)
        corners = Set{Int}(vertex for face in pair_faces for vertex in face)
        nodes = copy(corners)
        if is_quadratic(a.full_mesh)
            edges = Set{Tuple{Int,Int}}(minmax(face[l], face[r]) for face in pair_faces
                                        for (l, r) in ((1, 2), (2, 3), (1, 3)))
            for record in (a, b), index in record.tetrahedron_indices
                quadratic = record.full_mesh.quadratic_tetrahedra[index]
                for (offset, (l, r)) in enumerate(QUADRATIC_TETRAHEDRON_EDGES)
                    minmax(quadratic[l], quadratic[r]) in edges && push!(nodes, quadratic[4 + offset])
                end
            end
        end
        push!(seam_links, (i, j, nodes))
        push!(seams, (region_ids=[a.id, b.id], mesh_id=a.mesh_id, face_count=length(pair_faces),
                      node_count=length(nodes)))
    end
    return (; seams, seam_faces, seam_links)
end

"""
    seam_node_owners(seam_links)

Union-find over region-local node copies `(record index, full-mesh node)`, joined only along
the seam faces of each region pair. Maps every linked copy to its class representative: the
copy in the earliest record, so it is aggregated before the copies that reuse it.
"""
function seam_node_owners(seam_links)
    parent = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    function root(copy)
        p = get!(parent, copy, copy)
        p == copy && return copy
        r = root(p)
        parent[copy] = r
        return r
    end
    for (i, j, nodes) in seam_links, node in nodes
        a, b = root((i, node)), root((j, node))
        a == b || (parent[max(a, b)] = min(a, b))
    end
    return Dict(copy => root(copy) for copy in collect(keys(parent)))
end

"""
The ROCm and Metal coupled FEM caches scale mass rows by a per-node bulk loss factor, which is
single-valued at a seam node only when both regions agree. Interior, CPU and CUDA coupled
solves use per-region element operators and accept any factors.
"""
function require_uniform_seam_bulk_loss(fem_domains)
    factor(id) = only(domain.material.bulk_loss_factor for domain in fem_domains.domains if domain.id == id)
    for seam in fem_domains.region_seams
        a, b = seam.region_ids
        factor(a) == factor(b) || error(
            "Region seam between $(repr(a)) and $(repr(b)) joins different bulk loss factors; " *
            "coupled FEM-BEM solves support region seams only between regions with equal bulk loss.")
    end
    return nothing
end

"""Wire form of the region seams (diagnostic `fem_region_seams`)."""
region_seams_wire(seams) = [Dict{String,Any}("region_ids" => seam.region_ids, "mesh_id" => seam.mesh_id,
    "face_count" => seam.face_count, "node_count" => seam.node_count) for seam in seams]

"""
Nodal values in the wire layout: one row per region-local vertex, regions contiguous in
request order, so a seam node appears once in each region that contains it.
"""
region_node_values(values, domains) = reduce(vcat, (values[domain.vertex_indices] for domain in domains))

"""
    append_region_vertices!(vertices, bulk_loss_factor_by_vertex, record, region_index, seam_owner, aggregated_by_owner)

Append record `region_index`'s vertices to the aggregated mesh and return its region-local ->
aggregated vertex map. A seam-linked copy (`seam_node_owners`) reuses the node already
aggregated for its class; every other vertex gets a node of its own.
"""
function append_region_vertices!(vertices, bulk_loss_factor_by_vertex, record, region_index,
                                 seam_owner, aggregated_by_owner)
    selection = record.selection
    vertex_indices = Vector{Int}(undef, length(selection.mesh.vertices))
    for (local_index, full) in enumerate(record.full_vertices)
        owner = get(seam_owner, (region_index, full), nothing)
        existing = owner === nothing ? 0 : get(aggregated_by_owner, owner, 0)
        if existing > 0
            vertex_indices[local_index] = existing
            continue
        end
        push!(vertices, selection.mesh.vertices[local_index])
        push!(bulk_loss_factor_by_vertex, record.bulk_loss_factor)
        vertex_indices[local_index] = length(vertices)
        owner === nothing || (aggregated_by_owner[owner] = length(vertices))
    end
    return vertex_indices
end
