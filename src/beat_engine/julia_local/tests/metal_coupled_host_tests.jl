# Pure policy/projection checks: no Metal import, GPU, mesh or solver required.
module MetalCoupledHostTests
using Test, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineMetalCoupledHost.jl"))

@testset "coupled Metal assembly policy" begin
    @test resolve_metal_coupled_bem_assembly("auto").mode == :combined
    @test resolve_metal_coupled_bem_assembly(" COMBINED ").mode == :combined
    @test resolve_metal_coupled_bem_assembly(:operators, "host singular mode").fallback_reason === nothing
    for reason in ("host singular mode", "unsupported quadrature", "diagnostics")
        @test resolve_metal_coupled_bem_assembly(:auto, reason) == (mode=:operators, fallback_reason=reason)
        @test_throws ErrorException resolve_metal_coupled_bem_assembly(:combined, reason)
    end
    @test_throws ErrorException resolve_metal_coupled_bem_assembly(:typo)
end

@testset "retained C host projection and ownership" begin
    a = ComplexF32[1+im 2; 3im 4-im]
    c = ComplexF32[1+2im -3im 2; -1 4+im .5im]
    q = sparse(Float32[1 0; 0 -.5; .25 1])
    motion = ComplexF32[1+im; -2im; .5;;]
    prescribed = hcat(motion, conj.(motion))
    blocks = project_metal_coupled_host_blocks((; a, c), q, motion, prescribed)
    @test blocks.bem_lhs == a
    @test blocks.bem_lhs !== a
    @test blocks.bem_interface_block ≈ c * Matrix(q)
    @test blocks.bem_motion_block ≈ c * motion
    @test blocks.bem_prescribed_rhs ≈ -c * prescribed
    # The baseline returns rhs_operator = -C; check all three sign conventions.
    rhs_operator = -c
    @test blocks.bem_interface_block ≈ -(rhs_operator * q)
    @test blocks.bem_motion_block ≈ -(rhs_operator * motion)
    @test blocks.bem_prescribed_rhs ≈ rhs_operator * prescribed
    fill!(a, 0); fill!(c, 0)
    @test !iszero(norm(blocks.bem_lhs))
    @test !iszero(norm(blocks.bem_interface_block))
    empty_blocks = project_metal_coupled_host_blocks((a=a, c=c), spzeros(Float32, 3, 0),
        zeros(ComplexF32, 3, 0), zeros(ComplexF32, 3, 0))
    @test size(empty_blocks.bem_interface_block) == (2, 0)
    @test empty_blocks.bem_motion_block === nothing
    @test size(empty_blocks.bem_prescribed_rhs) == (2, 0)
    @test_throws ErrorException project_metal_coupled_host_blocks((; a, c), spzeros(Float32, 2, 1), motion, prescribed)
end

# Execute the untyped gather kernels on ordinary arrays with a CPU thread-index
# stub. Load only these two function definitions, avoiding Metal imports/macros.
const thread_index = Ref(Int32(1))
thread_position_in_grid_1d() = thread_index[]
function load_gather_definition(filename, name)
    source = Meta.parseall(read(joinpath(@__DIR__, "..", "src", filename), String))
    definition = only(filter(source.args) do node
        node isa Expr && node.head == :function && node.args[1] isa Expr &&
            node.args[1].head == :call && node.args[1].args[1] == name
    end)
    Core.eval(@__MODULE__, definition)
end
load_gather_definition("BeatEngineMetalCoupledBurtonMiller.jl", :_metal_coupled_flux_gather_kernel!)
load_gather_definition("BeatEngineMetalBurtonMiller.jl", :_metal_fused_lhs_gather_kernel!)

@testset "combined gather cell ownership and component indexing on CPU" begin
    elements = Int32[1, 2]
    positions = Int32[1, 2]
    offsets = Int32[1, 2, 4, 6, 7]
    incidents = Int32[1, 1, 2, 1, 2, 2]
    locals = Int32[1, 2, 1, 3, 2, 3]
    dp0 = Int32[2, 1] # Check a non-identity face-to-DP0 mapping.
    nodes = Int32[1, 2, 3, 4]
    packed = Int32[1, 2, 5, 3, 6, 7]
    blocks = Float32.(1:96)
    a = zeros(ComplexF32, 4, 4)
    rhs_operator = zeros(ComplexF32, 4, 2)
    for i in 1:8
        thread_index[] = Int32(i)
        _metal_coupled_flux_gather_kernel!(rhs_operator, blocks, elements, positions, offsets,
            incidents, locals, dp0, Int32(2), Int32(1), Int32(2), Int32(4), Int32(4))
    end
    for i in 1:16
        thread_index[] = Int32(i)
        _metal_fused_lhs_gather_kernel!(a, blocks, positions, offsets, incidents, locals,
            nodes, offsets, packed, Int32(1), Int32(4), Int32(2), Int32(4), Int32(4))
    end
    expected_a = zeros(ComplexF32, 4, 4)
    expected_rhs = zeros(ComplexF32, 4, 2)
    triangle_nodes = ((1, 2, 3), (2, 3, 4))
    # Independent local-block scatter reference for two adjacent triangles.
    for trial in 1:2, test in 1:2
        pair = test + 2 * (trial - 1)
        for r in 1:3
            row = triangle_nodes[test][r]
            expected_rhs[row, dp0[trial]] += complex(blocks[pair + (r + 17) * 4], blocks[pair + (r + 20) * 4])
            for c in 1:3
                column = triangle_nodes[trial][c]
                component = r + 3 * (c - 1)
                expected_a[row, column] += complex(blocks[pair + (component - 1) * 4], blocks[pair + (component + 8) * 4])
            end
        end
    end
    @test a == expected_a
    @test rhs_operator == expected_rhs
end

end
