# Hardware-free parser gate, standalone or included by runtests.jl.
module FieldOutputPointsTests

using Test, StaticArrays, JSON

if isdefined(Main, :CompiledWorkloadBundle)
    const Driver = Main.CompiledWorkloadBundle
else
    import BeatEngineCompiledCpuBundle
    const Driver = BeatEngineCompiledCpuBundle
end

field_output(id, quantity, points) = Dict(
    "id" => id, "quantity" => quantity, "options" => Dict("points_m" => points),
)

function caught_error(f)
    try
        f()
    catch exception
        return (typeof(exception), sprint(showerror, exception))
    end
    return nothing
end

@testset "field output points are typed and bitwise unchanged" begin
    outputs = JSON.parse(JSON.json(Any[
        field_output("polar", "exterior_pressure", [
            [0.1, -0.0, 1e-45], [1e38, -2.75, 3.125], [0.1, -0.0, 1e-45], [5e-324, 0, 0],
        ]),
        field_output("sphere", "exterior_pressure", [[0, 2, -3]]),
        field_output("interface", "interface_radiated_pressure", [[4, -5, 6]]),
        Dict("id" => "boundary", "quantity" => "bem_boundary_pressure"),
    ]))
    raw_points = outputs[1]["options"]["points_m"]
    for T in (Float32, Float64)
        parsed = @inferred Driver.parse_field_output_points(outputs, T)
        @test parsed isa Dict{String,Vector{SVector{3,T}}}
        @test Set(keys(parsed)) == Set(["polar", "sphere", "interface"])
        for id in keys(parsed)
            @test parsed[id] isa Vector{SVector{3,T}}
        end
        @test isequal(parsed["polar"], SVector{3,T}[
            (T(0.1), T(-0.0), T(1e-45)), (T(1e38), T(-2.75), T(3.125)),
            (T(0.1), T(-0.0), T(1e-45)), (T(5e-324), zero(T), zero(T)),
        ])
        @test issubnormal(T === Float32 ? parsed["polar"][1][3] : parsed["polar"][4][1])
        @test parsed["sphere"] == [SVector{3,T}(0, 2, -3)]
        @test parsed["interface"] == [SVector{3,T}(4, -5, 6)]
        @test outputs[1]["options"]["points_m"] === raw_points
    end
    empty_outputs = JSON.parse(JSON.json(Any[]))
    @test (@inferred Driver.parse_field_output_points(empty_outputs, Float32)) ==
          Dict{String,Vector{SVector{3,Float32}}}()
    parsed = Driver.parse_field_output_points(outputs, Float64)
    raw_points[1][1] = 9.0
    @test parsed["polar"][1][1] == 0.1
    source = read(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"), String)
    @test !occursin("SVector{3,FloatType}(FloatType.", source)
end

@testset "field output points preserve validation errors" begin
    for T in (Float32, Float64), quantity in ("exterior_pressure", "interface_radiated_pressure")
        empty_error = quantity == "exterior_pressure" ?
                      "exterior_pressure output requires options.points_m." :
                      "interface_radiated_pressure requires points_m."
        for output in (field_output("field", quantity, Any[]), Dict("id" => "field", "quantity" => quantity))
            outputs = JSON.parse(JSON.json(Any[output]))
            @test caught_error(() -> Driver.parse_field_output_points(outputs, T)) ==
                  (ErrorException, empty_error)
        end
        # Shape errors come from the SVector constructor, as before the hoist;
        # compare the type and the length, not StaticArrays' exact wording.
        for point in (Any[], Any[1, 2], Any[1, 2, 3, 4])
            outputs = JSON.parse(JSON.json(Any[field_output("field", quantity, Any[point])]))
            kind, message = caught_error(() -> Driver.parse_field_output_points(outputs, T))
            @test kind == DimensionMismatch
            @test occursin("length 3", message)
        end
    end
end

end # module
