# Hardware-free parser gate, standalone or included by the CPU/reference suites.
module FieldOutputPointsTests

using Test, StaticArrays
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))

field_output(id, quantity, points) = Dict(
    "id" => id, "quantity" => quantity, "options" => Dict("points_m" => points),
)

# The pre-hoist validation and conversion, including the earlier request-level
# finite JSON check and interface-specific shape check.
function legacy_field_points(outputs, ::Type{T}) where {T}
    BeatEngineContract.finite_json(outputs, "request.outputs")
    parsed = Dict{String,Any}()
    for output in outputs
        quantity = output["quantity"]
        quantity in ("exterior_pressure", "interface_radiated_pressure") || continue
        raw_points = get(get(output, "options", Dict{String,Any}()), "points_m", Any[])
        if quantity == "interface_radiated_pressure"
            (!isempty(raw_points) && all(point -> point isa AbstractVector && length(point) == 3 &&
                all(value -> value isa Real && !(value isa Bool) && isfinite(value), point), raw_points)) ||
                error("Interface radiation requires finite observation points with shape (point, 3).")
        else
            isempty(raw_points) && error("exterior_pressure output requires options.points_m.")
        end
        parsed[output["id"]] = [SVector{3,T}(T.(point)) for point in raw_points]
    end
    return parsed
end

function caught_error(f)
    try
        f()
    catch exception
        return (typeof(exception), sprint(showerror, exception))
    end
    return nothing
end

@testset "field output points are typed and bitwise unchanged" begin
    # JSON-decoded arrays carry Any elements, as in a real worker request.
    raw_points = JSON.parse("[[0.1, -0.0, 1e-45], [1e38, -2.75, 3.125], [0.1, -0.0, 1e-45]]")
    outputs = Any[
        field_output("polar", "exterior_pressure", raw_points),
        field_output("sphere", "exterior_pressure", JSON.parse("[[0, 2, -3]]")),
        field_output("interface", "interface_radiated_pressure", JSON.parse("[[4, -5, 6]]")),
        Dict("id" => "boundary", "quantity" => "bem_boundary_pressure"),
    ]
    for T in (Float32, Float64)
        parsed = @inferred parse_field_output_points(outputs, T)
        @test parsed isa Dict{String,Vector{SVector{3,T}}}
        @test Set(keys(parsed)) == Set(["polar", "sphere", "interface"])
        legacy = legacy_field_points(outputs, T)
        unsigned = T === Float32 ? UInt32 : UInt64
        for id in keys(parsed)
            @test parsed[id] isa Vector{SVector{3,T}}
            @test reinterpret(unsigned, collect(Iterators.flatten(parsed[id]))) ==
                  reinterpret(unsigned, collect(Iterators.flatten(legacy[id])))
        end
        @test isequal(parsed["polar"], SVector{3,T}[
            (T(0.1), T(-0.0), T(1e-45)), (T(1e38), T(-2.75), T(3.125)),
            (T(0.1), T(-0.0), T(1e-45)),
        ])
        @test parsed["sphere"] == [SVector{3,T}(0, 2, -3)]
        @test parsed["interface"] == [SVector{3,T}(4, -5, 6)]
        @test outputs[1]["options"]["points_m"] === raw_points
    end
    @test (@inferred parse_field_output_points(Any[], Float32)) == Dict{String,Vector{SVector{3,Float32}}}()
    parsed = parse_field_output_points(outputs, Float64)
    raw_points[1][1] = 9.0
    @test parsed["polar"][1][1] == 0.1
end

@testset "field output points preserve validation errors" begin
    for T in (Float32, Float64), quantity in ("exterior_pressure", "interface_radiated_pressure")
        for points in (Any[], Any[Any[]], Any[Any[1, 2]], Any[Any[1, 2, 3, 4]],
                       Any[Any[NaN, 0, 1]], Any[Any[0, Inf, 1]], Any[Any[0, 1, -Inf]])
            outputs = Any[
                Dict("id" => "boundary", "quantity" => "bem_boundary_pressure"),
                field_output("field", quantity, points),
            ]
            before = caught_error(() -> legacy_field_points(outputs, T))
            after = caught_error(() -> parse_field_output_points(outputs, T))
            @test before !== nothing
            @test after == before
        end
        missing = Any[Dict("id" => "field", "quantity" => quantity)]
        @test caught_error(() -> parse_field_output_points(missing, T)) ==
              caught_error(() -> legacy_field_points(missing, T))
    end
    nonfinite = Any[field_output("field", "exterior_pressure", Any[Any[0, NaN, 1]])]
    @test caught_error(() -> parse_field_output_points(nonfinite, Float32)) ==
          (ErrorException, "BEAT contract request.outputs[0].options.points_m[0][1]: must contain finite JSON values")
    malformed_interface = Any[field_output("field", "interface_radiated_pressure", Any[Any[0, 1]])]
    @test caught_error(() -> parse_field_output_points(malformed_interface, Float32)) ==
          (ErrorException, "Interface radiation requires finite observation points with shape (point, 3).")
end

end # module
