using Test
using .BeatEngineWorkerCleanup

struct WorkerCompletionTestIO <: IO
    pending::IOBuffer
    visible::IOBuffer
    steps::Vector{Symbol}
end

Base.write(io::WorkerCompletionTestIO, byte::UInt8) = write(io.pending, byte)
Base.unsafe_write(io::WorkerCompletionTestIO, pointer::Ptr{UInt8}, size::UInt) =
    unsafe_write(io.pending, pointer, size)
function Base.flush(io::WorkerCompletionTestIO)
    push!(io.steps, :flush)
    write(io.visible, take!(io.pending))
    return nothing
end

@testset "Aggressive solve completion before reclamation" begin
    events = Dict{String,Any}[]
    steps = Symbol[]
    output = WorkerCompletionTestIO(IOBuffer(), IOBuffer(), steps)
    emit = event -> begin
        push!(events, event)
        push!(steps, :emit)
        println(output, event["type"], " ", event["solved_count"])
    end
    reclaim = () -> begin
        push!(steps, :reclaim)
        @test String(take!(output.visible)) == "completed 3\n"
        @test isempty(take!(output.pending))
    end
    @test finish_aggressive_solve!(3, emit, reclaim; output=output) === nothing
    @test steps == [:emit, :flush, :reclaim]
    @test events == [Dict("type" => "completed", "solved_count" => 3)]
end

@testset "Post-completion reclaim failure preserves the next request" begin
    events = Dict{String,Any}[]
    output = IOBuffer()
    errors = IOBuffer()
    emit = event -> begin
        push!(events, event)
        println(output, event["type"], " ", event["solved_count"])
    end
    attempts = Ref(0)
    reclaim = () -> begin
        attempts[] += 1
        attempts[] == 1 && error("synthetic reclaim failure")
    end
    @test finish_aggressive_solve!(1, emit, reclaim; output=output, error_output=errors) === nothing
    @test occursin("Post-solve reclamation failed: synthetic reclaim failure", String(take!(errors)))
    @test finish_aggressive_solve!(2, emit, reclaim; output=output, error_output=errors) === nothing
    @test attempts[] == 2
    @test events == [Dict("type" => "completed", "solved_count" => count) for count in 1:2]
    @test String(take!(output)) == "completed 1\ncompleted 2\n"
    @test isempty(take!(errors))
end

@testset "Emission failure is not swallowed" begin
    reclaimed = Ref(false)
    emit = event -> error("synthetic emit failure")
    reclaim = () -> (reclaimed[] = true)
    @test_throws ErrorException finish_aggressive_solve!(1, emit, reclaim; output=IOBuffer())
    @test !reclaimed[]
end
