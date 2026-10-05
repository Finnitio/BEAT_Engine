# Resolve the real compiled-driver closures without a GPU or a solve.
# Standalone with julia_local, or included by runtests.jl.
module CompiledDriverClosureTests

using Test

# Reuse the bundle already loaded by the workload suite when included there.
if isdefined(Main, :CompiledWorkloadBundle)
    const Driver = Main.CompiledWorkloadBundle
else
    import BeatEngineCompiledCpuBundle
    const Driver = BeatEngineCompiledCpuBundle
end

include(joinpath(@__DIR__, "..", "..", "julia_engine", "CompiledDriverClosures.jl"))

@testset "compiled driver closures match Metal precompile inventory" begin
    # The shared resolver uses only(matches), so missing or ambiguous captured
    # fields fail this hardware-free gate before a Metal package build.
    closures = compiled_driver_closure_types(Driver)
    @test keys(closures) == (:producer, :neumann, :field_values)
    for closure in values(closures)
        T = Base.unwrap_unionall(closure)
        @test T <: Function
        @test parentmodule(T) === Driver
    end
end

end
