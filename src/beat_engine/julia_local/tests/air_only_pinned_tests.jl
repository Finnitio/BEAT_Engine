module AirOnlyPinnedTests
# Air-only requests must stay bit-identical to BEAT Engine v0.7.0 (c82bb7a) after the porous
# region work: interior P1/P2 tubes and the tiny coupled FEM-BEM workload (condensed and
# monolithic), each pinned to a SHA-256 of every output array produced by c82bb7a on the
# reference machine (Linux x86_64, Julia 1.12, OpenBLAS, 2 Julia and 2 BLAS threads). The coupled
# dense solves depend on the BLAS thread count (1 differs from 2 or 4), and earlier test files
# leave it changed, so the test sets it.
using Test, SHA, LinearAlgebra
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))
include(joinpath(@__DIR__, "porous_tube_helpers.jl"))  # also porous_reference.jl

# --- pinned requests (also used to regenerate the digests with the c82bb7a engine) ---
function air_only_pinned_requests()
    lossy = Dict{String,Any}("bulk_loss_factor" => 0.05)
    p1, _ = porous_tube_request(lossy, 20; length_m=0.1, frequencies=(400.0, 1700.0))
    p2, _ = porous_tube_request(Dict{String,Any}(), 8; length_m=0.1, frequencies=(400.0, 1700.0), order=2)
    minus = deepcopy(p1)
    minus["solver_options"]["phasor_convention"] = "exp(-i omega t)"
    coupled(condensed) = begin
        request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
        request["solver_options"]["precision"] = "float64"
        request["solver_options"]["static_condensation"] = condensed
        push!(request["outputs"], Dict("id" => "pressure", "quantity" => "fem_nodal_pressure",
            "target_ids" => [], "options" => Dict()))
        request
    end
    return ["interior P1 lossy air tube" => p1, "interior P1 tube, exp(-i omega t)" => minus,
            "interior P2 air tube" => p2, "coupled tiny condensed" => coupled(true),
            "coupled tiny monolithic" => coupled(false)]
end

"""SHA-256 over every result's frequency and every quantity's id and raw array bytes, in order."""
function air_only_digest(results)
    context = SHA.SHA256_CTX()
    for result in results
        SHA.update!(context, reinterpret(UInt8, [Float64(result["freq_hz"])]))
        for quantity in result["quantities"]
            SHA.update!(context, Vector{UInt8}(String(quantity["id"])))
            SHA.update!(context, base64decode(quantity["values"]["content_base64"]))
        end
    end
    return bytes2hex(SHA.digest!(context))
end
# --- end pinned requests ---

# Produced by engine c82bb7a (v0.7.0) on the reference machine: include the c82bb7a
# BeatEngineCompiledDriver.jl and CompiledCoupledWorkload.jl, then this file's pinned-request
# block, and print air_only_digest(solve_coupled_workload(request).results) per request.
const AIR_ONLY_GOLDEN = Dict(
    "interior P1 lossy air tube" => "851b9d930a837092fbffa903f9dff04af9bf2036edb9451f6e0ee9949fadae14",
    "interior P1 tube, exp(-i omega t)" => "d8e952ddb1be6f7f16ab0e054365c22b17047273458ff2d828c6efb6303d47c0",
    "interior P2 air tube" => "70fd3dae647d479714b97611f83780e9a7eb3898c66577afa83cbbd6a358c16c",
    "coupled tiny condensed" => "d4c14cffa7f82ab5714c750c27ebccba062522644052dda44874f7b77cf31512",
    "coupled tiny monolithic" => "4ab65b4ced3cc7ce7ce419900ddb61b4a58065d8821e82f9a32cecfeaab298f5",
)

@testset "air-only outputs are bit-identical to v0.7.0 (c82bb7a)" begin
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(2)
    try
        for (name, request) in air_only_pinned_requests()
            digest = air_only_digest(solve_coupled_workload(request).results)
            @test (name, digest) == (name, AIR_ONLY_GOLDEN[name])
        end
    finally
        BLAS.set_num_threads(blas_threads)
    end
end

end # module
