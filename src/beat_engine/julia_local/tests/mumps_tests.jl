# MUMPS Schur backend tests. MUMPS_seq_jll ships only with the macOS Metal environment, so run
# this under that project. Everything here runs on the CPU; no Metal device is needed:
#
#     julia --threads=2 --project=src/beat_engine/julia_metal src/beat_engine/julia_local/tests/mumps_tests.jl
using Test, StaticArrays, LinearAlgebra

ENV["BLAB_RUN_COUPLED_REFERENCE"] = "1"
ENV["BLAB_COUPLED_QUADRATURE_ORDER"] = "1"
ENV["BLAB_COUPLED_SINGULAR_ORDER"] = "1"

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))

@testset "MUMPS ships with this environment" begin
    @test !isnothing(Base.locate_package(BeatEngineCoupledCondensed.BeatEngineMumps.MUMPS_SEQ_PKGID))
    @test !isnothing(Base.locate_package(BeatEngineCoupledCondensed.BeatEngineMumps.OPENBLAS32_PKGID))
end

@testset "MUMPS LP64 BLAS selection" begin
    Mumps = BeatEngineCoupledCondensed.BeatEngineMumps
    ilp64_before = [(lib.libname, lib.interface) for lib in BLAS.get_config().loaded_libs if lib.interface == :ilp64]
    library = Mumps.mumps_library()
    @test library.available
    @test library.blas in ("accelerate", "openblas")
    # The forward fills only the LP64 slots MUMPS calls; Julia's own ILP64 BLAS is unchanged.
    @test [(lib.libname, lib.interface) for lib in BLAS.get_config().loaded_libs if lib.interface == :ilp64] ==
          ilp64_before
    accelerate = Sys.isapple() ? Base.Libc.Libdl.dlopen(Mumps.ACCELERATE_FRAMEWORK; throw_error=false) : nothing
    new_lapack = accelerate !== nothing &&
                 Base.Libc.Libdl.dlsym(accelerate, "zgemm\$NEWLAPACK"; throw_error=false) !== nothing
    preference = Mumps.mumps_blas_preference()
    if Sys.ARCH === :aarch64 && new_lapack && preference == "auto"
        # Where Accelerate's new interface exists, `auto` uses it unless its self-test failed,
        # in which case the loader recorded why and fell back to OpenBLAS32.
        @test library.blas == "accelerate" || !isempty(Mumps.ACCELERATE_SELF_TEST_FALLBACK[])
    elseif preference == "accelerate"
        @test library.blas == "accelerate"
    elseif preference == "openblas"
        @test library.blas == "openblas"
    end
    if library.blas == "accelerate"
        # Accelerate's thread-safe "new LAPACK" entry points, never its legacy ones.
        for name in ("zgemm", "ztrsm", "zgetrf", "zpotrf", "zsytrf")
            new_entry = Base.Libc.Libdl.dlsym(accelerate, name * "\$NEWLAPACK")
            legacy_entry = Base.Libc.Libdl.dlsym(accelerate, name * "_")
            @test BLAS.lbt_get_forward(name * "_", :lp64) == new_entry
            @test BLAS.lbt_get_forward(name * "_", :lp64) != legacy_entry
        end
        # Complex-returning and single-precision functions keep OpenBLAS32 (return-convention ABI).
        for name in ("zdotc_", "zdotu_", "cdotc_", "sdot_")
            @test BLAS.lbt_get_forward(name, :lp64) != something(
                Base.Libc.Libdl.dlsym(accelerate, chop(name) * "\$NEWLAPACK"; throw_error=false), C_NULL)
        end
        # A complex-returning call through the LP64 table still returns the right value.
        zdotc = BLAS.lbt_get_forward("zdotc_", :lp64)
        x = ComplexF64[1 + 2im, 3 - 1im]
        y = ComplexF64[2 - 1im, 1 + 1im]
        @test ccall(zdotc, ComplexF64, (Ref{Int32}, Ptr{ComplexF64}, Ref{Int32}, Ptr{ComplexF64}, Ref{Int32}),
            2, x, 1, y, 1) ≈ dot(x, y)
    end
    @test withenv(Mumps.mumps_blas_preference, "BLAB_MUMPS_BLAS" => nothing) == "auto"
    @test withenv(Mumps.mumps_blas_preference, "BLAB_MUMPS_BLAS" => "OpenBLAS") == "openblas"
    @test_throws "BLAB_MUMPS_BLAS" withenv(Mumps.mumps_blas_preference, "BLAB_MUMPS_BLAS" => "mkl")
end

include(joinpath(@__DIR__, "coupled_mumps_tests.jl"))

@testset "MUMPS precompile state is released and reloads" begin
    mumps = BeatEngineCoupledCondensed.BeatEngineMumps
    library = mumps.mumps_library()
    @test library.available
    solver = mumps.MumpsSchurSolver(library; threads=1)
    @test solver.initialized
    mumps.reset_precompile_state!()
    @test !solver.initialized
    @test mumps.LIBRARY[] === nothing
    @test isempty(mumps.LIVE_SOLVERS)
    @test !mumps.ATEXIT_REGISTERED[]
    reloaded = mumps.mumps_library()
    @test reloaded.available
    @test reloaded.version == library.version
    @test mumps.ATEXIT_REGISTERED[]
    mumps.reset_precompile_state!()
end
