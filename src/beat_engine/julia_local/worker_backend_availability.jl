function worker_backend_availability()
    backends = Dict{String,Any}("cpu" => Dict("available" => true, "reason" => ""))
    for (name, accelerator) in (
        ("cuda", BeatEngineCore.CUDA_MODULE),
        ("rocm", BeatEngineCore.AMDGPU_MODULE),
        ("metal", BeatEngineCore.METAL_MODULE),
    )
        available, reason = try
            if accelerator === nothing
                (false, "Package is not loaded in this Julia environment.")
            else
                functional = accelerator.functional()
                if functional && name == "rocm"
                    functional = accelerator.functional(:rocblas) && accelerator.functional(:rocsolver)
                end
                (functional, functional ? "" : "Runtime/device or required solver libraries are not functional.")
            end
        catch exception
            (false, sprint(showerror, exception))
        end
        backends[name] = Dict{String,Any}("available" => available, "reason" => reason,
            "phasor_conventions" => name == "rocm" ? [NEGATIVE_TIME_PHASOR] : [NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR])
        # Porous region materials are qualified on CPU and CUDA only; the others announce air so
        # clients refuse such requests before submission (the driver refuses them as well).
        name in ("rocm", "metal") && (backends[name]["fem_region_material_models"] = ["air"])
    end
    return backends
end

