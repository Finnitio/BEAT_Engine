# Opt-in refinement of a CPU ComplexF32 LU. The default fused solve never calls this.
# A Float64 matvec of rounded coefficients repairs factorization/solve error;
# it cannot repair assembly error. Callers may supply an independently assembled
# residual operator, but must include that assembly's time and memory in measurements.
function beat_refine_cpu_lu(matrix::AbstractMatrix{ComplexF32}, rhs::AbstractVecOrMat{ComplexF32};
        residual_matrix=matrix, residual_rhs=rhs, max_steps::Int=3, rtol::Real=1e-10)
    1 <= max_steps <= 3 || throw(ArgumentError("max_steps must be in 1:3"))
    isfinite(rtol) && rtol > 0 || throw(ArgumentError("rtol must be finite and positive"))
    n = size(matrix,1)
    size(matrix,2) == n || throw(DimensionMismatch("matrix must be square"))
    size(residual_matrix) == size(matrix) || throw(DimensionMismatch("residual matrix differs"))
    b = rhs isa AbstractMatrix ? rhs : reshape(rhs,:,1)
    br = residual_rhs isa AbstractMatrix ? residual_rhs : reshape(residual_rhs,:,1)
    size(b,1) == n && size(br) == size(b) || throw(DimensionMismatch("right-hand sides differ"))
    A64, b64 = ComplexF64.(residual_matrix), ComplexF64.(br)
    F = lu!(copy(matrix))
    x = ComplexF64.(F \ b)
    r = similar(x)
    delta = similar(b,ComplexF32)
    rhs_norms = [norm(view(b64,:,j)) for j in axes(b64,2)]
    residuals = function(x)
        copyto!(r,b64)
        mul!(r,A64,x,-1.,1.)
        [rhs_norms[j] == 0 ? (norm(view(r,:,j)) == 0 ? 0. : Inf) :
            norm(view(r,:,j))/rhs_norms[j] for j in axes(r,2)]
    end
    errors = residuals(x)
    history = [copy(errors)]
    steps = 0
    status = :step_limit
    for step in 1:max_steps
        if all(<=(rtol),errors)
            status = :converged
            break
        end
        delta .= r
        for j in axes(delta,2)
            errors[j] <= rtol && fill!(view(delta,:,j),0)
        end
        ldiv!(F,delta)
        next = x + delta
        next_errors = residuals(next)
        # Retain the last accepted solution independently for every excitation.
        improved = false
        for j in axes(next,2)
            if isfinite(next_errors[j]) && next_errors[j] < errors[j]
                copyto!(view(x,:,j),view(next,:,j))
                errors[j] = next_errors[j]
                improved = true
            end
        end
        steps = step
        errors = residuals(x)
        push!(history,copy(errors))
        if !improved
            status = :stagnated
            break
        end
    end
    all(<=(rtol),errors) && (status = :converged)
    solution = ComplexF32.(x)
    returned_errors = residuals(ComplexF64.(solution))
    output = rhs isa AbstractMatrix ? solution : vec(solution)
    working_status = any(!isfinite, errors) ? :nonfinite : status
    # Only call the returned pressure converged when its rounded values meet
    # the requested tolerance. Float64 working convergence can be lost here.
    status = if any(!isfinite, returned_errors)
        :nonfinite
    elseif all(<=(rtol), returned_errors)
        :converged
    elseif working_status === :converged
        :rounding_limited
    else
        working_status
    end
    return output, (;steps,status,working_status,rtol,working_relative_residuals=errors,
        returned_relative_residuals=returned_errors,history)
end
