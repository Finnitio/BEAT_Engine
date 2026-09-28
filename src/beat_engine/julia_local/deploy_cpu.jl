# Host-array counterpart of deploy_cuda_gmres, with the same warm-start and residual contract.
function deploy_cpu_gmres(
    apply_operator,
    right_hand_side;
    tolerance,
    max_iterations,
    initial_guess=nothing,
)
    T = typeof(real(zero(eltype(right_hand_side))))
    max_iterations > 0 || error("Deploy speaker ROM GMRES iteration limit must be positive.")
    tolerance > zero(T) || error("Deploy speaker ROM GMRES tolerance must be positive.")
    initial_guess === nothing || length(initial_guess) == length(right_hand_side) || error(
        "Deploy speaker ROM GMRES initial guess size mismatch.",
    )
    rhs_norm = norm(right_hand_side)
    if initial_guess === nothing && rhs_norm <= eps(T)
        return (
            zeros(eltype(right_hand_side), length(right_hand_side)),
            0,
            zero(T),
            T[],
            0,
            zero(T),
        )
    end
    residual_vector = nothing
    operator_applications = 0
    initial_guess_scale = one(eltype(right_hand_side))
    if initial_guess === nothing
        residual_vector = copy(right_hand_side)
    else
        initial_action = apply_operator(initial_guess)
        operator_applications += 1
        action_norm_squared = real(dot(initial_action, initial_action))
        if action_norm_squared > eps(T)
            initial_guess_scale = dot(initial_action, right_hand_side) / action_norm_squared
        end
        residual_vector = copy(right_hand_side)
        residual_vector .-= initial_guess_scale .* initial_action
    end
    beta = norm(residual_vector)
    residual_scale = max(rhs_norm, eps(T))
    initial_relative_residual = beta / residual_scale
    if beta <= eps(T) || initial_relative_residual <= tolerance
        solution = initial_guess === nothing ?
                   zeros(eltype(right_hand_side), length(right_hand_side)) :
                   initial_guess_scale .* initial_guess
        return (solution, 0, T(initial_relative_residual), T[], operator_applications, T(initial_relative_residual))
    end
    basis = zeros(eltype(right_hand_side), length(right_hand_side), max_iterations + 1)
    hessenberg = zeros(Complex{T}, max_iterations + 1, max_iterations)
    residual_history = T[]
    solution = nothing
    used_iterations = 0
    final_coefficients = Complex{T}[]
    view(basis, :, 1) .= residual_vector ./ beta
    for iteration in 1:max_iterations
        work = apply_operator(view(basis, :, iteration))
        operator_applications += 1
        for previous in 1:iteration
            coefficient = dot(view(basis, :, previous), work)
            hessenberg[previous, iteration] = coefficient
            work .-= coefficient .* view(basis, :, previous)
        end
        next_norm = norm(work)
        hessenberg[iteration + 1, iteration] = next_norm
        if next_norm > eps(T)
            view(basis, :, iteration + 1) .= work ./ next_norm
        end
        small_rhs = zeros(Complex{T}, iteration + 1)
        small_rhs[1] = beta
        coefficients = view(hessenberg, 1:(iteration + 1), 1:iteration) \ small_rhs
        residual = norm(
            small_rhs - view(hessenberg, 1:(iteration + 1), 1:iteration) * coefficients,
        ) / residual_scale
        push!(residual_history, T(residual))
        used_iterations = iteration
        final_coefficients = coefficients
        (residual <= tolerance || abs(hessenberg[iteration + 1, iteration]) <= eps(T)) && break
    end
    correction = view(basis, :, 1:used_iterations) * final_coefficients
    if initial_guess === nothing
        solution = correction
    else
        solution = initial_guess_scale .* initial_guess
        solution .+= correction
    end
    return (
        solution,
        used_iterations,
        last(residual_history),
        residual_history,
        operator_applications,
        T(initial_relative_residual),
    )
end
