#==========================================================================================
# EconMeth.Test
# Ryan Gilland
==========================================================================================#
import Statistics as stats
import Distributions as dist
import Random as random

# Load EconMeth library
include("../src/EconMeth.jl")
import .EconMeth as em

# Computes u(c)
function u(c, gamma)
    if gamma == 1
        return log(c)
    else
        return (c^(1 - gamma)) / (1 - gamma)
    end
end

# Computes u_c(c)
function u_c(c, gamma)
    return c^(-gamma)
end

# Computes u_c(c)^(--1)
function u_c_inv(c, gamma)
    return c^(-1/gamma)
end

# Computes the expected value of u_c(c_1)
function Eu_c(states, y, c, gamma, P)
    N = length(states)

    i_y = findfirst(state -> state == y, states)
    P_y = P[i_y, :]
    return P_y' * u_c.(c, gamma)
end

# Computes the Euler equation
function cEuler(c, c1, y, R, beta, gamma, states, P)
    lhs = u_c(c, gamma)
    rhs = beta * R * Eu_c(states, y, c1, gamma, P)
    return lhs - rhs
end

# Computes the Euler equation
function aEuler(a, a1, a2, y, R, beta, gamma, states, P)
    c = R*a + y - a1
    c1 = R*a1 .+ states .- a2
    if c <= 0.0 || minimum(c1) <= 0.0
        return Inf
    else
        lhs = u_c(c, gamma)
        rhs = beta * R * Eu_c(states, y, c1, gamma, P)
        return lhs - rhs
    end
end

# Policy function iteration using interpolation
function pfi_asset_interpolation(
    f,
    args,
    grid_length,
    grid_min,
    grid_max,
    shape,
    states;
    max_iterations=1000,
    solver_iterations=1000,
    value_tolerance=1e-4,
    solver_tolerance=1e-6,
    print_output=false
    )

    state_length = length(states)

    init_grid = Array{Union{Float64, Missing}}(undef, grid_length)
    init_grid[1] = grid_min
    init_grid[grid_length] = grid_max
    for i = 2:grid_length-1
        init_grid[i] = grid_distance(1, grid_length, grid_min, grid_max, i, shape)
    end

    guess_grid = Array{Union{Float64, Missing}}(undef, (grid_length, state_length))
    solve_grid = Array{Union{Float64, Missing}}(undef, (grid_length, state_length))
    next_guess_grid = Array{Union{Float64, Missing}}(undef, (grid_length, state_length))
    for j = 1:state_length
        guess_grid[:, j] = init_grid
    end
    solve_grid[:, :] = guess_grid[:, :]
    next_guess_grid[:, :] = guess_grid[:, :]

    for iteration = 1:max_iterations
        if print_output
            println("Iteration: ", iteration)
        end
        for i = 1:grid_length
            for j = 1:state_length
                init_value = init_grid[i]
                init_guess = init_grid[1]
                next_guess = next_guess_grid[1, :]
                f_result = f(init_value, init_guess, next_guess, states[j], args...)
                if f_result >= 0.0
                    solve_grid[i, j] = init_grid[1]
                else
                    init_guess = guess_grid[i, j]
                    value = bisection(
                        f,
                        (states[j], args...),
                        init_value,
                        guess_grid[:, j],
                        grid_length,
                        grid_min,
                        init_guess,
                        grid_max,
                        next_guess_grid;
                        max_iterations=solver_iterations,
                        tolerance=solver_tolerance
                    )
                    if typeof(value) != Missing
                        solve_grid[i, j] = value
                    else
                        if print_output
                            println("Solver failed to converge.")
                        end
                    end
                end
            end
        end
        guess_grid[:, :] = solve_grid[:, :]
        # Compute maximum error of a' - a''
        error = maximum(abs.(skipmissing(guess_grid .- next_guess_grid)))
        # If error <= tolerance, complete, else set a'' = a' and repeat until max
        # iterations reached
        if error <= value_tolerance
            println("Converged in $iteration iteration(s).")
            break
        elseif iteration == max_iterations
            println(
                "Interpolation failed to converge with a max absolute error of \
                $(round(error, digits=8))."
            )
            if print_output
                println("Component level errors:")
                for i = 1:grid_length
                    println(round.(abs.(skipmissing(guess_grid[i, :] .- next_guess_grid[i, :])), digits=8))
                end
            end
        else
            if print_output
                println(
                    "\nError $(round(error, digits=8)) outside of tolerance, updating Ay2."
                )
            end
            next_guess_grid[:, :] = guess_grid[:, :]
        end
    end

    return init_grid, guess_grid
end

