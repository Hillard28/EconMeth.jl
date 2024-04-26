#==========================================================================================
# EconMeth.Optimize.Root_Solve
# Ryan Gilland
==========================================================================================#
function bisection(
    f,
    args,
    init_value,
    guess_grid,
    grid_length,
    guess_min,
    guess_mid,
    guess_max,
    next_guess_grid;
    max_iterations=1000,
    tolerance=1e-6
    )
    
    for iteration = 1:max_iterations
        # Locate grid points that surround our midpoint guess
        pos = grid_locate(guess_grid, grid_length, guess_mid; reverse=false)
        # If the above procedure returns a single value, exact match, else use
        # interpolation
        if typeof(pos) == Tuple{Int64, Int64}
            next_guess = interpolate_vec(guess_grid, next_guess_grid, guess_mid, pos)
        else
            next_guess = next_guess_grid[pos, :]
        end
        f_result = f(init_value, guess_mid, next_guess, args...)
        # If function is greater than zero, optimal value is between startpoint and
        # midpoint, or else it is between midpoint and endpoint
        if f_result > 0.0 + tolerance
            guess_max = guess_mid
            guess_mid = (guess_min + guess_mid) / 2
        elseif f_result < 0.0 - tolerance
            guess_min = guess_mid
            guess_mid = (guess_mid + guess_max) / 2
        else
            return guess_mid
        end
    end
    return missing
end
