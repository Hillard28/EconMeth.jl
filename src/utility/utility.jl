#==========================================================================================
# EconMeth.Utility.Utility
# Ryan Gilland
==========================================================================================#
# Computes the distance between grid points using a shape parameter
function grid_distance(min, max, min_value, max_value, i, shape=1)
    min_value + (max_value - min_value)*((i - min) / (max - min))^shape
end

# Locates grid points surrounding a given value
function grid_locate(grid, grid_length, point; reverse=false)
    if grid[1] >= point
        return 1
    elseif grid[grid_length] <= point
        return grid_length
    else
        # Make search quicker if near end of grid
        if reverse
            for i = grid_length:-1:2
                if grid[i-1] == point
                    return i-1
                elseif grid[i-1] < point
                    return (i-1, i)
                end
            end
        else
            for i = 1:grid_length - 1
                if grid[i+1] == point
                    return i+1
                elseif grid[i+1] > point
                    return (i, i+1)
                end
            end
        end
    end
end

function interpolate_vec(grid, grid_int, point, pos)
    return grid_int[pos[1], :] .+ (point - grid[pos[1]]) .*
        ((grid_int[pos[2], :] .- grid_int[pos[1], :]) ./ (grid[pos[2]] - grid[pos[1]]))
end

function interpolate(grid, grid_int, point, pos)
    return grid_int[pos[1]] .+ (point - grid[pos[1]]) .*
        ((grid_int[pos[2]] .- grid_int[pos[1]]) ./ (grid[pos[2]] - grid[pos[1]]))
end
