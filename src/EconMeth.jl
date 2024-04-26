#==========================================================================================
# EconMeth
# Ryan Gilland
==========================================================================================#
module EconMeth

export
# utility.utility
    grid_distance,
    grid_locate,
    interpolate,
    interpolate_vec,

# markov.core
    MarkovChain,
    simulate!,

# markov.approximate
    tauchen,
    rouwenhorst,

# optimize.root_solve
    bisection

include("utility/utility.jl")
include("markov/core.jl")
include("markov/approximate.jl")
include("optimize/root_solve.jl")

end
