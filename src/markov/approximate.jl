#==========================================================================================
# EconMeth.Markov.Approximate
# Ryan Gilland
==========================================================================================#
import Distributions as dist

#==========================================================================================
# Tauchen / Rouwenhorst
==========================================================================================#
function tauchen(
    mean,
    variance,
    state_size,
    multiplier,
    lag_coefficient;
    print_output=false
    )
    
    std = sqrt(variance)
    states = Array{Float64}(undef, state_size)
    states[state_size] = multiplier*sqrt(variance / (1 - lag_coefficient^2))
    states[1] = -states[state_size]
    if print_output
        println("States:\n$(states[1])")
    end
    step_length = (states[state_size] - states[1])/(state_size - 1)
    for i = 2:state_size-1
        states[i] = states[1] + (i - 1)*step_length
        if print_output
            println(states[i])
        end
    end
    if print_output
        println(states[state_size])
    end
    
    # Standard normal distribution for the normalized states
    D = dist.Normal(0, 1)
    
    # Transition matrix
    pmatrix = Array{Float64}(undef, (state_size, state_size))
    for i = 1:state_size
        for j = 1:state_size
            if j == 1
                step_pos = (states[j] + step_length/2 - lag_coefficient*states[i])/std
                pmatrix[i, j] = dist.cdf(D, step_pos)
            elseif j == state_size
                step_neg = (states[j] - step_length/2 - lag_coefficient*states[i])/std
                pmatrix[i, j] = 1 - dist.cdf(D, step_neg)
            else
                step_pos = (states[j] + step_length/2 - lag_coefficient*states[i])/std
                step_neg = (states[j] - step_length/2 - lag_coefficient*states[i])/std
                pmatrix[i, j] = dist.cdf(D, step_pos) - dist.cdf(D, step_neg)
            end
        end
    end
    
    if print_output
        println("pmatrix:")
        for i = 1:state_size
            println(round.(pmatrix[i, :], digits=3))
        end
    end
    
    states .+= mean / (1 - lag_coefficient)

    return MarkovChain(pmatrix, states, missing, missing, missing)
end

function rouwenhorst(
    mean,
    variance,
    state_size,
    lag_coefficient;
    print_output=false
    )

    psi = sqrt(variance / (1 - lag_coefficient^2)) * sqrt(state_size - 1)

    states = Array{Float64}(undef, state_size)
    states[state_size] = psi
    states[1] = -psi
    if print_output
        println("States:")
        println(states[1])
    end
    step_length = (states[state_size] - states[1])/(state_size - 1)
    for i = 2:state_size-1
        states[i] = states[1] + (i - 1)*step_length
        if print_output
            println(states[i])
        end
    end
    if print_output
        println(states[state_size])
    end

    p = (1 + lag_coefficient) / 2
    q = p
    
    # Transition matrix
    pmatrix_n = Array{Float64}(undef, (2, 2))
    pmatrix_n[1, 1] = p
    pmatrix_n[1, 2] = 1 - p
    pmatrix_n[2, 1] = 1 - q
    pmatrix_n[2, 2] = q
    pmatrix = pmatrix_n
    if state_size > 2
        global pmatrix
        global pmatrix_n
        for n = 3:state_size
            global pmatrix
            global pmatrix_n
            pmatrix = zeros((n, n))
            pmatrix[1:n-1, 1:n-1] += p .* pmatrix_n
            pmatrix[1:n-1, 2:n] += (1 - p) .* pmatrix_n
            pmatrix[2:n, 1:n-1] += (1 - q) .* pmatrix_n
            pmatrix[2:n, 2:n] += q .* pmatrix_n
            pmatrix[2:n-1, :] ./= 2
            pmatrix_n = pmatrix
        end
    end
    
    if print_output
        println("pmatrix:")
        for i = 1:state_size
            println(round.(pmatrix[i, :], digits=3))
        end
    end

    states .+= mean / (1 - lag_coefficient)
    
    return MarkovChain(pmatrix, states, missing, missing, missing)
end
