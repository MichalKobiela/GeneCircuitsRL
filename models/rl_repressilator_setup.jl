
include("model_repressilator.jl")


# Compute normalized autocorrelation
function autocorrelation(signal::Vector{Float64})
    n = length(signal)
    μ = mean(signal)
    σ² = var(signal)
    acf = [sum((signal[1:end-l] .- μ) .* (signal[l+1:end] .- μ)) / ((n - l) * σ²) for l in 0:n-2]
    return acf
end

# Extract the second peak (ignoring lag=0)
function second_peak_from_acf(acf::Vector{Float64})
    for i in 3:length(acf)-1
        if acf[i] > acf[i-1] && acf[i] > acf[i+1]
            return acf[i]
        end
    end
    return 0.0
end

# Loss function: negative second autocorrelation peak (to maximize it during optimization)
function loss(params::Vector{Float64})
    signal = simulate_p1_trajectory(params)
    acf = autocorrelation(signal)
    return -second_peak_from_acf(acf)
end

Random.seed!(123) 
signal = simulate_p1_trajectory(param_array)

plot!(signal, xlabel="Time", ylabel="Protein p₁", lw=2, title="Repressilator p₁ trajectory")

acf_vals = autocorrelation(signal)

Random.seed!(123) 
second = second_peak_from_acf(acf_vals)

Random.seed!(123) 
loss_val = loss(param_array)

@assert second == -loss_val
plot(acf_vals, xlabel="Lag", ylabel="ACF", title="Autocorrelation of p₁")
println("Second peak value: ", second)


params_paper = [
    7.0,       # H
    17.6822,   # γm
    100.0,     # kX
    113.4372,  # km
    199.8989   # K
]

# Parameter bounds as constants
const H_MIN, H_MAX = 3.0, 7.0
const K_MIN, K_MAX = 10.0, 200.0
const KM_MIN, KM_MAX = 3.0, 120.0
const KX_MIN, KX_MAX = 100.0, 1000.0
const GM_MIN, GM_MAX = 4.0, 50.0

const LEAKAGE = 0.15       # fixed leakage coefficient
const GAMMA_X = 1.0        # fixed protein degradation rate

# Map normalized [-1,1] → [low, high]
function denormalize_param(x, low, high)
    return (x + 1) * (high - low) / 2 + low
end

# Map real [low, high] → [-1,1]
function normalize_param(x, low, high)
    return 2 * (x - low) / (high - low) - 1
end

# From normalized vector to real vector
function denormalize_params(norm_params)
    H  = denormalize_param(norm_params[1], H_MIN, H_MAX)
    γm = denormalize_param(norm_params[2], GM_MIN, GM_MAX)
    kX = denormalize_param(norm_params[3], KX_MIN, KX_MAX)
    km = denormalize_param(norm_params[4], KM_MIN, KM_MAX)
    K  = denormalize_param(norm_params[5], K_MIN, K_MAX)
    return [H, γm, kX, km, K]
end

# From real parameters vector to normalized vector
function normalize_params(real_params)
    H  = normalize_param(real_params[1], H_MIN, H_MAX)
    γm = normalize_param(real_params[2], GM_MIN, GM_MAX)
    kX = normalize_param(real_params[3], KX_MIN, KX_MAX)
    km = normalize_param(real_params[4], KM_MIN, KM_MAX)
    K  = normalize_param(real_params[5], K_MIN, K_MAX)
    return [H, γm, kX, km, K]
end

# Setup loss wrapper: input params in [-1,1], output loss with real params
function reward(normalized_params)
    H  = denormalize_param(normalized_params[1], H_MIN, H_MAX)
    γm = denormalize_param(normalized_params[2], GM_MIN, GM_MAX)
    kX = denormalize_param(normalized_params[3], KX_MIN, KX_MAX)
    ϵ  = LEAKAGE
    km = denormalize_param(normalized_params[4], KM_MIN, KM_MAX)
    γX = GAMMA_X
    K  = denormalize_param(normalized_params[5], K_MIN, K_MAX)

    real_params = [
        H,
        γm,
        kX,
        ϵ,
        km,
        γX,
        K
    ]
    return -loss(real_params)
end

using Main.Threads
function reward_repeated(normalized_params)
    res = zeros(6)
    @threads for i in 1:6
        res[i] = reward(normalized_params)
    end
    return mean(res)
end


Random.seed!(123) 
setup_loss_val = reward(normalize_params(params_paper)) 
Random.seed!(123) 
loss_val = loss(vcat(params_paper[1:3], LEAKAGE, params_paper[4], GAMMA_X, params_paper[5]))
@assert setup_loss_val == -loss_val
@assert denormalize_params(normalize_params(params_paper)) == params_paper

function evaluate_params(params)
    res= []
    for i in 1:1000
        Random.seed!(i) 
        push!(res, reward(params))
    end
    return res
end
