include("rl_repressilator_setup.jl")

LEAKAGE_DECREASED = 0.05

GAMMA_X_MIN = 0.8
GAMMA_X_MAX = 1.1


# Extract the second peak (ignoring lag=0)
function second_peak_from_acf_loc(acf::Vector{Float64})
    for i in 3:length(acf)-1
        if acf[i] > acf[i-1] && acf[i] > acf[i+1]
            return i/length(acf)
        end
    end
    return 0.0
end


# Loss function: negative second autocorrelation peak (to maximize it during optimization)
function loss_amort(params::Vector{Float64}, desired_freq::Float64 = 1/10.0)
    signal = simulate_p1_trajectory(params)
    acf = autocorrelation(signal)
    return sqrt((second_peak_from_acf_loc(acf) - desired_freq).^2)*100
end

# Loss function: negative second autocorrelation peak (to maximize it during optimization)
function freq(signal)
    acf = autocorrelation(signal)
    return second_peak_from_acf_loc(acf)
end


prams_array_1 = [
    7.0,       # H
    17.6822,   # γm
    100.0,     # kX
    0.05,  # ϵ (decreased leakage)
    113.4372,  # km
    0.9,       # γX (fixed)
    199.8989   # K
]

Random.seed!(123) 
signal = simulate_p1_trajectory(prams_array_1)

# Plot
plot(signal, xlabel="Time", ylabel="Protein p₁", lw=2, title="Repressilator p₁ trajectory")

# Optional: show ACF and second peak value
acf_vals = autocorrelation(signal)


second = second_peak_from_acf(acf_vals)
second_peak_loc = 1/second_peak_from_acf_loc(acf_vals)
Random.seed!(123)
loss_val = loss_amort(prams_array_1,1/7) + loss(prams_array_1)*0.3

function unnorm_params(normalized_params)
    H  = denormalize_param(normalized_params[1], H_MIN, H_MAX)
    γX = denormalize_param(normalized_params[2], GAMMA_X_MIN, GAMMA_X_MAX)
    γm = denormalize_param(normalized_params[3], GM_MIN, GM_MAX)
    kX = denormalize_param(normalized_params[4], KX_MIN, KX_MAX)
    ϵ  = LEAKAGE_DECREASED
    km = denormalize_param(normalized_params[5], KM_MIN, KM_MAX)
    K  = denormalize_param(normalized_params[6], K_MIN, K_MAX)

    return [
        H,
        γm,
        kX,
        ϵ,
        km,
        γX,
        K
    ]
end

# Setup loss wrapper: input params in [-1,1], output loss with real params
function reward_amort(normalized_params, desired_freq)
    real_params = unnorm_params(normalized_params)
    return -loss_amort(real_params, desired_freq) - loss(real_params)*0.3
end

function plot_amort(normalized_params)
    return simulate_p1_trajectory(unnorm_params(normalized_params))
end


function observation(params)
    return plot_amort(params)
end

function observation_freq(params)
    #todo
    return second_peak_from_acf_loc(autocorrelation(plot_amort(params)))
end

maxs = []
freqs = []
rewards = []

function fix_seed(n)
    Random.seed!(n)
end

Random.seed!(123)  # For reproducibility
for i=1:500
    params_unc = rand(2) * 2 .- 1  # Random normalized params in [-1, 1]
    params_design = rand(4) * 2 .- 1  # Random normalized params in [-1, 1]
    params_arr = vcat(params_unc, params_design)
    push!(rewards, reward_amort(params_arr, 1/7))
    push!(maxs, maximum(observation(params_arr)))
    push!(freqs, observation_freq(params_arr))
end

m_obs = mean(maxs)
std_obs = std(maxs)

m_freq = mean(freqs)
std_freq = std(freqs)

m_reward = mean(rewards)
std_reward = std(rewards)

function observation_normalized(params)
    return (observation(params) .- m_obs) ./ std_obs
end

function observation_freq_normalized(params)
    return (observation_freq(params)-m_freq) / std_freq
end

function reward_normalized_past_amort(params, desired_freq)
    return (reward_amort(params, desired_freq) - m_reward) / std_reward
end

params_unc = rand(2) * 2 .- 1  # Random normalized params in [-1, 1]
params_design = rand(4) * 2 .- 1  # Random normalized params in [-1, 1]
params_arr = vcat(params_unc, params_design)

plot(observation_normalized(params_arr), xlabel="Time", ylabel="Normalized Protein p₁", lw=2, title="Repressilator p₁ trajectory (normalized)")

1/observation_freq(params_arr)

params = zeros(6)
plot(observation_normalized(params), xlabel="Time", ylabel="Normalized Protein p₁", lw=2, title="Repressilator p₁ trajectory (normalized)")

1/observation_freq(params)

design_paper = [
    17.6822,   # γm
    100.0,     # kX
    113.4372,  # km
    199.8989   # K
]

unc_paper = [
    7.0,       # H
    1.0        # γX (fixed)
]

design_paper_normalized = [
    normalize_param(design_paper[1], GM_MIN, GM_MAX),
    normalize_param(design_paper[2], KX_MIN, KX_MAX),
    normalize_param(design_paper[3], KM_MIN, KM_MAX),
    normalize_param(design_paper[4], K_MIN, K_MAX)
]
unc_paper_normalized = [
    normalize_param(unc_paper[1], H_MIN, H_MAX),
    normalize_param(unc_paper[2], GAMMA_X_MIN, GAMMA_X_MAX)
]

params_paper = vcat(unc_paper_normalized, design_paper_normalized)
plot(observation(params_paper), xlabel="Time", ylabel="Normalized Protein p₁", lw=2, title="Repressilator p₁ trajectory (normalized, paper params)")

1/observation_freq(params_paper)

params_unc = rand(2) * 2 .- 1  # Random normalized params in [-1, 1]

params_paper_unc = vcat(params_unc, design_paper_normalized)

plot!(observation(params_paper_unc), xlabel="Time", ylabel="Normalized Protein p₁", lw=2, title="Repressilator p₁ trajectory (normalized, paper params + random uncertainty)")

1/observation_freq(params_paper_unc)

function first_obs(unc_params)
    return observation_normalized(vcat(unc_params, design_paper_normalized))[1:128]
end

function first_obs_freq(unc_params)
    return observation_freq_normalized(vcat(unc_params, design_paper_normalized))
end

function observation(unc_params, design_params)
    return observation_normalized(vcat(unc_params, design_params))[1:128]
end