include("rl_repressilator_setup.jl")

LEAKAGE_DECREASED = 0.15 # it is not decreased to be refactored
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

function freq(signal)
    acf = autocorrelation(signal)
    return second_peak_from_acf_loc(acf)
end



prams_array_1 = [
    7.0,       # H
    17.6822,   # γm
    100.0,     # kX
    0.15,  # ϵ 
    113.4372,  # km
    1.0,       # γX (fixed)
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

function denormalize(normalized_params)
    H  = denormalize_param(normalized_params[1], H_MIN, H_MAX)
    γm = denormalize_param(normalized_params[2], GM_MIN, GM_MAX)
    kX = denormalize_param(normalized_params[3], KX_MIN, KX_MAX)
    ϵ  = LEAKAGE_DECREASED
    km = denormalize_param(normalized_params[4], KM_MIN, KM_MAX)
    γX = GAMMA_X
    K  = denormalize_param(normalized_params[5], K_MIN, K_MAX)
    return [H, γm, kX, ϵ, km, γX, K]
end

# Setup loss wrapper: input params in [-1,1], output loss with real params
function reward_amort(normalized_params, desired_freq)
    real_params = denormalize(normalized_params)
    return -loss_amort(real_params, desired_freq) - loss(real_params)*0.3
end

function plot_amort(normalized_params)
    real_params = denormalize(normalized_params)
    return simulate_p1_trajectory(real_params)
end

freqs = []
Random.seed!(123)  # For reproducibility
for i=1:500
    params = rand(7) * 2 .- 1  # Random params in [-1, 1]
    push!(freqs, freq(plot_amort(params)))
end

m_freq = mean(freqs)
std_freq = std(freqs)

1/(m_freq+ std_freq)
1/(m_freq - std_freq)

function normalize_freq(freq)
    return (freq - m_freq) / std_freq
end

function denormalize_freq(normalized_freq)
    return normalized_freq * std_freq + m_freq
end

# normalize reward
rewards = []
for i=1:500
    params = rand(7) * 2 .- 1  # Random params in [-1, 1]
    freq_val = rand() * 2 - 1 
    reward_val = reward_amort(params, denormalize_freq(freq_val))
    push!(rewards, reward_val)
end
m_reward = mean(rewards)
std_reward = std(rewards)

function normalized_reward_amort(params, desired_freq)
    return (reward_amort(normalized_params, desired_freq) - m_reward) / std_reward
end