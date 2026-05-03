using Catalyst, JumpProcesses, ModelingToolkit, Plots, DifferentialEquations
using StatsBase
using Random

@variables t m₁(t) m₂(t) m₃(t) p₁(t) p₂(t) p₃(t)
@parameters km kX γm γX K H ϵ  # Transcription, translation, degradation, Hill, leakage

repressilator = @reaction_network begin
    # Transcription with Hill repression and leakage
    (km * (ϵ + (1 - ϵ) * (1 / (1 + (p₃ / K)^H)))), ∅ → m₁
    (km * (ϵ + (1 - ϵ) * (1 / (1 + (p₁ / K)^H)))), ∅ → m₂
    (km * (ϵ + (1 - ϵ) * (1 / (1 + (p₂ / K)^H)))), ∅ → m₃

    # Translation
    kX, m₁ → m₁ + p₁
    kX, m₂ → m₂ + p₂
    kX, m₃ → m₃ + p₃

    # Degradation
    γm, m₁ → ∅
    γm, m₂ → ∅
    γm, m₃ → ∅
    γX, p₁ → ∅
    γX, p₂ → ∅
    γX, p₃ → ∅
end

# Initial state
u0 = [
    m₁ => 0.0, m₂ => 0.0, m₃ => 0.0,
    p₁ => 0.0, p₂ => 0.0, p₃ => 0.0
]


tspan = (0.0, 50.0)

pmap = [
    H  => 7.0,
    γm => 17.6822,
    kX => 100.0,
    km => 113.4372,
    ϵ  => 0.15,
    γX => 1.0,      
    K  => 199.8989,
]

param_array = [
    7.0,       # H
    17.6822,   # γm
    100.0,     # kX
    0.15,      # ϵ
    113.4372,  # km
    1.0,        # γX (fixed)
    199.8989,  # K
]


Random.seed!(123) 
jump_inputs = JumpInputs(repressilator, u0, tspan, pmap)
jump_prob = JumpProblem(jump_inputs, Direct(), save_positions=(false, false))
sol = solve(jump_prob, saveat=0.5)
plot(sol[p₁])

Random.seed!(123)  
prob = remake(jump_prob; p=param_array)
sol = solve(prob, saveat=0.5)
plot!(sol[p₁, :])

function simulate_p1_trajectory(params::Vector{Float64})
    prob = remake(jump_prob; p=params)
    sol = solve(prob, saveat=0.25)[50:end]
    return sol[p₁, :]
end