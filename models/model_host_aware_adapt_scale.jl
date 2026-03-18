using DifferentialEquations
using Random
using Plots

# Native parameters
thetar = 426.8693338968694
k_cm = 0.005990373118888
nr = 7549.0
gmax = 1260.0
cl = 0
nume0 = 4.139172187824451
s0 = 1.0e4
vm = 5800.0
Km = 1.0e3
numr0 = 929.9678874564831
nx = 300.0
kq = 1.522190403737490e+05
Kp = 180.1378030928276
vt = 726.0
nump0 = 0.0
numq0 = 948.9349882947897
Kt = 1.0e3
nq = 4
aatot = 1.0e8
ns = 0.5
thetax = 4.379733394834643

# Define rate constants
b = 0
dm = 0.1
kb = 1
ku = 1.0
f = cl * k_cm
rates = [b, dm, kb, ku, f]

# Define initial conditions
rmr_0 = 0
em_0 = 0
rmp_0 = 0
rmq_0 = 0
rmt_0 = 0
et_0 = 0
rmm_0 = 0

zmm_0 = 0
zmr_0 = 0
zmp_0 = 0
zmq_0 = 0
zmt_0 = 0
mt_0 = 0
mm_0 = 0
q_0 = 0
p_0 = 0
si_0 = 0
mq_0 = 0
mp_0 = 0
mr_0 = 0
r_0 = 10.0
a_0 = 1000.0
init = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, et_0, rmm_0, zmm_0, zmr_0, zmp_0, zmq_0, zmt_0, mt_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]

# Define parameters
parameters = [thetar, k_cm, nr, gmax, cl, nume0, s0, vm, Km, numr0, nx, kq, Kp, vt, nump0, numq0, Kt, nq, aatot, ns, thetax]

# Define the differential equations as a function
function ribonew_mc2_odes!(dy, y, p, t)
    y = max.(y, 0)
    rates = p[1:5]
    parameters = p[6:end]
    
    b, dm, kb, ku, f = rates
    thetar, k_cm, nr, gmax, cl, nume0, s0, vm, Km, numr0, nx, kq, Kp, vt, nump0, numq0, Kt, nq, aatot, ns, thetax = parameters
    
    rmr, em, rmp, rmq, rmt, et, rmm, zmm, zmr, zmp, zmq, zmt, mt, mm, q, p, si, mq, mp, mr, r, a = y
    
    Kg = gmax / Kp
    gamma = gmax * a / (Kg + a)
    ttrate = (rmq + rmr + rmp + rmt + rmm) * gamma
    lam = ttrate / aatot
    fr = nr * (r + rmr + rmp + rmt + rmm + rmq + zmr + zmp + zmt + zmm + zmq) /
         (nr * (r + rmr + rmp + rmt + rmm + rmq + zmr + zmp + zmt + zmm + zmq) + nx * (p + q + et + em))
    nucat = em * vm * si / (Km + si)
    
    dy[1] = kb * r * mr + b * zmr - ku * rmr - gamma / nr * rmr - f * rmr - lam * rmr
    dy[2] = gamma / nx * rmm - lam * em
    dy[3] = kb * r * mp + b * zmp - ku * rmp - gamma / nx * rmp - f * rmp - lam * rmp
    dy[4] = kb * r * mq + b * zmq - ku * rmq - gamma / nx * rmq - f * rmq - lam * rmq
    dy[5] = kb * r * mt + b * zmt - ku * rmt - gamma / nx * rmt - f * rmt - lam * rmt
    dy[6] = gamma / nx * rmt - lam * et
    dy[7] = kb * r * mm + b * zmm - ku * rmm - gamma / nx * rmm - f * rmm - lam * rmm
    dy[8] = f * rmm - b * zmm - lam * zmm
    dy[9] = f * rmr - b * zmr - lam * zmr
    dy[10] = f * rmp - b * zmp - lam * zmp
    dy[11] = f * rmq - b * zmq - lam * zmq
    dy[12] = f * rmt - b * zmt - lam * zmt
    dy[13] = (nume0 * a / (thetax + a)) + ku * rmt + gamma / nx * rmt - kb * r * mt - dm * mt - lam * mt
    dy[14] = (nume0 * a / (thetax + a)) + ku * rmm + gamma / nx * rmm - kb * r * mm - dm * mm - lam * mm
    dy[15] = gamma / nx * rmq - lam * q
    dy[16] = gamma / nx * rmp - lam * p
    dy[17] = (et * vt * s0 / (Kt + s0)) - nucat - lam * si
    dy[18] = (numq0 * a / (thetax + a) / (1 + (q / kq)^nq)) + ku * rmq + gamma / nx * rmq - kb * r * mq - dm * mq - lam * mq
    dy[19] = (nump0 * a / (thetax + a)) + ku * rmp + gamma / nx * rmp - kb * r * mp - dm * mp - lam * mp
    dy[20] = (numr0 * a / (thetar + a)) + ku * rmr + gamma / nr * rmr - kb * r * mr - dm * mr - lam * mr
    dy[21] = ku * rmr + ku * rmt + ku * rmm + ku * rmp + ku * rmq +
             gamma / nr * rmr + gamma / nr * rmr +
             gamma / nx * rmt + gamma / nx * rmm + gamma / nx * rmp + gamma / nx * rmq -
             kb * r * mr - kb * r * mt - kb * r * mm - kb * r * mp - kb * r * mq - lam * r
    dy[22] = ns * nucat - ttrate - lam * a
end

# Define the time span and initial conditions
t0 = 0.0
tf = 1e7  # Final time for simulation
tspan = (t0, tf)

# Define solver options (adjust if necessary)
t = [t0, tf]
prob = ODEProblem(ribonew_mc2_odes!, init, tspan, [rates; parameters])
sol = solve(prob, Rosenbrock23())

function solve_warmup(p)
    remake_prob = remake(prob; p=p)
    sol_warmup = solve(remake_prob, Rosenbrock23(),
                       saveat=1e7)
    return sol_warmup[:,end]
end

@time solve_warmup([rates; parameters])
@assert solve_warmup([rates; parameters])== sol[:,end]
# Access the solution
y = sol

rmr = y[1, :]
em = y[2, :]
rmp = y[3, :]
rmq = y[4, :]
rmt = y[5, :]
et = y[6, :]
rmm = y[7, :]
zmm = y[8, :]
zmr = y[9, :]
zmp = y[10, :]
zmq = y[11, :]
zmt = y[12, :]
mt = y[13, :]
mm = y[14, :]
q = y[15, :]
p = y[16, :]
si = y[17, :]
mq = y[18, :]
mp = y[19, :]
mr = y[20, :]
r = y[21, :]
a = y[22, :]










# Extracting the last elements of arrays for initial conditions
rmr_0 = rmr[end]
em_0 = em[end]
rmp_0 = rmp[end]
rmq_0 = rmq[end]
rmt_0 = rmt[end]
et_0 = et[end]
rmm_0 = rmm[end]
zmm_0 = zmm[end]
zmr_0 = zmr[end]
zmp_0 = zmp[end]
zmq_0 = zmq[end]
zmt_0 = zmt[end]
mt_0 = mt[end]
mm_0 = mm[end]
q_0 = q[end]
p_0 = p[end]
si_0 = si[end]
mq_0 = mq[end]
mp_0 = mp[end]
mr_0 = mr[end]
r_0 = r[end]
a_0 = a[end]

# Randomize initial conditions for the GFP species
meanGFP = em[end]
meanmGFP = mm[end]
meanrmGFP = rmm[end]

Random.seed!(123)  # For reproducibility
mg_0 = meanmGFP + 0.3 * meanmGFP * randn()
rmg_0 = meanrmGFP + 0.3 * meanrmGFP * randn()
g_0 = meanGFP + 0.3 * meanGFP * randn()




# Define the initial conditions array
init_2 = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, rmg_0, et_0, rmm_0, zmm_0, zmr_0, zmp_0, zmq_0, zmt_0, mt_0, mg_0, g_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]

function compute_init_2(parameters, rates, seed=nothing)
    # Perform warm-up computation (replace with actual warm-up logic)
    y = solve_warmup([rates; parameters])  # Call the warm-up function with the extracted parameters

    # Extract only the necessary values from the warm-up result
    em = y[2, :]
    mm = y[14, :]
    rmm = y[7, :]

    meanGFP = em[end]
    meanmGFP = mm[end]
    meanrmGFP = rmm[end]

     # Randomize initial conditions for the GFP species
    if seed !== nothing
        Random.seed!(seed)  # Set the random seed for reproducibility
    end
    mg_0 = meanmGFP + 0.3 * meanmGFP * randn()
    rmg_0 = meanrmGFP + 0.3 * meanrmGFP * randn()
    g_0 = meanGFP + 0.3 * meanGFP * randn()

    # Define the initial conditions array
    init_2 = [rmr_0, em_0, rmp_0, rmq_0, rmt_0, rmg_0, et_0, rmm_0, zmm_0, zmr_0, zmp_0, zmq_0, zmt_0, mt_0, mg_0, g_0, mm_0, q_0, p_0, si_0, mq_0, mp_0, mr_0, r_0, a_0]

    return init_2
end

@assert compute_init_2(parameters, rates, 123) == init_2

# INDUCTION PARAMETER
numg0 = 25

# Redefine the parameter vector for the GFP model
parameters_2 = [cl, nume0, vm, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, gmax, thetax, Km, Kp, Kt, numg0, kq, numr0, nump0]
# Define rate constants
b = 0
dm = 0.1
kb = 1
ku = 1.0
kb_g = 1.0
ku_g = 0.2
f = cl * k_cm
dmg = log(2) / 2
dg = log(2) / 4
rates_2 = [b, dm, kb, ku, f, dmg, dg, kb_g,ku_g]

# Define the ODE system as a function
function ribonew_mc2_gfp_odes!(dydt, y, p, t)
    y = max.(y, 0)
    # Extract rate constants and parameters
    rates = p[1:9]    # First 7 are the rates
    parameters = p[10:end]  # Rest are the parameters

    b, dm, kb, ku, f, dmg, dg, kb_g, ku_g = rates
    cl, nume0, vm, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, gmax, thetax, Km, Kp, Kt, numg0, kq, numr0, nump0 = parameters

    # Extract variables from y
    rmr, em, rmp, rmq, rmt, rmg, et, rmm, zmm, zmr, zmp, zmq, zmt, mt, mg, g, mm, q, p, si, mq, mp, mr, r, a = y

    # Intermediate variables
    Kg = gmax / Kp
    gamma = gmax * a / (Kg + a)
    ttrate = (rmq + rmr + rmp + rmt + rmm + rmg) * gamma
    lam = ttrate / aatot
    fr = nr * (r + rmr + rmp + rmt + rmm + rmq + rmg + zmr + zmp + zmt + zmm + zmq) /
         (nr * (r + rmr + rmp + rmt + rmm + rmq + rmg + zmr + zmp + zmt + zmm + zmq) + nx * (p + q + et + em + g))
    nucat = em * vm * si / (Km + si)

    # Define the system of ODEs
    dydt[1] = +kb * r * mr + b * zmr - ku * rmr - gamma / nr * rmr - f * rmr - lam * rmr # rmr
    dydt[2] = +gamma / nx * rmm - lam * em # em
    dydt[3] = +kb * r * mp + b * zmp - ku * rmp - gamma / nx * rmp - f * rmp - lam * rmp # rmp
    dydt[4] = +kb * r * mq + b * zmq - ku * rmq - gamma / nx * rmq - f * rmq - lam * rmq # rmq
    dydt[5] = +kb * r * mt + b * zmt - ku * rmt - gamma / nx * rmt - f * rmt - lam * rmt # rmt
    dydt[6] = +kb_g * r * mg - ku_g * rmg - gamma / nx * rmg # rmg
    dydt[7] = +gamma / nx * rmt - lam * et # et
    dydt[8] = +kb * r * mm + b * zmm - ku * rmm - gamma / nx * rmm - f * rmm - lam * rmm # rmm
    dydt[9] = +f * rmm - b * zmm - lam * zmm # zmm
    dydt[10] = +f * rmr - b * zmr - lam * zmr # zmr
    dydt[11] = +f * rmp - b * zmp - lam * zmp # zmp
    dydt[12] = +f * rmq - b * zmq - lam * zmq # zmq
    dydt[13] = +f * rmt - b * zmt - lam * zmt # zmt
    dydt[14] = +(nume0 * a / (thetax + a)) + ku * rmt + gamma / nx * rmt - kb * r * mt - dm * mt - lam * mt # mt
    dydt[15] = +(numg0 * a / (thetax + a)) + ku_g * rmg + gamma / nx * rmg - kb_g * r * mg - dmg * mg - lam * mg # mg
    dydt[16] = +gamma / nx * rmg - dg * g - lam * g # g
    dydt[17] = +(nume0 * a / (thetax + a)) + ku * rmm + gamma / nx * rmm - kb * r * mm - dm * mm - lam * mm # mm
    dydt[18] = +gamma / nx * rmq - lam * q # q
    dydt[19] = +gamma / nx * rmp - lam * p # p
    dydt[20] = +(et * vt * s0 / (Kt + s0)) - nucat - lam * si # si
    dydt[21] = +(numq0 * a / (thetax + a) / (1 + (q / kq)^nq)) + ku * rmq + gamma / nx * rmq - kb * r * mq - dm * mq - lam * mq # mq
    dydt[22] = +(nump0 * a / (thetax + a)) + ku * rmp + gamma / nx * rmp - kb * r * mp - dm * mp - lam * mp # mp
    dydt[23] = +(numr0 * a / (thetar + a)) + ku * rmr + gamma / nr * rmr - kb * r * mr - dm * mr - lam * mr # mr
    dydt[24] = +ku * rmr + ku * rmt + ku * rmm + ku * rmp + ku * rmq + gamma / nr * rmr + gamma / nr * rmr + gamma / nx * rmt + gamma / nx * rmm + gamma / nx * rmp + gamma / nx * rmq + ku_g * rmg + gamma / nx * rmg - kb * r * mr - kb * r * mt - kb * r * mm - kb * r * mp - kb * r * mq - lam * r - kb_g * r * mg # r
    dydt[25] = +ns * nucat - ttrate - lam * a # a
end

# Define the time span
t0 = 0.0
tf = 1e7  # Final time for simulation
tspan = (t0, tf)


# Define the ODE problem
prob_2 = ODEProblem(ribonew_mc2_gfp_odes!, init_2, tspan, [rates_2; parameters_2])

# Solve using the Rosenbrock method
sol = solve(prob_2, Rosenbrock23())

function solve_prob(parameters, numg0, unc_params ,seed = nothing)


    thetar, k_cm, nr, gmax, cl, nume0, s0, vm, Km, numr0, nx, kq, Kp, vt, nump0, numq0, Kt, nq, aatot, ns, thetax = parameters

    ns = unc_params[1]

    parameters[20] = ns

    # Define rate constants
    b = 0
    dm = 0.1
    kb = 1
    ku = 1
    f = cl * k_cm
    rates = [b, dm, kb, ku, f]


    #compute init
    init_2 = compute_init_2(parameters, rates, seed)

    
    # Redefine the parameter vector for the GFP model
    parameters_2 = [cl, nume0, vm, vt, aatot, s0, nx, numq0, nq, nr, ns, thetar, k_cm, gmax, thetax, Km, Kp, Kt, numg0, kq, numr0, nump0]

    # Define rate constants
    b = 0
    dm = 0.1
    kb = 1
    ku = 1
    kb_g = unc_params[2]*2
    ku_g = unc_params[3]*2
    f = cl * k_cm
    dmg = log(2) / 2
    dg = log(2) / 4
    rates_2 = [b, dm, kb, ku, f, dmg, dg, kb_g, ku_g]

    remake_prob = remake(prob_2; p=[rates_2; parameters_2], u0=init_2)
    sol_prob = solve(remake_prob, Rosenbrock23(),
                     saveat=1e7)
    return sol_prob[:,end]
end

# Extract the solution

y = sol
solve_prob(parameters, numg0,rand(3) ,123)

# @assert y[:,end] == solve_prob(parameters, numg0, 123)

numg0_values = exp.(collect(0:1:10)) 
# Plotting the effect of numg0 on the final value of p

solve_prob(parameters, numg0, rand(3))[16]  # p is the 16th variable in the solution
final_p_values = Float64[]
unc = rand(3)
@time for numg0_val in numg0_values
    final_p = solve_prob(parameters, numg0_val,unc)
    push!(final_p_values, final_p[16])  # p is the 16th variable in the solution
end

plot!(final_p_values)
plot!(legend=nothing)

function reward_unnorm(numg0_unnorm,unc)
    return solve_prob(parameters,exp(numg0_unnorm*10),unc)[16]
end

vals = rand(1000)
rewards = zeros(1000)
@time for (i, val) in enumerate(vals)
    rewards[i] = reward_unnorm(val, unc)
end

using Statistics
Random.seed!(1234)
mm = mean(rewards)
stdstd = std(rewards)

function reward(numg0_unnorm,unc)
    return reward_unnorm(numg0_unnorm,unc)/stdstd
end

function denormalize_numg0(normalized_val)
    return log(normalized_val) / 10
end

function denormalize_reward(normalized_reward, stdstd)
    return normalized_reward * stdstd
end


reward(0.5, unc)

plot(final_p_values)
plot!(legend=nothing)