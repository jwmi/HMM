# Demo of univariate normal HMM using Viterbi algorithm and Baum-Welch.
include("models.jl")

module demo_normal
using PyPlot
using Distributions
using NormalHMM

# settings
m = 2
n = 1000

# parameters
log_pi = log([.4,.6])
log_T = log([.9 .1
             .2 .8])
phi = [Normal(-1,1),Normal(1,1)]

# simulate data
x,z0 = NormalHMM.generate(n,log_pi,log_T,phi)

# compute optimal z
z = NormalHMM.viterbi(x,log_pi,log_T,phi)

# estimate params
tolerance = 1e-3
log_pi_est,log_T_est,phi_est,log_m = NormalHMM.estimate(x,m,tolerance)

# display results
println("\nViterbi percent correct:")
println(mean(z.==z0))
println("\nTrue transition matrix:")
println(exp(log_T))
println("\nEstimated transition matrix:")
println(exp(log_T_est))
println("\nTrue emission distributions:")
println(phi)
println("\nEstimated emission distributions:")
println(phi_est)

figure(1); clf(); hold(true)
plot(1:n,x,"k.")
plot(1:n,[-1,1][z0],"b-")

figure(2); clf(); hold(true)
plot(1:n,[-1,1][z0],"b-")
plot(1:n,[-1,1][z],"r-")
ylim(-2,2)

end # module



