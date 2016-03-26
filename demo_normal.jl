# Demo of univariate normal HMM and Viterbi algorithm.
include("models.jl")

module demo_normal
using PyPlot
using Distributions
using NormalHMM

# settings
m = 2
n = 100

# parameters
log_pi = log([.4,.6])
log_T = log([.9 .1
             .2 .8])
phi = [Normal(-1,1),Normal(1,1)]

# simulate data
x,z0 = NormalHMM.generate(n,log_pi,log_T,phi)

# compute optimal z
z = NormalHMM.viterbi(x,log_pi,log_T,phi)

# display results
println(mean(z.==z0))

figure(1); clf(); hold(true)
plot(1:n,x,"k.")
plot(1:n,[-1,1][z0],"b-")

figure(2); clf(); hold(true)
plot(1:n,[-1,1][z0],"b-")
plot(1:n,[-1,1][z],"r-")
ylim(-2,2)

end # module



