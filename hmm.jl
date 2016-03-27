# Core functions for hidden Markov models

# Copyright (c) 2016 Jeffrey W. Miller
# This software is released under the MIT License.
# If you use this software in your research, please cite:
# 
#   Jeffrey W. Miller (2016). Lecture Notes on Advanced Stochastic Modeling. Duke University, Durham, NC.

# This code assumes the following have been defined:
#   log_q = function such that log_q(x_t,phi[z_t]) = log p(x_t | Z_t = z_t), where phi is a vector of parameters.
#   qrnd = function such that qrnd(phi[z_t]) generates a sample from p(x_t|z_t), where phi is a vector of parameters.
#   phi_init = function that creates an array of phi parameters.
#   phi_max! = function that finds the maximal phi parameters for the given data x and weights gamma.

# The functions below involve the following variables:
#   m = number of hidden states
#   n = length of observed sequence
#   log_pi = length m vector with log_pi[s] = log p(Z_1=s).
#   log_T = m-by-m matrix with log_T[r,s] = log p(Z_t=s | Z_{t-1}=r) = log of transition probability r->s.
#   phi = length m vector with phi[s] = parameters for the emission distribution p(x_t | z_t=s).

using Distributions

# Generate a sequence of n hidden states z and observations x.
# OUTPUT:
#    x,z = vectors x_{1:n} and z_{1:n} sampled from p(x_{1:n},z_{1:n}).
function generate(n,log_pi,log_T,phi)
    m = length(log_pi)
    z = zeros(Int,n)
    D = [Categorical(exp(vec(log_T[r,:]))) for r=1:m]
    z[1] = rand(Categorical(exp(log_pi)))
    for t = 2:n
        z[t] = rand(D[z[t-1]])
    end
    x = [qrnd(phi[z[t]]) for t=1:n]
    return x,z
end

# Find most probable sequence of hidden states z given observations x.
# OUTPUT: z = a vector z_{1:n} maximizing p(z_{1:n} | x_{1:n}).
function viterbi(x,log_pi,log_T,phi)
    m = length(log_pi)  # number of hidden states
    n = length(x)  # length of observed sequence
    F = zeros(n,m)  # maxima
    A = zeros(n,m)  # maximizers

    # Compute F and A
    for s = 1:m
        F[1,s] = log_pi[s] + log_q(x[1],phi[s])
    end
    for j = 2:n
        for s = 1:m
            maxval = -Inf
            maxind = 0
            for r = 1:m
                newval = F[j-1,r] + log_T[r,s]
                if newval > maxval
                    maxval = newval
                    maxind = r
                end
            end
            F[j,s] = maxval +  log_q(x[j],phi[s])
            A[j,s] = maxind
        end
    end

    # Recover optimal z
    z = zeros(Int,n)
    z[n] = indmax(F[n,:])
    for t = n-1:-1:1
        z[t] = A[t+1,z[t+1]]
    end
    return z
end

# Compute log(sum(exp(x))) in a way that avoids numerical underflow/overflow.
function logsumexp(x)
    mx = maximum(x)
    if mx==-Inf
        return -Inf
    elseif mx==Inf
        return Inf
    else
        s = 0.0
        for xi in x
            s += exp(xi-mx)
        end
        return log(s)+mx
    end
end

# Run the forward algorithm to compute log(p(x_{1:j},z_j) for each j, z_j.
# OUTPUT: G = n-by-m matrix with G[j,s] = log(p(x_{1:j},z_j=s)).
function forward(x,log_pi,log_T,phi)
    m = length(log_pi)  # number of hidden states
    n = length(x)  # length of observed sequence
    G = zeros(n,m)  # G[j,s] = log(p(x_{1:j},z_j=s))
    l = zeros(m)  # temporary vector

    for s = 1:m
        G[1,s] = log_pi[s] + log_q(x[1],phi[s])
    end
    for j = 2:n
        for s = 1:m
            for r = 1:m
                l[r] = G[j-1,r] + log_T[r,s] + log_q(x[j],phi[s])
            end
            G[j,s] = logsumexp(l)
        end
    end
    log_m = logsumexp(G[n,:])  # log(p(x_{1:n}))
    return G,log_m
end

# Run the backward algorithm to compute log(p(x_{j+1:n}|z_j)) for each j, z_j.
# OUTPUT: H = n-by-m matrix with H[j,s] = log(p(x_{j+1:n}|z_j=s)).
function backward(x,log_pi,log_T,phi)
    m = length(log_pi)  # number of hidden states
    n = length(x)  # length of observed sequence
    H = zeros(n,m)  # H[j,r] = log(p(x_{j+1:n}|z_j=r))
    l = zeros(m)  # temporary vector

    for r = 1:m
        H[n,r] = 0.0
    end
    for j = n-1:-1:1
        for r = 1:m
            for s = 1:m
                l[s] = H[j+1,s] + log_T[r,s] + log_q(x[j+1],phi[s])
            end
            H[j,r] = logsumexp(l)
        end
    end
    return H
end

# Estimate the HMM parameters using the Baum-Welch algorithm.
function estimate(x,m,tolerance)
    n = length(x)  # length of observed sequence

    # randomly initialize parameters
    log_pi = log(rand(Dirichlet(m,1)))
    log_T = zeros(m,m)
    for r = 1:m; log_T[r,:] = log(rand(Dirichlet(m,1))); end
    phi = phi_init(m,x)

    # initialize EM variables
    gamma = zeros(n,m)  # gamma[j,r] = p(z_j=r | x)
    delta = zeros(n-1,m,m)  # delta[j,r,s] = p(z_j=r, z_{j+1}=s | x)
    log_m = -Inf
    log_m_old = 0.0
    n_iterations = 0

    println("\nEstimating parameters using Baum-Welch...")
    while abs(log_m - log_m_old) > tolerance
        log_m_old = log_m

        # E-step: compute gamma and delta
        G,log_m = forward(x,log_pi,log_T,phi)
        H = backward(x,log_pi,log_T,phi)
        for j = 1:n
            for r = 1:m
                gamma[j,r] = exp(G[j,r] + H[j,r] - log_m)
            end
        end
        for j = 1:n-1
            for r = 1:m
                for s = 1:m
                    delta[j,r,s] = exp(G[j,r] + log_T[r,s] + 
                        log_q(x[j+1],phi[s]) + H[j+1,s] - log_m)
                end
            end
        end

        # M-step: update log_pi, log_T, and phi
        log_pi = vec(log(gamma[1,:] ./ sum(gamma[1,:])))
        C = squeeze(sum(delta,1),1)
        log_T = log(C./sum(C,2))
        phi_max!(phi,x,gamma)

        # check change in log likelihood
        @assert(log_m > log_m_old)

        n_iterations += 1
        println(n_iterations, " ", log_m)
    end
    return log_pi,log_T,phi,log_m
end



