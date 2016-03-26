
# HMM with discrete emission distributions, assumed to be on {1,...,K}.
module DiscreteHMM
	using Distributions

	# Emission distribution specification
	log_q(x,p) = log(p[x])  # log density
	qrnd(p) = rand(Categorical(p))  # random sample

	# Initialize an array of emission distribution parameters
	function phi_init(m,x)
		K = maximum(x)
	    return [rand(Dirichlet(K,1)) for s = 1:m]
	end

	# Maximum likelihood estimate of array of emission distribution
	# parameters using data x with weights gamma.
	function phi_max!(phi,x,gamma)
	    m = length(phi)
	    n = length(x)
	    for s = 1:m
	    	ph = phi[s]
	    	ph[:] = 0.0
	        for j = 1:n
	            ph[x[j]] += gamma[j,s]
	        end
	        ph[:] /= sum(ph)
	    end
	end

	# Core HMM code
	include("hmm.jl")
end # module


# HMM with univariate normal emission distributions.
module NormalHMM
	using Distributions

	# Emission distribution specification
	log_q(x,D) = logpdf(D,x)  # log density
	qrnd(D) = rand(D)  # random sample

	# Initialize an array of emission distribution parameters
	function phi_init(m,x)
		D_m = Normal(mean(x),std(x))
		D_v = InverseGamma(1.0,var(x))
	    return [Normal(rand(D_m),sqrt(rand(D_v))) for s = 1:m]
	end

	# Maximum likelihood estimate of array of emission distribution
	# parameters using data x with weights gamma.
	function phi_max!(phi,x,gamma)
	    @assert(false,"This function is not yet implemented.")
	end

	# Core HMM code
	include("hmm.jl")
end # module


# HMM with multivariate normal emission distributions, 
# with diagonal covariance matrices.
module MVNdiagHMM
	using Distributions

	# Emission distribution specification
	log_q(x,D) = logpdf(D,x)  # log density
	qrnd(D) = rand(D)  # random sample

	# Initialize an array of emission distribution parameters
	function phi_init(m,x)
		@assert(false,"This function is not yet implemented.")
	end

	# Maximum likelihood estimate of array of emission distribution
	# parameters using data x with weights gamma.
	function phi_max!(phi,x,gamma)
	    @assert(false,"This function is not yet implemented.")
	end

	# Core HMM code
	include("hmm.jl")
end # module


