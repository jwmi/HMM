
module HMM
	using Distributions

	log_q(x,p) = log(p[x])
	qrnd(p) = rand(Categorical(p))

	function phi_init(m,x)
		V = maximum(x)
	    return [rand(Dirichlet(V,1)) for s = 1:m]
	end

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

	include("hmm.jl")
end # module


module test_text
using HMM
using JLD

# Settings
m = 10  # number of hidden states to use
from_file = false
tolerance = 1e-2
n_gen = 500

# Read and preprocess data
f = open("agrange.txt")
s = readall(f)
close(f)
# Remove whitespace and some punctuation
s = replace(s,r"\s+"," ")
s = replace(s,r"[\"]","")
s = replace(s,r"' | '"," ")
s = replace(s,r"(\W) | (\W)",s" \1 ")
s = strip(s)
# Extract vocabulary
words = split(s,' ')
vocabulary = sort(unique(words))
V = length(vocabulary)  # size of vocabulary
code = Dict(zip(vocabulary,1:V))  # mapping from words to indices
# Encode the text as a numerical sequence
x = Int[code[w] for w in words]

function compose(x,vocabulary)
	s = join(vocabulary[x]," ")
	s = replace(s,r" (\W) ",s"\1 ")
	return s
end


# Estimate parameters using Baum-Welch
if from_file
	(log_pi,log_T,phi,log_m) = load("estimated_params.jld","params")
else
	log_pi,log_T,phi,log_m = HMM.estimate(x,m,tolerance)
	save("estimated_params.jld","params",(log_pi,log_T,phi,log_m))
end


# Generate a new sequence from the estimated model
y,~ = HMM.generate(n_gen,log_pi,log_T,phi)

println(compose(y,vocabulary))

end # module

