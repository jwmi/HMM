
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
m = 100 # 100 # number of hidden states to use
n_max = 10000000  # maximum number of words to use
from_file = false
run_profiler = false
tolerance = 0.01  # 1e-3
n_gen = 250

# Read and preprocess data
f = open("simple.txt")
#f = open("eggs.txt")
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
words = words[1:min(n_max,length(words))]  # Consider only the first n words
vocabulary = sort(unique(words))
V = length(vocabulary)  # size of vocabulary
code = Dict(zip(vocabulary,1:V))  # mapping from words to indices
# Encode the text as a numerical sequence
x = Int[code[w] for w in words]
n = length(x)

# Save to file
writedlm("x.txt",x)
writedlm("code.txt",[(1:V) vocabulary],' ')

function compose(x,vocabulary)
	s = join(vocabulary[x]," ")
	s = replace(s,r" (\W) ",s"\1 ")
	return s
end

# Sanity check of forward and backward algorithm implementations
log_pi = log(ones(m)/m)
log_T = log(ones(m,m)/m)
phi = [ones(V)/V for r = 1:m]
G = HMM.forward(x,log_pi,log_T,phi)
log_m = HMM.logsumexp(G[n,:])
println(log_m)
println(-n*log(V))


# Estimate parameters using Baum-Welch
if from_file
	(log_pi,log_T,phi,log_m) = load("estimated_params.jld","params")
else
    if run_profiler
        @profile log_pi,log_T,phi,log_m = HMM.estimate(x,m,tolerance)
        Profile.print() #format = :flat)
        Profile.clear()
    else
		log_pi,log_T,phi,log_m = HMM.estimate(x,m,tolerance)
	end
	save("estimated_params.jld","params",(log_pi,log_T,phi,log_m))
end


# Generate a new sequence from the estimated model
y,~ = HMM.generate(n_gen,log_pi,log_T,phi)

println(compose(y,vocabulary))

end # module

