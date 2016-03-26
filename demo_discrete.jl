# Demo using DiscreteHMM and Baum-Welch algorithm 
# to estimate a simple model of natural language.
include("models.jl")

module demo_discrete
using DiscreteHMM
can_save = (Pkg.installed("JLD")!=nothing)
if can_save; using JLD; end

# Settings
m = 100 # 100 # number of hidden states to use
n_max = 100000  # maximum number of words to use
from_file = false
run_profiler = false
tolerance = 1e-2  # 1e-3
n_gen = 1000
filename = "simple.txt"

# Read and preprocess data
function preprocess(filename,n_max)
	# Read file
	f = open(filename)
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
	V = length(vocabulary)
	code = Dict(zip(vocabulary,1:V))  # mapping from words to indices
	return words,vocabulary,code
end

# Produce the numerical sequence corresponding to a sequence of words.
function encode(words,code)
	return Int[code[w] for w in words]
end

# Produce the sequence of text corresponding to a numerical sequence.
function compose(x,vocabulary)
	s = join(vocabulary[x]," ")
	s = replace(s,r" (\W) ",s"\1 ")
	return s
end
words,vocabulary,code = preprocess(filename,n_max)
V = length(vocabulary)

# Encode the text as a numerical sequence
x = encode(words,code)
n = length(x)

# Save to file
#writedlm("x.txt",x)
#writedlm("code.txt",[(1:V) vocabulary],' ')


# Sanity check of forward and backward algorithm implementations
if false
	log_pi = log(ones(m)/m)
	log_T = log(ones(m,m)/m)
	phi = [ones(V)/V for r = 1:m]
	G = DiscreteHMM.forward(x,log_pi,log_T,phi)
	log_m = DiscreteHMM.logsumexp(G[n,:])
	println(log_m)
	println(-n*log(V))
end


# Estimate parameters using Baum-Welch
if from_file
	(log_pi,log_T,phi,log_m) = load("estimated_params.jld","params")
else
    if run_profiler
        @profile log_pi,log_T,phi,log_m = DiscreteHMM.estimate(x,m,tolerance)
        Profile.print() #format = :flat)
        Profile.clear()
    else
		log_pi,log_T,phi,log_m = DiscreteHMM.estimate(x,m,tolerance)
	end
	if can_save
		save("estimated_params.jld","params",(log_pi,log_T,phi,log_m))
	end
end


# Generate a new sequence from the estimated model
y,~ = DiscreteHMM.generate(n_gen,log_pi,log_T,phi)
println(compose(y,vocabulary))

end # module

