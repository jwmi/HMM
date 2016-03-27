# Demo using DiscreteHMM and Baum-Welch algorithm 
# to estimate a simple model of natural language.
include("models.jl")

module demo_discrete
using DiscreteHMM
can_save = (Pkg.installed("JLD")!=nothing)  # check if JLD is installed
if can_save; using JLD; end

# Settings
m = 100  # number of hidden states to use in estimated model
n_max = 100000  # maximum number of words/symbols to use
from_file = false  # load estimated parameters from file
run_profiler = false  # run profiler to assess performance
tolerance = 1e-2  # convergence tolerance for Baum-Welch
n_gen = 1000  # length of sequence to generate using estimated parameters
filename = "simple.txt"  # file containing a sample of simple natural language

# Read and preprocess data
function preprocess(filename,n_max)
	# Read file
	f = open(filename)
	s = readall(f)
	close(f)
	# Remove whitespace and remove some punctuation
	s = replace(s,r"\s+"," ")
	s = replace(s,r"[\"]","")
	s = replace(s,r"' | '"," ")
	s = replace(s,r"(\W) | (\W)",s" \1 ")
	s = strip(s)
	words = split(s,' ')
	# Consider only the first n_max words/symbols
	words = words[1:min(n_max,length(words))]
	# Extract vocabulary
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

# Preprocess and encode the text in the given file
words,vocabulary,code = preprocess(filename,n_max)
V = length(vocabulary)
x = encode(words,code)
n = length(x)

# Save encoding to file
#writedlm("x.txt",x)
#writedlm("code.txt",[(1:V) vocabulary],' ')


# Sanity check of forward algorithm implementation
if true
	log_pi = log(ones(m)/m)
	log_T = log(ones(m,m)/m)
	phi = [ones(V)/V for r = 1:m]
	G,log_m = DiscreteHMM.forward(x,log_pi,log_T,phi)
	log_m_exact = -n*log(V)
	@assert(abs(log_m - log_m_exact) < 1e-10, "Log marginal does not match")
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
println("Generating a sample of text using the estimated model:")
y,zy = DiscreteHMM.generate(n_gen,log_pi,log_T,phi)
println(compose(y,vocabulary))

end # module

