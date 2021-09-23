using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
using Pkg
Pkg.add("Distributions")
using Pkg
Pkg.add("ForwardDiff")

    #Question1
    using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code
#Using Q1 from Pset3 
function mlogit_with_Z(theta, X, Z, y)
        
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end
startvals = [2*rand(7*size(X,2)).-1; .1]
td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
# run the optimizer
theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
theta_hat_mle_ad = theta_hat_optim_ad.minimizer
# evaluate the Hessian at the estimates
H  = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println([theta_hat_mle_ad theta_hat_mle_ad_se]
#Question 2
#Interpretation of gamma#
#Question 3
n.
using Distributions
include("lgwt.jl") # make sure the function gets read in
# define distribution
d = Normal(0,1) # mean=0, standard deviation=1
e.
# get quadrature nodes and weights for 7 grid points
nodes, weights = lgwt(7,-4,4)
# now compute the integral over the density and verify it's 1
sum(weights.*pdf.(d,nodes))
# now compute the expectation and verify it's 0
sum(weights.*nodes.*pdf.(d,nodes))
#computing integrals using the above
d = Normal(0,2)
nodes, weights = lgwt(7,-5,5)
sum(weights.*(nodes.).*pdf.(d,nodes))

d = Normal(0,2)
nodes, weights = lgwt(10,-5,5)
sum(weights.*(nodes.).*pdf.(d,nodes))

#Question4
