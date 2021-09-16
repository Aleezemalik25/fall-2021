using Pkg
Pkg.add("HTTP")
using Pkg
Pkg.add("Optim")
using Pkg
Pkg.add("GLM")
using Pkg
Pkg.add("LinearAlgebra")
using Pkg
Pkg.add("Random")
using Pkg
Pkg.add("Statistics")
using Pkg
Pkg.add("DataFrames")
using Pkg
Pkg.add("CSV")
using Pkg
Pkg.add("FreqTables")
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
function ps3()
    #Question1
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

function mlogit(Beta, X, Z, y)

    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigBeta = [reshape(Beta,K,J-1) zeros(K)]

    bigZ = zeros(N,J)
    for j=1:J
        bigZ[:,j] = Z[:,j]-Z[:,J]
    end

    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        XZ=cat(X,bigZ[:,j],dims=3)
        num[:,j] = exp.(XZ*bigBeta[:,j])
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

mlogit_hat_optim = optimize(b-> mlogit(b,X,Z, y), rand(7*(size(X,2))), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(mlogit_hat_optim.minimizer)
#Question2
#Gamma represents the change in utility in percentage terms#
#Question3#
function nested_logit(alpha, X, Z, y, nesting_structure)
    beta = alpha[1:end-3]
    lambda = alpha[end-2:end-1]
    gamma = alpha[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigBeta = [repeat(beta[1:K],1,length(nesting_structure[1])) repeat(beta[K+1:2K],1,length(nesting_structure[2])) zeros(K)]
end
ps3()
