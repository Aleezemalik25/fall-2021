using Pkg
Pkg.add("MultivariateStats")
using Optim 
using GLM 
using LinearAlgebra 
using Random 
using Statistics 
using DataFrames 
using DataFramesMeta 
using CSV 
using MultivariateStats 
using HTTP
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://github.com/OU-PhD-Econometrics/fall-2021/blob/master/ProblemSets/PS8-factor/nlsy.csv"

using DataFrames;
df = CSV.read(HTTP.get(url).body, DataFrame)
ols = glm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
print(ols)
#I tried to make this code run but the variables are string and I read we need to use Dict to make the actual variable names appear so it wouldn't give an error but I don't know how to apply Dict to a code#

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::
#I think we would make a matrix of the asvab variables and then run a correlation of the new matrix which would be a submatrix. I had a bit of difficulty writing the code here#


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::
olsasvab= glm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
print(olsasvab)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::
M=fit(PCA, asvabMat; maxoutdim=1)
asvabPCA=MultivariateStats.transform(M,asvabMat)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 5
#:::::::::::::::::::::::::::::::::::::::::::::::::::
M=fit(FactorAnalysis, asvabMat; maxoutdim=1)
asvabFactorAnalysis=MultivariateStats.transform(M,asvabMat)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::

