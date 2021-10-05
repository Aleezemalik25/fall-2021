import Pkg; Pkg.add("DataFramesMeta")
using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
# read in function to create state transitions for dynamic model
#The function didn't work for me for some reason - I tried it a couple of times but couldn't resolve the error so I'm copying the code from the file you uploaded which creates the grid#
function create_grids()

    function xgrid(theta,xval)
        N      = length(xval)
        xub    = vcat(xval[2:N],Inf)
        xtran1  = zeros(N,N)
        xtran1c = zeros(N,N)
        lcdf   = zeros(N)
        for i=1:length(xval)
            xtran1[:,i]   = (xub[i] .>= xval).*(1 .- exp.(-theta*(xub[i] .- xval)) .- lcdf)
            lcdf        .+= xtran1[:,i]
            xtran1c[:,i] .+= lcdf
        end
        return xtran1, xtran1c
    end

    zval   = collect([.25:.01:1.25]...)
    zbin   = length(zval)
    xval   = collect([0:.125:25]...)
    xbin   = length(xval)
    tbin   = xbin*zbin
    xtran  = zeros(tbin,xbin)
    xtranc = zeros(xbin,xbin,xbin)
    for z=1:zbin
        xtran[(z-1)*xbin+1:z*xbin,:],xtranc[:,:,z] = xgrid(zval[z],xval)
    end

    return zval,zbin,xval,xbin,xtran
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# load in the data
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
# create bus id variable
df = @transform(df, bus_id = 1:size(df,1))

#---------------------------------------------------
# reshape from wide to long (must do this twice be-
# cause DataFrames.stack() requires doing it one 
# variable at a time)
#---------------------------------------------------
# first reshape the decision variable
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))

# next reshape the odometer variable
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))

# join reshaped df's back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model using GLM
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
theta_hat = glm(@formula(Y ~ Odometer + Branded ), df_long, Binomial(), LogitLink())
println(theta_hat)
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3a,b,c: read in data for dynamic model
#:::::::::::::::::::::::::::::::::::::::::::::::::::
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
# [ convert other data frame columns to matrices]
zval,zbin,xval,xbin,xtran = create_grids()
@views @inbounds function myfun(theta)   #3f?#
N=size(xtran,1) #
T=size(Y,2) #
FV=zeros(N,2,T+1) # 
#Now write four nested for loops over each of the possible states#
    for t in T:-1:1 #Loop backwards over t from T +1 to 1#
    for b in 0:1#Loop over the two possible brand states {0,1}#
    for z in 1:zbin#Loop over the possible permanent route usage states (i.e. from 1 to zbin)#
    for x in 1:xbin#Loop over the possible odometer states (i.e. from 1 to xbin#
    rowtm=x+(z-1)*xbin #Create an object that marks the row of the transition matrix that we need to - This will be x + (z-1)*xbin (where x indexes the mileage bin and z indexes the route usage bin), given how the xtran matrix was constructed in the create grids() function.#
    end
    #conditional value function#
v1=theta[1]+theta[2]*xval[x]+theta[3]*b+ beta*xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
v0=beta*xtran[1+(z-1)*xbin,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
FV[row,b+1,t]=beta*log(exp(v1)+exp(v0))
beta=0.9
#3d#
loglikelihood=0
v_0=beta*xtran[1+(:Zst-1)*xbin,:]'*FV[(:Zst-1)*xbin+1:Zst*xbin,b+1,t+1]
v_1=theta[1]+theta[2]*xval[:xst]+theta[3]*b+ beta*xtran[row,:]'*FV[(:zst-1)*xbin+1:z*xbin,b+1,t+1]
#3e#
return=-loglikelihood