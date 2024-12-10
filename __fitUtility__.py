# General functionalities
import numpy as np
import pandas as pd
import scipy as sp

# Fit utility
import statsmodels.api as sm
import statsmodels.stats.moment_helpers as sm_help
import pymc as pm

# Miscelaneous utility
import arviz as az
import copy
import pytensor.tensor as pt



# General utility: calculate efficiency for x values given a set of a parameters
def calc_eff(x, avals, scaling=1000):
    logeff = avals[0]*np.ones_like(x)
    
    for iOrder in range(1, len(avals)):
        logeff += avals[iOrder]*np.log(x/scaling)**iOrder

    return np.exp(logeff)



##############################################
##############################################
### S T A T S M O D E L S    U T I L I T Y ###
##############################################
##############################################

# Regular statsmodels fit return fit value and if requested upper and lower parts of the confidence band
def sm_predict(x, fitresult, scaling=1000, uncertainty=False, conf=0.68):
    # If x is a single value, make it a np array of length 1
    if isinstance(x, (int, float)):
        x = np.array([x])
        
    # Extract fit parameters and prepare design matrix
    params = fitresult.params
    vals = np.zeros((len(params), len(x)))

    for i in range(len(params)):
        vals[i] = np.log(x/scaling)**i

    designX   = np.column_stack(vals)
    # Fit prediction at x values
    yPredict = designX @ fitresult.params

    
    # If not uncertainty asked, return mean values
    if not uncertainty:
        return yPredict
    
    # If uncertainty asked, calculate confidence band
    covMatrix = fitresult.cov_params()
    NminK     = fitresult.df_resid
    Tdistr    = sp.stats.t(NminK)
    tMult     = Tdistr.ppf(1-(1-conf)/2)

    ySigma= np.zeros_like(x)

    for index in range(len(x)):
        xvec = vals.T[index]
        ySigma[index] = np.sqrt(xvec @ covMatrix @ xvec)

    
    LowerCI  = yPredict - tMult*ySigma
    UpperCI  = yPredict + tMult*ySigma
    
    return yPredict, LowerCI, UpperCI



# Creates a model for the empirical fit function up to order N
def sm_logmodel(N, scaling=1000):
    model = "logeff ~ "

    for iOrder in range(1, N+1):
        if iOrder != 1:
            model += " + "
            model += f'I(np.log(E/{scaling})**{iOrder})'
        else:
            model += f'I(np.log(E/{scaling}))'
        
    return model



###############################################
###############################################
###### B A Y E S I A N    U T I L I T Y #######
###############################################
###############################################

# Perform a Bayesian fit with pymc
def likelihood_fit(data, a_init, cov_init, Sources, activities,
                    order=2, scaling=1000, beta=0,
                    nCores=8, nChains=4, nSteps=1000, 
                    saveTrace=False, nameTrace="traces/myTrace.nc"):
    
    # Initialize arrays
    E       = [None] * len(Sources)
    A       = [None] * len(Sources)
    I_arr   = [None] * len(Sources)
    t       = [None] * len(Sources)
    logC    = [None] * len(Sources)
    sigma_C = [None] * len(Sources)
    sigma_A = [None] * len(Sources)
    factor  = [None] * len(Sources)
    LL      = [None] * len(Sources)
    
    # Model neglects intensity, time, and energy uncertainty
    with pm.Model() as effModel:
        def logCModel(E, a_params, A, I, t):
            logCounts = pt.log(A*I*t) + a_params[0]
            logE = pt.log(E/scaling)

            for iOrder in range(1, order+1):
                logCounts += a_params[iOrder] * logE**iOrder

            return logCounts


        # Prior on the a parameters; sample from multivariate distribution to account for initial correlation
        a_params = pm.MvNormal('a_vals', mu=a_init, cov=cov_init, shape=order+1)
        
        
        # For each source define it's own dataset, activity prior, and likelihood
        for iSource, nameSource in enumerate(Sources):
            if beta == 0:
                # No hyperprior, just gaussian
                sigma_A[iSource] = activities[nameSource].s
            else:
                factor[iSource] = pm.Gamma(nameSource + '_factor', mu=1.0, sigma=1.0/beta)
                sigma_A[iSource] = pm.Deterministic(nameSource + '_sigmaA', activities[nameSource].s * factor[iSource])
            
            A[iSource]     = pm.Gamma(nameSource + '_A', mu=activities[nameSource].n, sigma=sigma_A[iSource])
            E[iSource]     = pm.Data(nameSource + "_E", data[nameSource]['E'])
            I_arr[iSource] = np.array(data[nameSource]['I'])
            t[iSource]     = np.array(data[nameSource]['t'])

            logC[iSource]    = pm.Data(nameSource + '_logC', data[nameSource]['logC'])
            sigma_C[iSource] = np.array(data[nameSource]['logC_err'])

            model          = logCModel(E[iSource], a_params, A[iSource], I_arr[iSource], t[iSource])
            LL[iSource]    = pm.Normal(nameSource + "_LL", 
                                       mu=model, sigma=sigma_C[iSource], observed=logC[iSource])
            
        # Sampling procedure
        print("Start sampling...")
        trace = pm.sample(nSteps, chains=nChains, cores=nCores, return_inferencedata=True)
        print("Done")
        
        # Save trace if requested
        if saveTrace:
            trace.to_netcdf(nameTrace)
            print("Saved the trace")
        
        return trace


# Initial statsmodels fit + proper fit
def bayesian_fit(data, dictData, activities, Sources, nameTrace, newTrace=True, order=2, scaling=1000, beta=0, cov_mult=4, nCores=8, nChains=4, nSteps=1000):
    # Make initial fit
    model   = sm_logmodel(order, scaling=scaling)
    initFit = sm.WLS.from_formula(formula=model, data=data, weights=(data['logeff_err'])**(-2)).fit()
    a_init   = list(initFit.params)
    cov_init = cov_mult*np.array(initFit.cov_params())

    # Sampling
    nameTrace = f'traces/{nameTrace}.nc'

    if newTrace:
        trace = likelihood_fit(dictData, a_init, cov_init, Sources, activities,
                            order=order, scaling=scaling, beta=beta,
                            nCores=nCores, nChains=nChains, nSteps=nSteps,
                            saveTrace=True, nameTrace=nameTrace)
    else:
        trace = az.from_netcdf(nameTrace)
        
    return trace


# Relative efficiency compared to reference point
def bayes_relative_efficiency(E_linspace, refEnergy, nameTrace, scaling=1000):
    trace    = az.from_netcdf(f'traces/{nameTrace}.nc')
    stacked  = trace.posterior.stack(draws=("chain", "draw"))
    avals    = stacked.a_vals.values
    nSamples = len(avals.T)

    relEff = np.zeros((nSamples, len(E_linspace)))
    refEffSample  = np.zeros(nSamples)
    RelEffMedian  = np.zeros(len(E_linspace))
    RelEffConfNeg = np.zeros(len(E_linspace))
    RelEffConfPos = np.zeros(len(E_linspace))

    for iSample in range(nSamples):
        eff_at_E = calc_eff(E_linspace, avals.T[iSample], scaling=scaling)
        
        refEffSample[iSample] = calc_eff(refEnergy, avals.T[iSample], scaling=scaling)
        relEff[iSample]       = eff_at_E/refEffSample[iSample]


    for iEnergy in range(len(E_linspace)):
        RelEffMedian[iEnergy]    = np.quantile(relEff.T[iEnergy], q=0.5)
        RelEffConfNeg[iEnergy] = np.quantile(relEff.T[iEnergy], q=0.317/2)
        RelEffConfPos[iEnergy] = np.quantile(relEff.T[iEnergy], q=1-0.317/2)
    
    refEff = np.quantile(refEffSample, q=0.5)

    return RelEffMedian, RelEffConfNeg, RelEffConfPos, refEff

    

# Calculate the ratio of the median of two traces
def bayes_ratio(E, nameTrace1, nameTrace2, scaling=1000):
    # Median for trace 1
    trace1    = az.from_netcdf(nameTrace1)
    stack1    = trace1.posterior.stack(draws=("chain", "draw"))
    avals1    = stack1.a_vals.values
    nSamples1 = len(avals1.T)
    nPoints   = len(E)

    firstSample = np.zeros((nSamples1, nPoints))
    firstMedian = np.zeros(nPoints)
    
    for iSample in range(nSamples1):
        firstSample[iSample] = calc_eff(E, avals1.T[iSample], scaling=scaling)
        
    for iEnergy in range(nPoints):
        firstMedian[iEnergy] = np.quantile(firstSample.T[iEnergy], q=0.5)


    # Median for trace 2
    trace2    = az.from_netcdf(nameTrace2)
    stack2    = trace2.posterior.stack(draws=("chain", "draw"))
    avals2    = stack2.a_vals.values
    nSamples2 = len(avals2.T)
    nPoints   = len(E)

    secondSample = np.zeros((nSamples2, nPoints))
    secondMedian = np.zeros(nPoints)
    
    for iSample in range(nSamples2):
        secondSample[iSample] = calc_eff(E, avals2.T[iSample], scaling=scaling)

    for iEnergy in range(nPoints):
        secondMedian[iEnergy] = np.quantile(secondSample.T[iEnergy], q=0.5)


    # Calculate relative deviation
    ratio = 100 * ((firstMedian / secondMedian) - 1)
    return ratio


