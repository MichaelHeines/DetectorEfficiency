import numpy as np
import arviz as az

from __fitUtility__ import *


# From the trace, extract the fit model and the 1 sigma error
def trace_to_predict(trace, E_linspace, order=2, scaling=1000):
    # Get distribution of the a parameters
    stacked   = trace.posterior.stack(draws=("chain", "draw"))
    avals     = stacked.a_vals.values
    nSamples  = len(avals.T)
    nLinspace = len(E_linspace)
    
    # Efficiency for a certain set of a parameters
    bayesianPredict = np.zeros((nSamples, nLinspace))
    bayesianMedian  = np.zeros(nLinspace)
    bayesianConfNeg = np.zeros(nLinspace)
    bayesianConfPos = np.zeros(nLinspace)

    for iSample in range(nSamples):
        bayesianPredict[iSample] = calc_eff(E_linspace, avals.T[iSample], scaling=scaling)

    for iEnergy in range(nLinspace):
        bayesianMedian[iEnergy]   = np.quantile(bayesianPredict.T[iEnergy], q=0.5)
        bayesianConfNeg[iEnergy]  = np.quantile(bayesianPredict.T[iEnergy], q=0.317/2)
        bayesianConfPos[iEnergy]  = np.quantile(bayesianPredict.T[iEnergy], q=1-0.317/2)

    return bayesianMedian, bayesianConfNeg, bayesianConfPos


# From the trace, extract the activities with corresponding 1 sigma errors
def trace_to_activity(trace, Sources):
    # Get distribution of the activities
    stacked  = trace.posterior.stack(draws=("chain", "draw"))    
    nSamples = len(stacked[Sources[0] + "_A"])
    nSources = len(Sources)

    activities = {}

    for source in Sources:
        bayesianPredict = stacked[source + "_A"].values
        value = np.quantile(bayesianPredict, q=0.5)
        lower = np.quantile(bayesianPredict, q=0.317/2)
        upper = np.quantile(bayesianPredict, q=1-0.317/2)
        error = 0.5*(upper - lower)

        activities[source] = [value, error]

    return activities


# From trace get efficiency at energy
def trace_to_efficiency(trace, energy, order=2, scaling=1000):
    # Get distribution of the a parameters
    stacked   = trace.posterior.stack(draws=("chain", "draw"))
    avals     = stacked.a_vals.values
    nSamples  = len(avals.T)
    
    bayesianPredict = np.zeros(nSamples)
    
    for iSample in range(nSamples):
        bayesianPredict[iSample] = calc_eff(energy, avals.T[iSample], scaling=scaling)

    bayesianMedian  = np.quantile(bayesianPredict, q=0.5)
    bayesianConfNeg = np.quantile(bayesianPredict, q=0.317/2)
    bayesianConfPos = np.quantile(bayesianPredict, q=1-0.317/2)

    return bayesianMedian, bayesianConfNeg, bayesianConfPos

