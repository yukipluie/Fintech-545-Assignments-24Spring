import pandas as pd
import numpy as np
from scipy.stats import norm, t
import sys
sys.path.append('../RiskManagement')
import risk_management as rm

# Read the data
x = pd.read_csv('problem1.csv')

# a. Using a normal distribution with an exponentially weighted variance (lambda=0.97)
def weights(lam,t):
    tw = 0
    w = np.zeros(t)
    for i in range(t):
        w[i] = (1-lam)*lam ** (t-i-1)
        tw += w[i]
    for i in range(t):
        w[i] = w[i]/tw
    return w

def ew_cov(df,lam, corr=False):
    '''
    Args:
    df: pandas dataframe
    lam: float
    corr: boolean
    Returns:
    cov: numpy array
    '''
    n = df.shape[1]
    t = df.shape[0]
    w = weights(lam,t)
    means = np.array(df.mean())
    xhat = df.copy()
    for i in range(n):
        xhat.iloc[:,i]=xhat.iloc[:,i]-means[i]
    result = xhat.multiply(w,axis=0).T @ xhat
    if corr:
        d = np.diag(result)
        result = result/np.sqrt(np.outer(d,d))
    return result

alpha = 0.05
lam = 0.97
cov_matrix = ew_cov(x,lam)
fd = rm.fit_normal_ewcov(x, lam)
var_absolute = rm.VaR_distribution(fd.error_model, alpha)
var_diff_from_mean = -norm.ppf(0.05, loc=0, scale=fd.error_model.std())
print("VaR_ew_absolute: ", var_absolute)
print("VaR_ew_diff_from_mean: ", var_diff_from_mean)

es_absolute = rm.ES_distribution(fd.error_model, 0.05)
es_diff_from_mean = rm.ES_distribution(norm(loc=0, scale=fd.error_model.std()), 0.05)
print("ES_ew_absolute: ", es_absolute)
print("ES_ew_diff_from_mean: ", es_diff_from_mean)

# b. Using a MLE fitted T distribution
fd = rm.fit_general_t(x)
var_absolute = rm.VaR_distribution(fd.error_model, alpha)
var_diff_from_mean = -t.ppf(alpha, df=fd.params[2], loc=0, scale=fd.params[1])
print("VaR_t_absolute: ", var_absolute)
print("VaR_t_diff_from_mean: ", var_diff_from_mean)
es_absolute = rm.ES_distribution(fd.error_model, alpha)
es_diff_from_mean = rm.ES_distribution(t(df=fd.params[2], loc=0, scale=fd.params[1]), alpha)
print("ES_t_absolute: ", es_absolute)
print("ES_t_diff_from_mean: ", es_diff_from_mean)


# c. Using a Historic Simulation
VaR_hist = rm.VaR_hist(x, alpha)
es = rm.ES_hist(x, alpha)
print("VaR_hist: ", VaR_hist)
print("ES: ", es)



