import numpy as np
import pandas as pd
import math
import scipy.stats as stats
from scipy.integrate import quad
from scipy.stats import norm,t,kurtosis
from scipy.optimize import minimize


# 1. Covariance estimation techniques
def coveriance_matrix(m):
    '''Calculate the sample covariance matrix of a given matrix m
    Args:
    m: a matrix of size n x p
    Returns:
    covariance_matrix: a covariance matrix of size p x p
    '''
    mean_vector = np.mean(m, axis=0)
    centered_data = m - mean_vector
    covariance_matrix = np.dot(centered_data.T, centered_data) / (m.shape[0] - 1)
    return covariance_matrix

# Part of exponentially_weighted_covariance: Generate weight
def weights(lamda,t):
    tw = 0
    w = np.zeros(t)
    for i in range(t):
        w[i] = (1-lamda)*lamda ** (t-i-1)
        tw += w[i]
    for i in range(t):
        w[i] = w[i]/tw
    return w

# EW cov + var
def exponentially_weighted_covariance(df,lamda):
    n = df.shape[1]
    t = df.shape[0]
    w = weights(lamda,t)
    means = np.array(df.mean())
    xhat = df.copy()
    for i in range(n):
        xhat.iloc[:,i]=xhat.iloc[:,i]-means[i]
    cov = xhat.multiply(w,axis=0).T @ xhat
    return cov

# EW corr + Var
def exponential_weighted_variance(df, lam=0.97):
    T = df.shape[0]
    weights = np.exp(np.linspace(0, -(T-1), T)*np.log(lam))
    weighted_var = np.average(df**2, axis=0, weights=weights)
    return weighted_var

# Pearson corr + var
def pearson_correlation(df):
    cov_matrix = np.cov(df.T)
    cor_matrix = np.corrcoef(df.T)
    return cor_matrix

def variance(df):
    var_vector = np.var(df, axis=0)
    return var_vector

# Pearson correlation and EW variance
def pearson_covariance_ew_covar(df,lamda):
    var_vector_ew = exponential_weighted_variance(df.iloc[:, 1:])
    cor_matrix_pearson = pearson_correlation(df.iloc[:, 1:])
    cov_matrix_ew_var = cor_matrix_pearson * np.sqrt(np.outer(var_vector_ew, var_vector_ew))
    return cov_matrix_ew_var

# Non PSD fixes for correlation matrices
def is_psd(matrix):
    eigenvalues = np.linalg.eigh(matrix)[0]
    return np.all(eigenvalues >= -1e-8)

def chol_psd(a):
    n = a.shape[0]
    root = np.zeros((n,n))

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        temp = a[j,j] - s
        if temp <= 0 and abs(temp) < 1e-8:
            temp = 0.0
        
        root[j,j] = np.sqrt(temp)
        if root[j,j] == 0.0:
            root[j,(j+1):n] = 0.0
        else:
            ir = 1.0 / root[j,j]
            for i in range(j+1,n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i,j] = (a[i,j] - s) * ir
    return root

def near_psd(a, epsilon=0):
    invSD = None
    out = a.copy()
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    print(vals)
    T = 1.0 / np.dot((vecs * vecs), vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(np.dot(invSD, out), invSD)
    return out

# The first projection for Higham method which assume that the weight matrix is diagonal
def pu(x):
    n = x.shape[0]
    x_pu = x.copy()
    for i in range(n):
        for j in range(n):
            if i==j:
                x_pu[i][j]=1
    return x_pu

# The second projection for Higham method
def ps(x,w=None):
    n = x.shape[0]
    if w != None:
        w_diag = np.diag(w)
    else:
        w_diag = np.diag(np.ones(n))
    x_w = np.sqrt(w_diag) @ x @ np.sqrt(w_diag)
    vals, vecs = np.linalg.eigh(x_w)
    vals[vals<1e-8]=0
    l = np.diag(vals)
    x_pos = vecs @ l @ vecs.T
    w_inv = np.linalg.inv(np.sqrt(w_diag))
    out = w_inv @ x_pos @ w_inv
    return out

# Frobenius Norm
def fnorm(x):
    n = x.shape[0]
    result = 0
    for i in range(n):
        for j in range(n):
            result += x[i][j] ** 2
    return result

# Higham's 2002 nearest psd correlation function
def higham(a,gamma0=np.inf,K=100,tol=1e-08):
    delta_s = [0]
    gamma = [gamma0]
    Y = [a]
    for k in range(1,K+1):
        R_k = Y[k-1] - delta_s[k-1]
        X_k = ps(R_k)
        delta_s_k = X_k - R_k
        delta_s.append(delta_s_k)
        Y_k = pu(X_k)
        Y.append(Y_k)
        gamma_k = fnorm(Y_k-a)
        gamma.append(gamma_k)
        if gamma_k -gamma[k-1] < tol:
            vals = np.linalg.eigh(Y_k)[0]
            if vals.min() >= 1e-8:
                break
            else:
                continue
    return Y[-1]






# Simulation Methods
#normal simulation
def normal_sim(a,nsim,seed,means=[],fixmethod=near_psd):
    eigval_min = np.linalg.eigh(a)[0].min()
    if eigval_min < 1e-08:
        a = fixmethod(a)
    l = chol_psd(a)
    m = l.shape[0]
    np.random.seed(seed)
    z = np.random.normal(size=(m,nsim))
    X = (l @ z).T
    if means.size != 0:
        if means.size != m:
            raise Exception("Mean size does not match with cov")
        for i in range(m):
            X[:,i] = X[:,i] + means[i]
    return X

def pca_vecs(cov):
    eigvalues, eigvector = np.linalg.eigh(cov)
    vals = np.flip(eigvalues)
    vecs = np.flip(eigvector,axis=1)
    posv_ind = np.where(vals >= 1e-8)[0]
    vals = vals[posv_ind]
    vecs = vecs[:,posv_ind]
    vals = np.real(vals)
    return vals,vecs

def vals_pct(vals,vecs,pct):
    tv = vals.sum()
    for k in range(len(vals)):
        explained = vals[:k+1].sum()/tv
        if explained >= pct:
            break
    return vals[:k+1],vecs[:,:k+1]

# pca simulation
def pca_sim(a,nsim,seed,means=[],pct=None):
    vals,vecs = pca_vecs(a)
    if pct != None:
        vals,vecs = vals_pct(vals,vecs,pct)
    B = vecs @ np.diag(np.sqrt(vals))
    m = vals.size
    np.random.seed(seed)
    r = np.random.normal(size=(m,nsim))
    out = (B @ r).T
    if means.size != 0:
        if means.size != out.shape[1]:
            raise Exception("Mean size does not match with cov")
        for i in range(m):
            out[:,i] = out[:,i] + means[i]
    return out

# Historical Simulation
def return_calculate(df,method):
    if df.columns[0]=='Date':
        ind = df.columns[1:]
        datesig = True
    else:
        ind = df.columns
        datesig = False
    p = df.loc[:,ind]
    n = p.shape[1]
    t = p.shape[0]
    p2 = np.zeros((t-1,n))
    for i in range(t-1):
        for j in range(n):
            p2[i,j]=p.iloc[i+1,j]/p.iloc[i,j]
    if method.upper()== "DISCRETE":
        p2 = p2 -1
    elif  method.upper()== "LOG":
        p2 = np.log(p2)
    else:
        raise Exception("Method be either discrete or log")
    out = pd.DataFrame(data=p2,columns=ind)
    if datesig == True:
        out.insert(0,'Date',np.array(df.loc[1:,'Date']))
    return out

def port_cal(port,stockdata,portdata,method="discrete"):
    if port == "All":
        port_prices = stockdata.loc[:,portdata['Stock']]
        port_info = portdata
    else:
        port_info = portdata[portdata['Portfolio']==port]
        port_prices = stockdata.loc[:,port_info['Stock']]

    cur_price = port_prices.iloc[-1,:]
    cur_value = (cur_price * np.array(port_info['Holding'])).sum()

    r = return_calculate(port_prices,method=method)
    return r,cur_price,cur_value,port_info

def sim_his(port,stockdata,portdata,seed,nsim=10000,alpha=0.05):
    r,cur_price,cur_value,portinfo = port_cal(port, stockdata, portdata)
    np.random.seed(seed)
    r_sim = r.sample(nsim,replace=True)
    p_new = (1+r_sim).mul(cur_price)
    port_value = p_new.mul(portinfo['Holding'].values).sum(axis=1)
    profit = port_value- cur_value
    profit = profit.to_numpy(copy=True)
    var = VaR(profit, alpha)
    es = ES(profit, alpha)
    return var,es

# Monte Carlo Simulation
def sim_mc(port,stockdata,portdata,seed,nsim=10000,alpha=0.05):
    r,cur_price,cur_value,portinfo = port_cal(port, stockdata, portdata)
    r_h = r.sub(r.mean(),axis=1)
    sigma= exponentially_weighted_covariance(r_h, 0.94)
    r_sim = pca_sim(sigma, nsim, seed, means=r.mean(), pct=None)
    r_sim =pd.DataFrame(r_sim,columns=r.columns)
    p_new = (r_sim+1).mul(cur_price)
    port_value = p_new.mul(portinfo['Holding'].values).sum(axis=1)
    profit = port_value- cur_value
    profit = profit.to_numpy(copy=True)
    var = VaR(profit, alpha)
    es = ES(profit, alpha)
    return var,es

# VaR calculation methods
def get_portfolio_price(portfolio, prices, portfolio_name, Delta=False):
    if portfolio_name == "All":
        assets = portfolio.drop('Portfolio',axis=1)
        assets = assets.groupby(["Stock"], as_index=False)["Holding"].sum()
    else:
        assets = portfolio[portfolio["Portfolio"] == portfolio_name]     
    stock_codes = list(assets["Stock"])
    assets_prices = pd.concat([prices["Date"], prices[stock_codes]], axis=1) 
    # print(assets_prices)
    current_price = np.dot(prices[assets["Stock"]].tail(1), assets["Holding"])
    holdings = assets["Holding"]   
    if Delta == True:
        asset_values = assets["Holding"].values.reshape(-1, 1) * prices[assets["Stock"]].tail(1).T.values
        delta = asset_values / current_price       
        # print("This is delta", delta)
        return current_price, assets_prices, delta   
    return current_price, assets_prices, holdings

# Calculate with Delta Normal
def calculate_delta_var(portfolio, prices, alpha=0.05, lambda_=0.94, portfolio_name="All"):
    current_price, assets_prices, delta = get_portfolio_price(portfolio, prices, portfolio_name, Delta=True)
    returns = return_calculate(assets_prices, dateColumn="Date").drop('Date', axis=1)
    assets_cov = exponentially_weighted_covariance(returns, lambda_)
    p_sig = np.sqrt(np.transpose(delta) @ assets_cov @ delta)
    var_delta = -current_price * stats.norm.ppf(alpha) * p_sig
    return current_price[0], var_delta[0][0]

# Calculate with historical simulation
def calculate_historic_var(portfolio, prices, alpha=0.05,n_simulation=1000, portfolio_name="All"):
    current_price, assets_prices, holdings = get_portfolio_price(portfolio, prices, portfolio_name)  
    returns = return_calculate(assets_prices, dateColumn="Date").drop("Date", axis=1)  
    assets_prices = assets_prices.drop('Date',axis=1)
    sim_returns = returns.sample(n_simulation, replace=True)
    sim_prices = np.dot(sim_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)   
    var_hist = -np.percentile(sim_prices, alpha*100) 
    return current_price[0], var_hist, sim_prices

def VaR(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    return -v



def VaR(a, alpha=0.05):
    """Calculate the Value at Risk (VaR) for a dataset."""
    x = np.sort(a)
    n = len(a)
    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    v = 0.5 * (x[nup - 1] + x[ndn - 1])  # Adjusted indexing for Python (0-based)
    return -v

def VaR_distribution(d, alpha=0.05):
    """Calculate the Value at Risk (VaR) for a distribution."""
    return -d.ppf(alpha)


# ES calculation
def ES(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    es = a[a<=v].mean()
    return -es

def ES(a, alpha=0.05):
    """Calculate the Expected Shortfall (ES) for a dataset."""
    x = np.sort(a)
    n = len(a)

    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    v = 0.5 * (x[nup - 1] + x[ndn - 1])  # Adjusted indexing for Python (0-based)
    es = np.mean(x[x <= v])
    return -es


def ES_distribution(d, alpha=0.05):
    """Calculate the Expected Shortfall (ES) for a distribution."""
    v = VaR_distribution(d, alpha=alpha)
    
    def integrand(x):
        return x * d.pdf(x)
    
    st, _ = quad(integrand, d.ppf(1e-12), -v) #quad is the function used for integration.
    return -st / alpha