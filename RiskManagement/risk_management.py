import pandas as pd
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm, t, skew, kurtosis
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.linalg import eigh
import scipy.linalg as la
from scipy import stats
from scipy import optimize
import statsmodels.api as sm
from fitter import Fitter


# 1. Covariance estimation techniques
def generate_with_missing(n, m, pmiss=0.25):
    x = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            if np.random.rand() >= pmiss:
                x[i, j] = np.random.randn()
            else:
                x[i, j] = np.nan
    return x

np.random.seed(2)
x = generate_with_missing(10, 5, pmiss=0.2)

# Calculate covariance
cov_matrix = np.cov(x, rowvar=False)


def missing_cov(x, skip_miss=True, func= "cov"):
    n, m = x.shape    # n: number of rows, m: number of columns, 10, 5
    n_miss = np.sum(np.isnan(x), axis=0)

    if func == "cov":
        fun = np.cov
    else:
        fun = np.corrcoef

    # Nothing missing, just calculate it.
    if np.sum(n_miss) == 0:
        return fun(x, rowvar=False)

    idx_missing = [set(np.where(np.isnan(col))[0]) for col in x.T] # Missing indices for each column

    if skip_miss:
        # Skipping Missing: Get all the rows which have values and calculate the covariance.
        rows = set(range(n)) 
        for c in range(m):
            for rm in idx_missing[c]:
                rows.discard(rm)
        rows = sorted(list(rows))
        return fun(x[rows, :], rowvar=False)
    else:
        # Pairwise: For each cell, calculate the covariance.
        out = np.empty((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i+1):
                rows = set(range(n))
                for c in (i, j):
                    for rm in idx_missing[c]:
                        rows.discard(rm)
                rows = sorted(list(rows))
                
                out[i,j] = fun(x[rows][:, [i, j]], rowvar=False)[0,1]
                if i != j:
                    out[j, i] = out[i, j]

        return out


# EW-Covariance

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

def near_psdCor(A, epsilon=0):
    n = A.shape[0]
    invSD = None
    out = A.copy()
    if np.sum(np.isclose(np.diag(out), 1.0)) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1 / (vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out


def near_psdCov(cov):
    sd, cor = CovtoCor(cov)
    near_psd_cor = near_psdCor(cor)
    near_psd_cov = CortoCov(sd, pd.DataFrame(near_psd_cor))
    return np.array(near_psd_cov)


# Higham's 2002 nearest psd correlation function,自己的：

def higham_nearestPSDcor(pc,epsilon=1e-9,maxIter=100,tol=1e-9):
    n = pc.shape[0]
    W = np.diag(np.ones(n))

    deltaS = 0

    Yk = pc.copy()
    norml = np.finfo(np.float64).max
    i=1

    while i <= maxIter:
        Rk = Yk - deltaS
        Xk = getps(Rk,W)
        deltaS = Xk - Rk
        Yk = getpu(Xk)
        norm = wgtNorm(Yk-pc,W)
        print(f"Norm: {norm}")
        minEigVal = np.min(np.linalg.eigvals(Yk))
        

        if norm - norml < tol and minEigVal > -epsilon:
            break
        norml = norm
        i += 1
    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print(f"Convergence failed after {i-1} iterations")
    return Yk

def wgtNorm(A,W):
    W05 = np.sqrt(W)
    W05 = W05 @ A @ W05
    return np.sum(W05 * W05).sum()

def Aplus(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.diag(np.maximum(vals,0))
    return vecs @ vals @ vecs.T

# The first projection for Higham method which assume that the weight matrix is diagonal
def getpu(x,w=None):
    Aret = x.copy()
    for i in range(Aret.shape[0]):
        Aret[i,i] = 1.0
    return Aret

# The second projection for Higham method
def getps(x,w=None):
    W05 = np.sqrt(w)
    iW = np.linalg.inv(W05)
    return iW @ Aplus(W05 @ x @ W05) @ iW

def higham_nearestPSDcov(cov):
    sd, cor = CovtoCor(cov)
    near_psd_cor = higham_nearestPSDcor(cor)
    near_psd_cov = CortoCov(sd, pd.DataFrame(near_psd_cor))
    return np.array(near_psd_cov)

def CovtoCor(cov):
    cov = pd.DataFrame(cov)
    var = np.diag(cov)
    var = var.astype('float64') 
    sd = np.sqrt(var)
    cor =  pd.DataFrame(np.dot(np.dot(np.diag(1 / sd), cov), np.diag(1 / sd)), 
                        columns = cov.columns, index = cov.columns)
    return sd, cor

def CortoCov(sd, cor):
    cov =  pd.DataFrame(np.dot(np.dot(np.diag(sd), cor), np.diag(sd)), 
                        columns = cor.columns, index = cor.columns)
    return cov



# Simulation Methods
# initial
def simulate_normal(N, cov, mean=None, seed=1234, fixmethod=chol_psd):
    n, m = cov.shape

    # Error Checking
    if n != m:
        raise ValueError(f"Covariance Matrix is not square ({n},{m})")

    out = np.empty((n, N))

    # If the mean is missing then set to 0, otherwise use provided mean
    _mean = np.zeros(n)
    if mean is not None:
        if n != len(mean):
            raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")
        _mean = mean.copy()

    # Take the root
    try:
        l = fixmethod(cov)
    except np.linalg.LinAlgError as e:
        if "Matrix is not positive definite" in str(e):
            # Matrix is not PD, assuming PSD and continuing.
            l = fixmethod(cov)
        else:
            raise e

    # Generate needed random standard normals
    rng = default_rng(seed)
    d = rng.normal(0.0, 1.0, size=(n, N))

    # Apply the standard normals to the Cholesky root
    out = np.dot(l, d)

    # Loop over iterations and add the mean
    for i in range(n):
        out[i, :] += _mean[i]

    return out


def simulateNormal(cov, times = 100000, seed = 1234, fixmethod=chol_psd):
    cov = fixmethod(cov)
    sim_df = pd.DataFrame(np.random.multivariate_normal(np.zeros(cov.shape[1]), cov, times), 
                          columns = (pd.DataFrame(cov)).columns)
    sim_df = np.array(sim_df.T)
    return sim_df


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
    if len(means) != 0:
        if len(means) != out.shape[1]:
            raise Exception("Mean size does not match with cov")
        for i in range(m):
            out[:,i] = out[:,i] + means[i]
    return out


def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    # If the mean is missing then set to 0, otherwise use the provided mean
    _mean = np.zeros(n)
    if mean is not None:
        _mean = mean.copy()

    # Eigenvalue decomposition
    vals, vecs = eigh(a)
    vals = np.real(vals)
    vecs = np.real(vecs)
    
    # Julia returns values lowest to highest, flip them and the vectors
    flip = np.arange(len(vals) - 1, -1, -1)
    vals = vals[flip]
    vecs = vecs[:, flip]
    
    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    
    if pctExp < 1:
        nval = 0
        pct = 0.0

        # Figure out how many factors we need for the requested percent explained
        for i in range(len(posv)):
            pct += vals[i] / tv
            nval += 1
            if pct >= pctExp:
                break

        if nval < len(posv):
            posv = posv[:nval]

    vals = vals[posv]
    vecs = vecs[:, posv]

    # Print information if needed
    # print(f"Simulating with {len(posv)} PC Factors: {np.sum(vals)/tv*100}% total variance explained")

    B = vecs @ np.diag(np.sqrt(vals))

    np.random.seed(seed)
    m = len(vals)
    r = np.random.randn(m, nsim)

    out = np.transpose(B @ r)

    # Loop over iterations and add the mean
    for i in range(n):
        out[:, i] += _mean[i]

    return out







def return_calculate(df,method="DISCRETE"):
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



def fit_normal_distribution(data):
    '''
    Args:
    data: dataframe
    Returns:
    mu: double
    sigma: double
    '''
    mu, sigma = norm.fit(data)
    return mu, sigma

def fit_t_distribution(data):
    df, mu, sigma = t.fit(data)
    return mu, sigma, df

def VaR_normal(data, alpha=0.05):
    mu, sigma = norm.fit(data)
    return norm.ppf(alpha, mu, sigma)







def VaR_t(data, alpha=0.05):
    df, mu, sigma = t.fit(data)
    return t.ppf(alpha, df, mu, sigma)

'''
def VaR(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    return -v
'''
def VaR_simulation(data, alpha=0.05, method="T", seed=1234):
    if method.upper() == "T":
        df, mu, sigma = t.fit(data)
        sim = t.rvs(df, mu, sigma, size=100000, random_state=seed)
        return VaR(sim, alpha)
    elif method.upper() == "NORMAL":
        mu, sigma = norm.fit(data)
        sim = norm.rvs(mu, sigma, size=100000, random_state=seed)
        return VaR(sim, alpha)
    else:
        raise Exception("Method must be either T or Normal")


# ES calculation
'''
def ES(a,alpha=0.05):
    a.sort()
    v= np.quantile(a,alpha)
    es = a[a<=v].mean()
    return -es
'''
def ES_normal(data, alpha=0.05):
    mu, sigma = norm.fit(data)
    es = mu - (sigma / alpha) * norm.pdf(norm.ppf(alpha))
    return es
    

def ES_t(data, alpha=0.05):
    df, mu, sigma = t.fit(data)
    es = mu - (sigma / (1 - df) * t.pdf(t.ppf(alpha, df), df))
    return es


def ES_simulation(data, alpha=0.05, method="T", seed=1234):
    if method.upper() == "T":
        df, mu, sigma = t.fit(data)
        sim = t.rvs(df, mu, sigma, size=100000, random_state=seed)
        return ES(sim, alpha)
    elif method.upper() == "NORMAL":
        mu, sigma = norm.fit(data)
        sim = norm.rvs(mu, sigma, size=100000, random_state=seed)
        return ES(sim, alpha)
    else:
        raise Exception("Method must be either T or Normal")
    

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
    sigma= ew_cov(r_h, 0.94)
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
    assets_cov = ew_cov(returns, lambda_)
    p_sig = np.sqrt(np.transpose(delta) @ assets_cov @ delta)
    var_delta = -current_price * norm.ppf(alpha) * p_sig
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



def VaR_hist(ror, alpha, seed = 2, n = 10000):
    np.random.seed(seed)
    ror = np.random.choice(ror.iloc[:, 0], size=n)    
    VaR = -np.quantile(ror, alpha)
    diff = VaR + np.mean(ror)
    return pd.DataFrame({"VaR Absolute": [VaR], 
                         "VaR Diff from Mean": [diff]})



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

def ES_hist(ror, alpha, seed = 2, n = 10000):
    VaR = VaR_hist(ror, alpha, seed, n)
    np.random.seed(seed)
    ror = np.random.choice(ror.iloc[:, 0], size=n)
    VaR = -VaR.loc[0, "VaR Absolute"]
    ES = -np.mean(ror[ror <= VaR])
    diff = ES + np.mean(ror)
    return pd.DataFrame({"ES Absolute": [ES], 
                         "ES Diff from Mean": [diff]})


class FittedModel:
    def __init__(self, params, beta, error_model, eval_func, errors, u):
        self.params = params
        self.beta = beta
        self.error_model = error_model
        self.eval = eval_func
        self.errors = errors
        self.u = u

def general_t_ll(mu, s, nu, x):
    td = t(df=nu, loc=mu, scale=s)
    return np.sum(np.log(td.pdf(x)))




def fit_general_t(x):
    nu, m, s = t.fit(x)
    error_model = t(df=nu, loc=m, scale=s)
    errors = x - m
    u = error_model.cdf(x)

    Params = [m, s, nu]

    def eval_model(u):
        return error_model.ppf(u)

    return FittedModel(Params, None, error_model, eval_model, errors, u)

def fit_normal(x):
    m, s = np.mean(x), np.std(x, ddof=1)
    error_model = norm(loc=m, scale=s)
    errors = x - m
    u = error_model.cdf(x)

    Params = [m, s]

    def eval_model(u):
        return error_model.ppf(u)

    return FittedModel(Params, None, error_model, eval_model, errors, u)

def fit_normal_ewcov(x, lamb=0.94):
    m = np.mean(x)
    ew = ew_cov(x, lamb)
    s = np.sqrt(ew)
    error_model = norm(loc=m, scale=s)
    errors = x - m
    u = error_model.cdf(x)

    Params = [m, s]

    def eval_model(u):
        return error_model.ppf(u)

    return FittedModel(Params, None, error_model, eval_model, errors, u)




def MLE_t(X, Y):
    X = sm.add_constant(X)
    def ll_t(params):
        df = params[0]
        s = params[1]
        b = params[2:]
        e = Y - np.dot(X, b)
        ll = np.sum(stats.t.logpdf(e, df=df, loc=0, scale=s))
        return -ll
    beta = np.zeros(X.shape[1])
    s = np.std(Y - np.dot(X, beta))
    df = 1
    params = [df, s]
    for i in beta:
        params.append(i)
    bnds = ((1e-9, None), (1e-9, None), (None, None), (None, None), (None, None), (None, None))
    res = optimize.minimize(ll_t, params, bounds=bnds, options={"disp": True})
    beta_mle = res.x[2:]
    return beta_mle


def fit_regression_t(data):
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    betas = MLE_t(X, Y)
    X = sm.add_constant(X)
    e = Y - np.dot(X, betas)
    f = Fitter(e, distributions = ['t'])
    f.fit()
    params = f.fitted_param['t']
    out = {"mu": [params[1]], 
           "sigma": [params[2]], 
           "nu": [params[0]]}
    out["Alpha"] = betas[0]
    for i in range(1, len(betas)):
        out["B" + str(i)] = betas[i]
    return pd.DataFrame(out)


def OLS(X, Y):
    n = X.shape[0]
    _X = np.column_stack([np.ones(n), X])
    B = np.linalg.inv(_X.T @ _X) @ _X.T @ Y
    e = Y - _X @ B
    return B, e


def copula_risk(portfolio, ror, nSim = 100000):
    portfolio["currentValue"] = portfolio["Holding"]  * portfolio["Starting Price"]
    portfolio.index = list(portfolio["Stock"])
    portfolio = portfolio.drop(columns = ['Stock'])
    models = {}
    mu = {}
    sigma = {}
    nu = {}
    U = pd.DataFrame()
    for stock in portfolio.index:
        if portfolio.loc[stock, "Distribution"] == "Normal":
            models[stock] = fit_normal(ror[stock])
            mu[stock] = models[stock].params[0]
            sigma[stock] = models[stock].params[1]
        elif portfolio.loc[stock, "Distribution"] == "T":
            models[stock] = fit_general_t(ror[stock])
            mu[stock] = models[stock].params[0]
            sigma[stock] = models[stock].params[1]
            nu[stock] = models[stock].params[2]
        U[stock] = (ror[stock] - mu[stock]) / sigma[stock]
    spcor = U.corr(method='spearman')
    uSim = simulate_pca(spcor, nSim)
    uSim = stats.norm.cdf(uSim)
    uSim = pd.DataFrame(uSim, columns = portfolio.index)
    simRet = pd.DataFrame()
    for stock in uSim.columns:
        if portfolio.loc[stock, "Distribution"] == "Normal":
            simRet[stock] = stats.norm.ppf(uSim[stock], loc = mu[stock], scale = sigma[stock])
        elif portfolio.loc[stock, "Distribution"] == "T":
            simRet[stock] = stats.t.ppf(uSim[stock], df = nu[stock], loc = mu[stock], scale = sigma[stock])
    # simulatedValue = portfolio["currentValue"] * (1 + simRet)
    pnl = portfolio["currentValue"] * simRet
    pnl["Total"] = 0
    risk = pd.DataFrame(columns = ["Stock", "VaR95", "ES95", "VaR95_Pct", "ES95_Pct"])
    for stock in pnl.columns[:-1]:
        pnl["Total"] += pnl[stock]
        i = risk.shape[0]
        risk.loc[i, "Stock"] = stock
        risk.loc[i, "VaR95"] = -np.percentile(pnl[stock], 5)
        risk.loc[i, "VaR95_Pct"] = risk.loc[i, "VaR95"] / portfolio.loc[stock, "currentValue"]
        ub = -risk.loc[i, "VaR95"]
        
        risk.loc[i, "ES95"] = -np.mean(pnl[pnl[stock] <= ub][stock])
        risk.loc[i, "ES95_Pct"] = risk.loc[i, "ES95"] / portfolio.loc[stock, "currentValue"]
    total_value = sum(portfolio["currentValue"])

    total_VaR = -np.percentile(pnl["Total"], 5)
    row_total = risk.shape[0]
    risk.loc[row_total, "Stock"] = "Total"
    risk.loc[row_total, "VaR95"] = total_VaR
    risk.loc[row_total, "VaR95_Pct"] = total_VaR / total_value
    ub = -risk.loc[row_total, "VaR95"]
    risk.loc[row_total, "ES95"] = -np.mean(pnl[pnl["Total"] <= ub]["Total"])
    risk.loc[row_total, "ES95_Pct"] = risk.loc[row_total, "ES95"] / total_value
    return risk