import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import t
import matplotlib.pyplot as plt



# Problem 2
# Assume the multiple linear regression model
df = pd.read_csv('problem2.csv')
print(df.head())
print(df.describe())
print(df.info())

x = df['x']
y = df['y']

# a.
# OLS
x = sm.add_constant(x)
modelOLS = sm.OLS(y, x)
resultsOLS = modelOLS.fit()
print(resultsOLS.summary())
error = resultsOLS.resid
sigmaOLS = np.std(error)
print("beta_OLS:", resultsOLS.params)
print("sigma_OLS:" , sigmaOLS)


# MLE: given the assumption of normality
def myll(params, x, y):
    beta, sigma = params[:-1], params[-1]
    e = y - np.dot(x, beta)
    s2 = sigma ** 2
    n = len(y)
    ll = -n / 2 * np.log(s2 * 2 * np.pi) - sum(e ** 2) / (2 * s2)
    return -ll

initial_params = np.concatenate((np.zeros(x.shape[1]), [1.0]))
results = minimize(myll, initial_params, args=(x, y), method='L-BFGS-B')

beta_hat, s_hat = results.x[:-1], results.x[-1]
print("beta_MLE", beta_hat)
print("sigma_MLE", s_hat)




# b.
# MLE: given the assumption of a T distribution of errors


def myll_t(params, x, y):
    beta, df, sigma = params[:-2], params[-2], params[-1]
    e = y - np.dot(x, beta)

    pdf_values = t.pdf(e, df)  
    ll = np.sum(np.log(pdf_values))
    
    return -ll

# initial_params_t = np.concatenate((np.zeros(x.shape[1]), [3.0, 1.0]))
initial_params_t = [0,0,1,1]

results_t = minimize(myll_t, initial_params_t, args=(x, y), method='L-BFGS-B')

beta_hat_t, df_hat, s_hat_t = results_t.x[:-2], results_t.x[-2], results_t.x[-1]
print("beta_MLE_t:", beta_hat_t)
print("sigma_MLE_t:", s_hat_t)


# Using R2 and adj-R2 to compare two MLE assumptions
def calc_r2(beta, X, Y):
    rr = Y - np.dot(X, beta)
    SSR = np.dot(rr, rr)
    tt = Y - np.mean(Y)
    SST = np.dot(tt, tt)
    R2 = 1 - SSR / SST

    n = len(Y)
    p = 1
    aR2 = 1 - ((1 - R2)*(n-1)/(n-p-1))

    return R2, aR2


print("R2(Normal):", calc_r2(beta_hat, x, y)[0])
print("Adj-R2(Normal):", calc_r2(beta_hat, x, y)[1])
print("R2(T-distribution):", calc_r2(beta_hat_t, x, y)[0])
print("Adj-R2(T-distribution):", calc_r2(beta_hat_t, x, y)[1])



# Using AIC and BIC to compare two MLE assumptions
n = len(y)
log_likelihood_normal = -results.fun  
aic_normal = 2 * (len(initial_params) - 1) - 2 * log_likelihood_normal
bic_normal = np.log(n) * (len(initial_params) - 1) - 2 * log_likelihood_normal

log_likelihood_t = -results_t.fun  
aic_t = 2 * (len(initial_params_t) - 1) - 2 * log_likelihood_t
bic_t = np.log(n) * (len(initial_params_t) - 1) - 2 * log_likelihood_t

print("AIC (Normal):", aic_normal)
print("BIC (Normal):", bic_normal)

print("AIC (T-distribution):", aic_t)
print("BIC (T-distribution):", bic_t)




# c.
dfx = pd.read_csv("problem2_x.csv")
x_x1 = dfx['x1']
x_x2 = dfx['x2']
# Fit the data using MLE given X=[X1,X2] follows the multivariate normal distribution
X = np.column_stack((x_x1, x_x2))

dfx1 = pd.read_csv("problem2_x1.csv")
X1_obs = dfx1['x1']

mu = np.mean(X, axis=0)
cov = np.cov(X, rowvar=False)
mvn = multivariate_normal(mean=mu, cov=cov)

conditional_means = []
confidence_intervals = []

for x1_observed in X1_obs:
    conditional_mean = mu[1] + cov[1, 0] / cov[0, 0] * (x1_observed - mu[0])
    conditional_variance = cov[1, 1] - cov[1, 0]**2 / cov[0, 0]

    ci_95 = stats.norm.interval(0.95, loc=conditional_mean, scale=np.sqrt(conditional_variance))

    conditional_means.append(conditional_mean)
    confidence_intervals.append(ci_95)

# Plot the results
plot_data = pd.DataFrame({
    'X1_observed': X1_obs,
    'Expected_X2': conditional_means,
    'Lower_CI': [ci[0] for ci in confidence_intervals],
    'Upper_CI': [ci[1] for ci in confidence_intervals]
})

plt.figure(figsize=(10, 6))
plt.scatter(plot_data['X1_observed'], plot_data['Expected_X2'], color='blue', label='Expected X2')
plt.fill_between(plot_data['X1_observed'], plot_data['Lower_CI'], plot_data['Upper_CI'], color='gray', alpha=0.5, label='95% Confidence Interval')
plt.xlabel('Observed X1')
plt.ylabel('Expected X2')
plt.title('Expected X2 Given X1 with 95% Confidence Interval')
plt.legend()
plt.grid(True)
plt.savefig('expected_x2_plot.png')

plt.show()

# Problem 3
# Fit the data in problem3.csv using AR(1) through AR(3) and MA(1) through MA(3), respectively
df = pd.read_csv('problem3.csv')

# Using ARMA package

y = df['x']
# Function to fit AR and MA models
def fit_model(order):
    model = sm.tsa.ARMA(y, order)
    result = model.fit()
    return result

# Fit AR(1) through AR(3)
ar_orders = range(1, 4)
ar_results = {f'AR({p})': fit_model(order=(p, 0)) for p in ar_orders}

# Fit MA(1) through MA(3)
ma_orders = range(1, 4)
ma_results = {f'MA({q})': fit_model(order=(0, q)) for q in ma_orders}

# Compare model fits using AIC
models = {**ar_results, **ma_results}

best_model = min(models, key=lambda k: models[k].aic)

# Print AIC values for comparison
for model, result in models.items():
    print(f'{model}: AIC = {result.aic}')

# Plot the data and the best-fit model
plt.plot(y, label='Data')
plt.plot(models[best_model].fittedvalues, label=f'Best Fit ({best_model})')
plt.legend()
plt.show()
