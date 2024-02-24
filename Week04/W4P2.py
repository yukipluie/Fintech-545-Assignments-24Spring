import pandas as pd
import numpy as np
from scipy.stats import norm, t
import statsmodels.api as sm
from return_function import return_calculate

DailyPrices = pd.read_csv('DailyPrices.csv')


print(DailyPrices.describe())
print(type(DailyPrices))


# Calculate the arithmetic returns for all prices.
returns = return_calculate(DailyPrices, method = "DISCRETE", date_column = "Date")
#print(type(returns))
print(returns.describe())

# Calculate VaR of META
meta_prices = DailyPrices['META']
meta_current_price = meta_prices.iloc[-1]
print("current: ", meta_current_price)
meta_returns = returns['META']

# Remove the mean from the series
meta_returns -= np.mean(meta_returns)
# print("after removing: ", meta_returns.describe())
alpha = 0.05

print(meta_returns.isnull().sum())
print(np.isfinite(meta_returns).all())
meta_returns = meta_returns.dropna()
print("dropped: ", meta_returns.describe())

print(np.std(meta_returns))

# 1. Normal Distribution
VaR_normal = - norm.ppf(alpha, np.mean(meta_returns), np.std(meta_returns)) * meta_current_price
print("VaR_normal: ", VaR_normal)

# 2.Using a normal distribution with an Exponentially Weighted variance (λ = 0. 94)
λ = 0.94
weighted_var = meta_returns.ewm(alpha = 1 - λ).var().iloc[-1]
#print("weighted_var: ", weighted_var)

VaR_ew = - norm.ppf(alpha, np.mean(meta_returns), np.sqrt(weighted_var)) * meta_current_price
print("VaR_ew: ", VaR_ew)

# 3. Using a MLE fitted T distribution.

params = t.fit(meta_returns)
print("params: ", params)
VaR_t = - t.ppf(alpha, *params) * meta_current_price
print("VaR_t: ", VaR_t)

# 4. Using a fitted AR(1) model.
order = (1, 0, 0)
model = sm.tsa.ARIMA(meta_returns, order = order)
results = model.fit()
std_resid = results.resid.std()
VaR_ar1 = - norm.ppf(alpha, np.mean(meta_returns), std_resid) * meta_current_price
print("VaR_ar1: ", VaR_ar1)

# 5. Using a Historic Simulation.
VaR_hist = - np.percentile(meta_returns, alpha * 100) * meta_current_price
print("VaR_hist: ", VaR_hist)







