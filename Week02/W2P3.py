import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('problem3.csv')
y = df['x']

# Function to fit AR model
def fit_ar_model(p):
    model = sm.tsa.ARIMA(y, order=(p, 0, 0))
    result = model.fit()
    return result

# Function to fit MA model
def fit_ma_model(q):
    model = sm.tsa.ARIMA(y, order=(0, 0, q))
    result = model.fit()
    return result

# Fit AR(1) through AR(3)
ar_orders = [1, 2, 3]
ar_results = {f'AR({p})': fit_ar_model(p) for p in ar_orders}

# Fit MA(1) through MA(3)
ma_orders = [1, 2, 3]
ma_results = {f'MA({q})': fit_ma_model(q) for q in ma_orders}

# Compare model fits using AIC and BIC
models = {**ar_results, **ma_results}

best_aic_model = min(models, key=lambda k: models[k].aic)
best_bic_model = min(models, key=lambda k: models[k].bic)

# Print AIC and BIC values for comparison
for model, result in models.items():
    aic_value = round(result.aic, 4)
    bic_value = round(result.bic, 4)
    print(f'{model}: AIC = {aic_value}, BIC = {bic_value}')

# Plot the data and the best-fit models
plt.close('all')
plt.plot(y, label='Data')
plt.plot(models[best_aic_model].fittedvalues, label=f'Best AIC Fit ({best_aic_model})')
plt.plot(models[best_bic_model].fittedvalues, label=f'Best BIC Fit ({best_bic_model})')
plt.legend()
plt.show()

