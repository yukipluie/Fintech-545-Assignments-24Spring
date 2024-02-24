import pandas as pd
import numpy as np

# Problem 1

DailyReturn = pd.read_csv('DailyReturn.csv')
DailyReturn = DailyReturn.drop(columns=DailyReturn.columns[0])

print(DailyReturn.describe())
print(type(DailyReturn))

# Create a routine for calculating an exponentially weighted covarience matrix
weights = pd.DataFrame()
cumulative_weights = pd.DataFrame()
n = len(DailyReturn)
x = np.empty(n)
w = np.empty(n)
cumulative_w = np.empty(n)

def populate_weights(x, w, cw, λ):
    n = len(x)
    tw = 0
    for i in range(n):
        x[i] = i + 1
        w[i] = (1-λ) * λ**i
        tw += w[i]
        cw[i] = tw

    for i in range(n):
        w[i] = w[i] / tw
        cw[i] = cw[i] / tw

# Calculate weights for λ=0.75
populate_weights(x, w, cumulative_w, 0.75)
weights['x'] = x.copy()
weights['λ=0.75'] = w.copy()
cumulative_weights['x'] = x.copy()
cumulative_weights['λ=0.75'] = cumulative_w.copy()

# Repeat for other λ values (0.90, 0.97, 0.99)
populate_weights(x, w, cumulative_w, 0.90)
weights['λ=0.90'] = w.copy()
cumulative_weights['λ=0.90'] = cumulative_w.copy()

populate_weights(x, w, cumulative_w, 0.97)
weights['λ=0.97'] = w.copy()
cumulative_weights['λ=0.97'] = cumulative_w.copy()

populate_weights(x, w, cumulative_w, 0.99)
weights['λ=0.99'] = w.copy()
cumulative_weights['λ=0.99'] = cumulative_w.copy()

# Print or further use the weights and cumulative_weights DataFrames
print(weights)
print(cumulative_weights)
col_075 = weights['λ=0.75'].values
#print(col_075)


def calculate_ew_covarience_matrix(X, weights):
    obs = len(X)
    assets = len(X.columns)
    ew_cov = np.zeros((assets, assets))
    for i in range(assets):
        for j in range(assets):
            mean_i = np.average(X.iloc[:, i])
            mean_j = np.average(X.iloc[:, j])
            for t in range(obs):
                ew_cov[i, j] += weights[t] * (X.iloc[t, i] - mean_i) * (X.iloc[t, j] - mean_j)
    return ew_cov
    
    



# ？Use package to calculate exponentially weighted covarience matrix


