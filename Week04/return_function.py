import numpy as np
import pandas as pd

def return_calculate(prices, method = "DISCRETE", date_column = "Date"):
    '''
    Calculate returns from prices, using the method specified.
    Parameters
    ----------
    prices : DataFrame
        A DataFrame containing prices of assets.
    method : str, optional
        The method to use to calculate returns. The default is "DISCRETE". You can also use "LOG" or "BROWNIAN".
    date_column : str, optional
        The name of the column containing dates. The default is "Date".
    Returns
    -------
    out : DataFrame
        A DataFrame containing returns.
    '''
    vars = list(prices.columns)
    n_vars = len(vars)
    vars = [var for var in vars if var != date_column]

    if n_vars == len(vars):
        raise ValueError("Date column not found")
    
    n_vars -= 1

    p = prices[vars].values
    n, m = p.shape
    rt = np.empty((n-1, m))

    for i in range(n-1):
        for j in range(m):
            rt[i,j] = p[i+1,j] / p[i,j]
            r_brown = p[i+1,j] - p[i,j]
    
    if method.upper() == "DISCRETE":
        rt -= 1.0
    elif method.upper() == "LOG":
        rt = np.log(rt)
    elif method.upper() == "BROWNIAN":
        rt = r_brown        
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\",\"BROWNIAN\")")

    dates = prices.iloc[1:n][date_column]
    out = pd.DataFrame({date_column: dates})

    # Create a list to store DataFrames to concatenate
    dfs_to_concat = [pd.DataFrame({vars[i]: rt[:, i]}) for i in range(n_vars)]

    # Concatenate all DataFrames at once
    out = pd.concat([out] + dfs_to_concat, axis=1)
    
    return out
