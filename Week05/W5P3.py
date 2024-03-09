import sys
sys.path.append('../RiskManagement')
import risk_management as rm
import numpy as np
import pandas as pd


DailyPrices = pd.read_csv('DailyPrices.csv')
Portfolio = pd.read_csv('Portfolio.csv')


# Calculate arithmetic returns
returns = rm.return_calculate(DailyPrices, method = "DISCRETE")

rowDailyPrices =DailyPrices.shape[0] - 1
for i in range(Portfolio.shape[0]):
    stock = Portfolio.loc[i, "Stock"]
    Portfolio.loc[i, "Starting Price"] =DailyPrices.loc[rowDailyPrices, stock]
    if Portfolio.loc[i, "Portfolio"] == "A" or Portfolio.loc[i, "Portfolio"] == "B":
        Portfolio.loc[i, "Distribution"] = "T"
    elif Portfolio.loc[i, "Portfolio"] == "C":
        Portfolio.loc[i, "Distribution"] = "Normal"
ror = rm.return_calculate(DailyPrices, method = "DISCRETE")
ror = ror.drop(columns=['Date'])

print(ror)
print(Portfolio)

portfolio_a = Portfolio.loc[Portfolio["Portfolio"] == "A"]
risk_a = rm.copula_risk(portfolio_a, ror)
print("Portfolio A Risk:")
print(risk_a)
portfolio_b = Portfolio.loc[Portfolio["Portfolio"] == "B"]
risk_b = rm.copula_risk(portfolio_b, ror)
print("Portfolio B Risk:")
print(risk_b)

portfolio_c = Portfolio.loc[Portfolio["Portfolio"] == "C"]
risk_c = rm.copula_risk(portfolio_c, ror)
print("Portfolio C Risk:")
print(risk_c)

risk = rm.copula_risk(Portfolio, ror)
print("Total Portfolio Risk:")
print(risk)
