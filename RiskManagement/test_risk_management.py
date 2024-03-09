import unittest
import numpy as np
import pandas as pd
import risk_management as rm
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.stats import norm, t
from scipy.stats import spearmanr

relative_tol = 1e-5
absolute_tol = 1e-8

relative_tol2 = 1e-1
absolute_tol2 = 1e-3

def are_approximately_equal(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance


# Test 1 - missing covariance calculations
# 1.1 Covariance Missing data, skip missing rows
x1 = pd.read_csv("data/test1.csv")
result1_1 = rm.missing_cov(x1.values, skip_miss=True, func="cov")
to1_1 = pd.read_csv("data/testout_1.1.csv")

are_approx_equal = np.allclose(result1_1, to1_1.values, rtol=relative_tol, atol=absolute_tol)
print(f"1.1Arrays are approximately equal: {are_approx_equal}")

# 1.2 Correlation Missing data, skip missing rows
result1_2 = rm.missing_cov(x1.values, skip_miss=True, func="corr")
to1_2 = pd.read_csv("data/testout_1.2.csv")
are_approx_equal = np.allclose(result1_2, to1_2.values, rtol=relative_tol, atol=absolute_tol)
print(f"1.2Arrays are approximately equal: {are_approx_equal}")

# 1.3 Covariance Missing data, pairwise
result1_3 = rm.missing_cov(x1.values, skip_miss=False, func="cov")
to1_3 = pd.read_csv("data/testout_1.3.csv")
are_approx_equal = np.allclose(result1_3, to1_3.values, rtol=relative_tol, atol=absolute_tol)
print(f"1.3Arrays are approximately equal: {are_approx_equal}")

# 1.4 Correlation Missing data, pairwise
result1_4 = rm.missing_cov(x1.values, skip_miss=False, func="corr")
to1_4 = pd.read_csv("data/testout_1.4.csv")
are_approx_equal = np.allclose(result1_4, to1_4.values, rtol=relative_tol, atol=absolute_tol)
print(f"1.4Arrays are approximately equal: {are_approx_equal}")


# 2.1 EW Covariance, lambda=0.97
x = pd.read_csv("data/test2.csv")
result2_1 = rm.ew_cov(x, lam=0.97)
to2_1 = pd.read_csv("data/testout_2.1.csv")
are_approx_equal = np.allclose(result2_1, to2_1.values, rtol=relative_tol, atol=absolute_tol)
print(f"2.1Arrays are approximately equal: {are_approx_equal}")

# 2.2 EW Correlation, lambda=0.94
to2_2 = pd.read_csv("data/testout_2.2.csv")
result2_2 = rm.ew_cov(x, lam=0.94, corr=True)
are_approx_equal = np.allclose(result2_2, to2_2.values, rtol=relative_tol, atol=absolute_tol)
print(f"2.2Arrays are approximately equal: {are_approx_equal}")

# 2.3 Covariance with EW Variance (l=0.94), EW Correlation (l=0.97)
to2_3 = pd.read_csv("data/testout_2.3.csv")
cout = rm.ew_cov(x, lam=0.97)
sd1 = np.sqrt(np.diag(cout))
cout = rm.ew_cov(x, lam=0.94)
sd = 1 / np.sqrt(np.diag(cout))
cout = np.diag(sd1) @ np.diag(sd) @ cout @ np.diag(sd) @ np.diag(sd1)
are_approx_equal = np.allclose(cout, to2_3.values, rtol=relative_tol, atol=absolute_tol)
print(f"2.3Arrays are approximately equal: {are_approx_equal}")

# 3.1 near_psd covariance
to3_1 = pd.read_csv("data/testout_3.1.csv")
x = pd.read_csv("data/testout_1.3.csv")
result3_1 = rm.near_psdCov(x.values)
#print(result3_1)
are_approx_equal = np.allclose(result3_1, to3_1.values, rtol=relative_tol, atol=absolute_tol)
print(f"3.1Arrays are approximately equal: {are_approx_equal}")

# 3.2 near_psd correlation
to3_2 = pd.read_csv("data/testout_3.2.csv")
x = pd.read_csv("data/testout_1.4.csv")
result3_2 = rm.near_psdCor(x.values)
are_approx_equal = np.allclose(result3_2, to3_2.values, rtol=relative_tol, atol=absolute_tol)
print(f"3.2Arrays are approximately equal: {are_approx_equal}")

# 3.3 Higham covariance ---？？？？这里确实有问题了。
to3_3 = pd.read_csv("data/testout_3.3.csv")
x = pd.read_csv("data/testout_1.3.csv")
result3_3 = rm.higham_nearestPSDcov(x)
#result3_3 = np.array(rm.higham_nearestPSDCov(x))
#result3_3 = rm.higham_nearestPSD2(x)
#result3_3 = rm.higham(x.values)

are_approx_equal = np.allclose(result3_3, to3_3.values, rtol=relative_tol, atol=absolute_tol)
print(f"3.3Arrays are approximately equal: {are_approx_equal}")

# 3.4 Higham correlation
to3_4 = pd.read_csv("data/testout_3.4.csv")
x = pd.read_csv("data/testout_1.4.csv")
result3_4 = rm.higham_nearestPSDcor(x)
are_approx_equal = np.allclose(result3_4, to3_4.values, rtol=relative_tol, atol=absolute_tol)
print(f"3.4Arrays are approximately equal: {are_approx_equal}")

# 4.1 chol_psd
to4_1 = pd.read_csv("data/testout_4.1.csv")
x = pd.read_csv("data/testout_3.1.csv")
result4_1 = rm.chol_psd(x.values)
are_approx_equal = np.allclose(result4_1, to4_1.values, rtol=relative_tol, atol=absolute_tol)
print(f"4.1Arrays are approximately equal: {are_approx_equal}")

# 5.1 Normal Simulation PD Input 0 mean - 100,000 simulations, compare input vs output covariance
to5_1 = pd.read_csv("data/testout_5.1.csv")
x = pd.read_csv("data/test5_1.csv")
result5_1 = rm.simulate_normal(100000, x.values)
result5_1 = np.cov(result5_1)
#print(to5_1.values)
#print(result5_1)
are_approx_equal = np.allclose(result5_1, to5_1.values, rtol=relative_tol2, atol=absolute_tol2)
print(f"5.1Arrays are approximately equal: {are_approx_equal}")


# 5.2 Normal Simulation PSD Input 0 mean - 100,000 simulations, compare input vs output covariance
to5_2 = pd.read_csv("data/testout_5.2.csv")
x = pd.read_csv("data/test5_2.csv")
result5_2 = rm.simulate_normal(100000, x.values, seed=4)
result5_2 = np.cov(result5_2)
#print(to5_2.values)
#print(result5_2)
are_approx_equal = np.allclose(result5_2, to5_2.values, rtol=relative_tol2, atol=absolute_tol2)
print(f"5.2Arrays are approximately equal: {are_approx_equal}")

# 5.3 Normal Simulation nonPSD Input, 0 mean, near_psd fix - 100,000 simulations, compare input vs output covariance
to5_3 = pd.read_csv("data/testout_5.3.csv")
x = pd.read_csv("data/test5_3.csv")
#result5_3 = rm.simulate_normal(100000, x.values, seed=4, fixmethod=rm.near_psdCov)
#result5_3 = np.cov(result5_3)
result5_3 = np.cov(rm.simulateNormal(x, fixmethod=rm.near_psdCov))

are_approx_equal = np.allclose(result5_3, to5_3.values, rtol=relative_tol2, atol=absolute_tol2)
print(f"5.3Arrays are approximately equal: {are_approx_equal}")

# 5.4 Normal Simulation PSD Input, 0 mean, higham fix - 100,000 simulations, compare input vs output covariance
to5_4 = pd.read_csv("data/testout_5.4.csv")
x = pd.read_csv("data/test5_3.csv")
#result5_4 = rm.simulate_normal(100000, x.values, seed=4, fixmethod=rm.higham_nearestPSDcov)
#result5_4 = np.cov(result5_4)
result5_4 = np.cov(rm.simulateNormal(x, fixmethod=rm.higham_nearestPSDcov))

are_approx_equal = np.allclose(result5_4, to5_4.values, rtol=relative_tol2, atol=absolute_tol2)
print(f"5.4Arrays are approximately equal: {are_approx_equal}")

# ok5.5 PCA Simulation, 99% explained, 0 mean - 100,000 simulations compare input vs output covariance
to5_5 = pd.read_csv("data/testout_5.5.csv")
x = pd.read_csv("data/test5_2.csv")
result5_5 = rm.pca_sim(x, 100000, pct=0.99, seed=4)
result5_5 = np.cov(result5_5, rowvar=False)
#print(to5_5.values)
#print(result5_5)
are_approx_equal = np.allclose(result5_5, to5_5.values, rtol=relative_tol2, atol=absolute_tol2)
print(f"5.5Arrays are approximately equal: {are_approx_equal}")

# 6.1 calculate arithmetic returns
to6_1 = pd.read_csv("data/test6_1.csv")
x = pd.read_csv("data/test6.csv")
result6_1 = rm.return_calculate(x, method="DISCRETE")
#print(to6_1.values)
#print(result6_1)

# 6.2 calculate log returns
to6_2 = pd.read_csv("data/test6_2.csv")
x = pd.read_csv("data/test6.csv")
result6_2 = rm.return_calculate(x, method="LOG")
#print(to6_2.values)
#print(result6_2)


# 7.1 fit normal distribution
x = pd.read_csv("data/test7_1.csv")
#print(x.values)
to7_1 = pd.read_csv("data/testout7_1.csv")
#print(to7_1.values)
fd = rm.fit_normal(x.values)
#print(fd.error_model.mean(), fd.error_model.std())

are_approx_equal1 = are_approximately_equal(fd.error_model.mean(), to7_1.values[0][0], tolerance=1e-9)
are_approx_equal2 =are_approximately_equal(fd.error_model.std(), to7_1.values[0][1], tolerance=1e-9)

print("7.1mu is approximately equal:", are_approx_equal1)
print("7.1sigma is approximately equal:", are_approx_equal2)


# 7.2 fit T distribution
x = pd.read_csv("data/test7_2.csv")
result7_2 = rm.fit_t_distribution(x.values)
#print(result7_2)
to7_2 = pd.read_csv("data/testout7_2.csv")
are_approx_equal1 = are_approximately_equal(result7_2[0], to7_2.values[0][0], tolerance=1e-5)
are_approx_equal2 =are_approximately_equal(result7_2[1], to7_2.values[0][1], tolerance=1e-5)
are_approx_equal3 =are_approximately_equal(result7_2[2], to7_2.values[0][2], tolerance=1e-5)
print("7.2mu is approximately equal:", are_approx_equal1)
print("7.2sigma is approximately equal:", are_approx_equal2)
print("7.2nu is approximately equal:", are_approx_equal3)


# 7.3 T Regression
x = pd.read_csv("data/test7_3.csv")
to7_3 = pd.read_csv("data/testout7_3.csv")
result7_3 = rm.fit_regression_t(x)
print(result7_3)
print(to7_3)


# 8.1 VaR from Normal Distribution
to8_1 = pd.read_csv("data/testout8_1.csv")
x = pd.read_csv("data/test7_1.csv")
fd = rm.fit_normal(x.values)
var_absolute = rm.VaR_distribution(fd.error_model, 0.05)
var_diff_from_mean = -norm.ppf(0.05, loc=0, scale=fd.error_model.std())
are_approx_equal = are_approximately_equal(var_absolute, to8_1.values[0][0], tolerance=1e-5)
are_approx_equal2 = are_approximately_equal(var_diff_from_mean, to8_1.values[0][1], tolerance=1e-5)
print(f"8.1VaR_absolute is approximately equal:", are_approx_equal)
print(f"8.1VaR_diff_from_mean is approximately equal:", are_approx_equal2)


# 8.2 VaR from T Distribution
to8_2 = pd.read_csv("data/testout8_2.csv")
x = pd.read_csv("data/test7_2.csv")
fd = rm.fit_general_t(x.values)
var_absolute = rm.VaR_distribution(fd.error_model, 0.05)
var_diff_from_mean = -t.ppf(0.05, df=fd.params[2], loc=0, scale=fd.params[1])
are_approx_equal = are_approximately_equal(var_absolute, to8_2.values[0][0], tolerance=1e-5)
are_approx_equal2 = are_approximately_equal(var_diff_from_mean, to8_2.values[0][1], tolerance=1e-5)
print(f"8.2VaR_absolute is approximately equal:", are_approx_equal)
print(f"8.2VaR_diff_from_mean is approximately equal:", are_approx_equal2)

# 8.3 VaR from Simulation -- compare to 8.2 values
to8_3 = pd.read_csv("data/testout8_3.csv")
x = pd.read_csv("data/test7_2.csv")
fd = rm.fit_general_t(x.values)

sim = fd.eval(np.random.rand(10000))
var_absolute = rm.VaR(sim)  #0.04342516386169576
var_diff_from_mean = rm.VaR(sim - np.mean(sim))  #0.08838482430402855
# 不同seed？

# 8.4 ES From Normal Distribution
to8_4 = pd.read_csv("data/testout8_4.csv")
x = pd.read_csv("data/test7_1.csv")
fd = rm.fit_normal(x.values)
es_absolute = rm.ES_distribution(fd.error_model, 0.05)
es_diff_from_mean = rm.ES_distribution(norm(loc=0, scale=fd.error_model.std()), 0.05)
are_approx_equal = are_approximately_equal(es_absolute, to8_4.values[0][0], tolerance=1e-5)
are_approx_equal2 = are_approximately_equal(es_diff_from_mean, to8_4.values[0][1], tolerance=1e-5)
print(f"8.4ES_absolute is approximately equal:", are_approx_equal)
print(f"8.4ES_diff_from_mean is approximately equal:", are_approx_equal2)


# 8.5 ES From T Distribution
to8_5 = pd.read_csv("data/testout8_5.csv")
x = pd.read_csv("data/test7_2.csv")
fd = rm.fit_general_t(x.values)
es_absolute = rm.ES_distribution(fd.error_model, 0.05)
es_diff_from_mean = rm.ES_distribution(t(df=fd.params[2], loc=0, scale=fd.params[1]), 0.05)
are_approx_equal = are_approximately_equal(es_absolute, to8_5.values[0][0], tolerance=1e-5)
are_approx_equal2 = are_approximately_equal(es_diff_from_mean, to8_5.values[0][1], tolerance=1e-5)
print(f"8.5ES_absolute is approximately equal:", are_approx_equal)
print(f"8.5ES_diff_from_mean is approximately equal:", are_approx_equal2)


# 8.6 ES From Simulation -- compare to 8.5 values
to8_6 = pd.read_csv("data/testout8_6.csv")    #0.076906	0.122426
x = pd.read_csv("data/test7_2.csv")
fd = rm.fit_general_t(x.values)
sim = fd.eval(np.random.rand(10000))
es_absolute = rm.ES(sim, 0.05)  #0.07625237010849682
es_diff_from_mean = rm.ES(sim - np.mean(sim), 0.05)  #0.12121203055082962



# 9.1 VaR/ES on 2 levels from simulated values - Copula
returns = pd.read_csv("data/test9_1_returns.csv")
portfolio = pd.read_csv("data/test9_1_portfolio.csv")

risk = rm.copula_risk(portfolio, returns, 100000)
to9_1 = pd.read_csv("data/testout9_1.csv")
print(risk)
print(to9_1)