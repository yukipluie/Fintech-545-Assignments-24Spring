import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Problem 1
df = pd.read_csv('problem1.csv')
x= df.values.flatten().astype(float)

# Calculate the mean, variance, skewness, and kurtosis using formulas
n = len(x)
mean_manual = sum(x) / n
sim_corrected = x - mean_manual
cm2 = sum(sim_corrected**2) / n
variance_manual = sum(sim_corrected**2) / (n - 1)

skewness_manual = sum(sim_corrected**3) / (n * cm2 **(3/2))
kurtosis_manual = sum(sim_corrected**4) / (n * cm2 **2) 
excessKurtosis_manual = kurtosis_manual - 3

print('Manual Calculation:')
print('The mean is: ', mean_manual)
print('The variance is: ', variance_manual)
print('The skewness is: ', skewness_manual)
print('The kurtosis is: ', excessKurtosis_manual)

# Calculate the mean, variance, skewness, and kurtosis using pandas functions
x1 = pd.Series(x)
mean_pd = x1.mean()
variance_pd = x1.var()
skewness_pd = x1.skew()
kurtosis_pd = x1.kurtosis()

print('\nUsing Pandas Functions:')
print('The mean is: ', mean_pd)
print('The variance is: ', variance_pd)
print('The skewness is: ', skewness_pd)
print('The kurtosis is: ', kurtosis_pd)

# Is your statistical package functions biased? 
# Prove or disprove your hypothesis respectively.
# The statistical package functions are biased. 这里再思考一下
# pandas uses the unbiased estimator


sample_size = 100
samples = 1000

d = np.random.normal(0, 1, sample_size)
d1 = pd.Series(d)

means = np.array([d1.mean() for i in range(samples)])
vars = np.array([d1.var() for i in range(samples)])
skews = np.array([d1.skew() for i in range(samples)])
kurts = np.array([d1.kurtosis() for i in range(samples)])

print("mean of means:", means.mean())
print("mean of vars:", vars.mean())
print("mean of skews:", skews.mean())
print("mean of kurts:", kurts.mean())

t_stat_means, p_value_means = stats.ttest_1samp(means, d.mean())
ttest_means = stats.ttest_1samp(means, d.mean())
p2_means = ttest_means.pvalue
print("p-value of means-", p_value_means)
print("Means Match the stats package test?:", np.isclose(p_value_means, p2_means))

t_stat_vars, p_value_vars = stats.ttest_1samp(vars, d.var())
ttest_vars = stats.ttest_1samp(vars, d.var())
p2_vars = ttest_vars.pvalue
print("p-value of vars-", p_value_vars)
print("Vars Match the stats package test?:", np.isclose(p_value_vars, p2_vars))

t_stat_skews, p_value_skews = stats.ttest_1samp(skews, stats.skew(d))
ttest_skews = stats.ttest_1samp(skews, stats.skew(d))
p2_skews = ttest_skews.pvalue
print("p-value of skews-", p_value_skews)
print("Skews Match the stats package test?:", np.isclose(p_value_skews, p2_skews))

t_stat_kurts, p_value_kurts = stats.ttest_1samp(kurts, stats.kurtosis(d))
ttest_kurts = stats.ttest_1samp(kurts, stats.kurtosis(d))
p2_kurts = ttest_kurts.pvalue
print("p-value of kurts-", p_value_kurts)
print("Kurts Match the stats package test?:", np.isclose(p_value_kurts, p2_kurts))


#t = mean(means)/sqrt(var(kurts)/samples)
#p = 2*(1 - cdf(TDist(samples-1),abs(t)))









