import numpy as np

P_tminus1 = 100
mu = 0
sigma = 0.2

n = 1000
r_rand = np.random.normal(mu, sigma, n)

P_tminus1_seq = np.empty(n) + P_tminus1

P_t_seq = np.empty(n)

# Return equation 1
P_t_seq1 = P_tminus1_seq + r_rand
Pt1_mean = np.mean(P_t_seq1)
Pt1_std = np.std(P_t_seq1)
print(Pt1_mean, Pt1_std)

# Return equation 2
P_t_seq2 = P_tminus1_seq * (1 + r_rand)
Pt2_mean = np.mean(P_t_seq2)
Pt2_std = np.std(P_t_seq2)
print(Pt2_mean, Pt2_std)

# Return equation 3
P_t_seq3 = P_tminus1_seq * np.exp(r_rand)
Pt3_mean = np.mean(P_t_seq3)
Pt3_std = np.std(P_t_seq3)
print(Pt3_mean, Pt3_std)
print("calculated mean:", P_tminus1 * np.exp(sigma**2/2))
print("calculated std:", P_tminus1 * np.exp(sigma**2/2) * np.sqrt(np.exp(sigma**2) - 1))