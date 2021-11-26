# Dexter Dysthe
# Dr. Johannes
# B9336
# 4 October 2021

import pandas as pd

import numpy as np
from scipy.stats import norm


# --------------------------------------------- Question 3 -------------------------------------------- #

# ----------------------- (a) ----------------------- #

def euro_call(S_0, K, sigma, r, T):
    """
    European call option pricing calculator (without dividends)
    """
    d_1 = 1 / (sigma * np.sqrt(T)) * (np.log(S_0 / K) + (r + (sigma ** 2) / 2) * T)
    d_2 = d_1 - sigma * np.sqrt(T)

    euro_call_price = S_0 * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)

    return euro_call_price


call_price = euro_call(100, 100, 0.25, 0.05, 0.25)
print("Exact price: ", call_price)


# ------------------- (b) and (c) --------------------- #

def euro_call_MC(N, S_0, K, sigma, r, T):
    """
    European call option Monte Carlo pricing calculator (without dividends)
    """
    prices = []
    for index in range(N):
        price = np.exp(-r * T) * max(0, S_0 * np.exp(
            (r - 0.5 * (sigma ** 2)) * T + sigma * np.random.normal(0, np.sqrt(T))) - K)
        prices.append(price)

    monte_carlo_price = np.mean(prices)
    monte_carlo_se = np.std(prices, ddof=1) / np.sqrt(N)

    return [monte_carlo_price, monte_carlo_se]


np.random.seed(37)

price_se_list = [euro_call_MC(N, 100, 100, 0.25, 0.05, 0.25) for N in [10, 10000, 1000000, 10000000]]
price_se_dict = {'N': [10, 10000, 1000000, 10000000],
                 'Monte Carlo Price': [price_se_list[i][0] for i in range(4)],
                 'Absolute Error of Estimate': [np.abs(price_se_list[i][0] - call_price) for i in range(4)],
                 'Standard Error': [price_se_list[i][1] for i in range(4)]}
price_se_df = pd.DataFrame(price_se_dict).set_index('N')

print(price_se_df)
