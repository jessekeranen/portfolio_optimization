from pandas_datareader import data as web
import quandl
import pandas as pd
import numpy as np
from datetime import date
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

'''
    
'''
def stocks(assets, names, start, end):
    df = pd.DataFrame()
    for stock in assets:
        df = pd.concat([df, quandl.get(stock, start_date=start, end_date=end, collapse="monthly")["Adj. Close"]], axis=1)
    df.columns = names;
    return df, df.pct_change()


def plotprices(title, my_stocks):
    plt.figure(figsize=(12.2, 4.5))  # width = 12.2in, height = 4.5
    for c in my_stocks.columns.values:
        plt.plot(my_stocks[c], label=c)  # plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)

    plt.title(title)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adj. Price USD ($)', fontsize=18)
    plt.legend(my_stocks.columns.values, loc='upper left')
    plt.show()


def sharperatio(returns, weights, cov, risk_free_rate, target):
    if target == -1:
        return -((returns.dot(weights) - 0.0005)/np.sqrt(portfoliovariance(weights, cov))), weights, [np.sum(weights)-1]
    if target == -2:
        return np.sqrt(portfoliovariance(weights, cov)), weights, [np.sum(weights)-1]
    else:
        return np.sqrt(portfoliovariance(weights, cov)), weights, [np.sum(weights)-1, returns.dot(weights)-target]


def portfoliovariance(weights, cov):
    return np.dot(weights.T, np.dot(cov, weights))


def portfolioreturns(profits, weights):
    return np.sum(profits.dot(weights))


def penalized_function(x, f, returns, cov, risk_free_rate, target, r):
    return f(returns, x, cov, risk_free_rate, target)[0] + r*alpha(returns, x, f, cov, risk_free_rate, target)


def alpha(returns, x, f, cov, risk_free_rate, target):
    (_, ieq, eq) = f(returns, x, cov, risk_free_rate, target)
    return sum([min([0, ieq_j])**2 for ieq_j in ieq]) + sum([eq_k**2 for eq_k in eq])

quandl.ApiConfig.api_key = "bx1qdehfWXg6SNKnicQC"

# This could be changed to ask stocks from terminal. Then also initial weigths need to be calculated every time.
assets = ["WIKI/AAPL", "WIKI/MSFT", "WIKI/TSLA", "WIKI/AMZN", "WIKI/GOOGL"]
names = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
risk_free_rate = 0.0005 # This could be loaded from Quandl

# Dates, also this could be asked from terminal
stockStartDate = date(2010, 6, 1)
today = date(2015, 1, 30)
# today = datetime.today().strftime('%Y-%m-%d')
title = 'Portfolio Adj. Close Price History'
interval = '1mo'
my_stocks, profits = stocks(assets, names, stockStartDate, today)

# plotprices(title, my_stocks)

# Covariance and correlation matrix of the data
covariance_matrix_annual = profits.cov()*12
correlation_matrix_monthly = profits.corr()

# Calculate annualized arithmetic monthly mean returns
returns = np.mean(profits*12)

# Calculate portfolio with highest sharpe ratio
sharpeportfolio = minimize(lambda x: penalized_function(x, sharperatio, returns, covariance_matrix_annual.values, risk_free_rate, -1, 100), weights, method='Nelder-Mead',
         options={'disp': True})

print(sharpeportfolio)
# Calculate expected return and standard deviation of the portfolio with highest sharpe ratio.
sharpe_return = portfolioreturns(profits, sharpeportfolio.x)
sharpe_var = portfoliovariance(sharpeportfolio.x, covariance_matrix_annual)
sharpe_std = np.sqrt(sharpe_var)

smh = sharperatio(returns, sharpeportfolio.x, covariance_matrix_annual, risk_free_rate, -1)

# Next portfolios with random weights are calculated to illustrate efficient frontier.
# Fisrt random 1000 sets of random numbers are generated and then they are normalized.
arr = np.random.rand(1000, 5)
arr = arr/arr.sum(axis=1, keepdims=True)

# Now calculate portfolio with lowest standard deviation
min_var_portf = minimize(lambda x: penalized_function(x, sharperatio, returns, covariance_matrix_annual.values, risk_free_rate, -2, 100), weights, method='Nelder-Mead',
         options={'disp': True})
min_var_portf_ret = portfolioreturns(returns, min_var_portf.x)


# Calculate optimal portfolios for every return from minvar portfolio on and store them in an array.
x = range(100)
array = []
array_weigths = []
for n in x:
    res = minimize(lambda x: penalized_function(x, sharperatio, returns, covariance_matrix_annual.values, risk_free_rate,(min_var_portf_ret + (max(returns)-min_var_portf_ret)*n/100), 100),
                   weights, method='Nelder-Mead',
                   options={'disp': False})
    array.append(res)
    array_weigths.append(res.x)


# Calculate std and return for each optimal portfolio for plotting purposes.
var = []
ret = []
for optm in array:
    var.append(portfoliovariance(optm.x, covariance_matrix_annual))
    ret.append(portfolioreturns(returns, optm.x))


# Area chart showing how weights change with different risk levels.
df = pd.DataFrame(array_weigths, index=np.sqrt(var))
df.columns = names
df[df < 0] = 0
df.plot(kind='area', stacked=True)
plt.ylabel("Allocation")
plt.show()


# Calculate also std and return for each randomly produced portfolio created above.
var2 = []
ret2 = []
for weight in arr:
    var2.append(portfoliovariance(weight, covariance_matrix_annual))
    ret2.append(portfolioreturns(returns, weight))

# Plot the results.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(var2, ret2, c='r', marker="o", label="Random")
ax.plot(var, ret, label="Optimal")
plt.show()