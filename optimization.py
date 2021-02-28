from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def stocks(names, start, end):
    df = pd.DataFrame()
    for stock in assets:
        df[stock] = web.DataReader(stock, data_source='yahoo', start=stockStartDate, end=today)['Adj Close']
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


def plotportfolios(portfolios, profits, sharpe, cov):
    plt.plot(np.sqrt(portfoliovariance(sharpe.x, cov)), portfolioreturns(profits, sharpe.x), marker='o', markerfacecolor='blue')
    for portfolio in portfolios:
        plt.plot(portfolio.fun, portfolioreturns(profits, portfolio.x), 'ro')
    plt.xlabel('Standard deviation of the portfolio', fontsize=18)
    plt.ylabel('Return of the portfolio', fontsize=18)
    plt.show()


def sharperatio(profits, weights, cov, target):
    if target == -1:
        return -portfolioreturns(profits, weights)/np.sqrt(portfoliovariance(weights, cov)), [], [np.sum(weights)-1]
    else:
        return np.sqrt(portfoliovariance(weights, cov)), [], [np.sum(weights)-1, portfolioreturns(profits, weights)-target]


def portfoliovariance(weights, cov):
    return np.dot(weights.T, np.dot(cov, weights))


def portfolioreturns(profits, weights):
    return np.sum(profits.mean()*weights)*252


def penalized_function(x, f, profits, cov, target, r):
    return f(profits, x, cov, target)[0] + r*alpha(profits, x, f, cov, target)


def alpha(profits, x, f, cov, target):
    (_, ieq, eq) = f(profits, x, cov, target)
    return sum([eq_k**2 for eq_k in eq])

assets = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

stockStartDate = '2013-01-01'
today = datetime.today().strftime('%Y-%m-%d')
title = 'Portfolio Adj. Close Price History'
my_stocks, profits = stocks(assets, stockStartDate, today)
# plotprices(title, my_stocks)


covariance_matrix_annual = profits.cov()*252

sharpeportfolio = minimize(lambda x: penalized_function(x, sharperatio, profits, covariance_matrix_annual.values, -1, 1), weights, method='Nelder-Mead',
         options={'disp': True})

print(sharpeportfolio)
sharpe = portfolioreturns(profits, sharpeportfolio.x)

x = range(20)
array = []
for n in x:
    res = minimize(lambda x: penalized_function(x, sharperatio, profits, covariance_matrix_annual.values, (sharpe-0.1+n/100), 100),
                   weights, method='Nelder-Mead',
                   options={'disp': False})
    array.append(res)

plotportfolios(array, profits, sharpeportfolio, covariance_matrix_annual)
