"""GARCH - Generalized Autoregressive Conditional Heteroskedasticity"""
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from xyplot_core import *
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt


def garch(underlying):
    """Calculates GARCH based on the underlying asset pandas dataframe"""
    r = underlying['returns']
    garch11 = arch_model(r, p=1, q=1)
    res = garch11.fit(update_freq=10)
    print(res.summary())
    params = list(res.params)
    print(params)
    residuals = res.resid
    res2 = list(residuals*residuals)
    sigma2 = np.zeros(len(residuals))

    for i in range(0, len(sigma2)-1):
        sigma2[i+1] = params[1]+params[2]*sigma2[i]+params[3]*res2[i]
    sigma2[0] = 0.05

    sigma = np.sqrt(sigma2)*np.sqrt(252)
    underlying['vol_garch'] = sigma
    return(underlying)


def plot_garch(underlying):
    """Plots volatility (GARCH and MA5 volatility)"""
    plot1 = MyPlot()
    plot1.append_data(underlying['day'], underlying['vol_garch'],
                      'r', '$\sigma_{GARCH(1,1)}$', linewidth=1.0)
    plot1.append_data(
        underlying['day'], underlying['volatility5'], 'k', '$\sigma_{MA5}$', linewidth=2.0)
    plot1.construct_plot("Volatility", "Date", "Volatility ($\sigma$)", save="volatility_garch.png",
                         xticks_bool=True, xymin=[0, 0], xymax=[761, 0.30], figsize=(10, 5))
