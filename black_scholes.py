"""Black Scholes functions"""
import scipy.stats
import numpy as np


def BS_d1(S, X, r, tau, sigma):
    """Auxiliary function for BS model"""
    d1 = (np.log(S/X)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    return(d1)


def BS_d2(S, X, r, tau, sigma):
    """Auxiliary function for BS model"""
    d2 = BS_d1(S, X, r, tau, sigma)-sigma*np.sqrt(tau)
    return(d2)


def BS(S, X, r, tau, sigma, cpflag):
    """Standard Black-Scholes formula for call and put options"""
    d1 = BS_d1(S, X, r, tau, sigma)
    d2 = BS_d2(S, X, r, tau, sigma)
    if cpflag == "C":
        C = S*scipy.stats.norm.cdf(d1)-X*np.exp(-r*tau) * \
            scipy.stats.norm.cdf(d2)
        return(C)
    elif cpflag == "P":
        P = -S*scipy.stats.norm.cdf(-d1)+X * \
            np.exp(-r*tau)*scipy.stats.norm.cdf(-d2)
        return(P)


def df_BS_function(df, *args):
    """Function indexing columns in pandas dataframe"""
    S = df[1]  # close
    X = df[2]  # strike
    r = df[4]  # /100 #discount-monthly
    tau = df[12]  # "normmat"
    sigma = df[args[0]]  # volat100
    cpflag = df[9]  # call or put
    C = BS(S, X, r, tau, sigma, cpflag)
    return(C)


def compute_and_append_black_scholes_columns(pdata):
    """Appending BS results to pandas dataframe"""
    pdata['BS5'] = pdata.apply(df_BS_function, args=[17], axis=1)
    pdata['BS20'] = pdata.apply(df_BS_function, args=[18], axis=1)
    pdata['BS60'] = pdata.apply(df_BS_function, args=[19], axis=1)
    pdata['BS100'] = pdata.apply(df_BS_function, args=[20], axis=1)
    pdata['BSgarch'] = pdata.apply(df_BS_function, args=[21], axis=1)


def append_moneyness_columns(pdata):
    """Appending moneyness results to pandas dataframe"""
    pdata['BS5-strike'] = pdata['BS5']/pdata['strike']
    pdata['BS20-strike'] = pdata['BS20']/pdata['strike']
    pdata['BS60-strike'] = pdata['BS60']/pdata['strike']
    pdata['BS100-strike'] = pdata['BS100']/pdata['strike']
    pdata['BSgarch-strike'] = pdata['BSgarch']/pdata['strike']
