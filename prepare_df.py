import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import garch


OPTION_DATASET_FILE = 'SPX04-07_final.csv'
FED_TBILLS_FILE = 'TBills-04-07-monthly.csv'
UNDERLYING_ASSET_FILE = 'SPX_underlying_cboe.csv'

def load_data(filename):
    """Loads dataset from .csv file"""
    rows = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    return(rows)


def transform_date_to_datetime(date):
    """Auxiliary function to create dataframe"""
    date = date.split("/")
    date = datetime.datetime(int(date[2]), int(date[0]), int(date[1]))
    return(date)


def deploy_data_structure(variables, rows):
    """Auxiliary function to create dataframe"""
    data = []
    for i in range(len(rows)):
        dict1 = dict()
        for j in range(len(rows[i])-1):
            if j == 0 or j == 8:
                rows[i][j] = transform_date_to_datetime(rows[i][j])
            dict1[variables[j]] = rows[i][j]
        data.append(dict1)
    return(data)


def prepare_dataframe():
    """loads data, change date to datetime format and prepares pd.DataFrame"""
    rows = load_data(OPTION_DATASET_FILE)
    data = deploy_data_structure(rows[0], rows[1:])
    pdata = pd.DataFrame(rows[1:], columns=rows[0])

    # transform data type
    pdata[['maturity']] = pdata[['maturity']].astype("int")
    pdata[['close']] = pdata[['close']].astype("float")
    pdata[['strike']] = pdata[['strike']].astype("float")
    pdata[['bid']] = pdata[['bid']].astype("float")
    pdata[['ask']] = pdata[['ask']].astype("float")
    pdata[['optionid']] = pdata[['optionid']].astype("int")
    pdata[['implied']] = pdata[['implied']].astype("float")
    pdata[['interest']] = pdata[['interest']].astype("float")
    pdata[['divrate']] = pdata[['divrate']].astype("float")

    # add new columns
    pdata['normmat'] = pdata['maturity']/365
    pdata['mid'] = (pdata['bid']+pdata['ask'])/2
    pdata['moneyness'] = pdata['close']/pdata['strike']
    pdata['mid_strike'] = pdata['mid']/pdata['strike']
    return(pdata)


def add_risk_free_rate_from_FED():
    """Obtains risk free rate from FED"""
    riskfreerate = load_data(FED_TBILLS_FILE)
    for row in riskfreerate[1:]:
        row[0] = transform_date_to_datetime(date=row[0])
        row[1] = float(row[1])

    riskdf = pd.DataFrame(riskfreerate[1:], columns=riskfreerate[0])
    return(riskdf)


def add_risk_free_rate_from_FED_to_pdata(pdata):
    """Appends risk free rate column"""
    riskdf = add_risk_free_rate_from_FED()

    if 'discount-monthly' not in pdata.columns:
        pdata = pdata.join(riskdf.set_index('date-rf'), on='date')
    return(pdata)


def prepare_underlying_asset(pdata):
    """Create dataframe for the underlying asset from the option data"""
    underlying = pdata.duplicated(subset="date")
    underlying = np.invert(underlying)
    underlying = pdata[underlying]
    underlying = underlying.sort_values(by="date")
    underlying = underlying[['date', 'close']]

    underlying_cboe = load_data(UNDERLYING_ASSET_FILE)
    for row in underlying_cboe[1:]:
        row[0] = transform_date_to_datetime(date=row[0])
        row[1] = float(row[1])
    cboe_df = pd.DataFrame(underlying_cboe[1:], columns=underlying_cboe[0])

    underlying = cboe_df.join(underlying.set_index('date'), on='date_cboe')

    underlying['returns'] = np.log(
        underlying.close_cboe/underlying.close_cboe.shift(1))
    underlying['volatility5'] = pd.Series.rolling(
        underlying.returns, window=5).std()*np.sqrt(252)
    underlying['volatility20'] = pd.Series.rolling(
        underlying.returns, window=20).std()*np.sqrt(252)
    underlying['volatility60'] = pd.Series.rolling(
        underlying.returns, window=60).std()*np.sqrt(252)
    underlying['volatility100'] = pd.Series.rolling(
        underlying.returns, window=100).std()*np.sqrt(252)

    underlying = underlying[103:-395]  # subsetting for existing option data
    underlying['day'] = range(len(underlying))
    underlying['intercept'] = np.ones(len(underlying))
    underlying = garch.garch(underlying)
    return(underlying)


def append_volatility_columns(pdata, underlying):
    """Appends the volatility of underlying asset to the option dataframe"""
    if 'volatility5' not in pdata:
        pdata = pdata.join(underlying[['date_cboe', 'volatility5', 'volatility20',
                                       'volatility60', 'volatility100', 'vol_garch']].set_index('date_cboe'), on='date')

    return(pdata)


def prepare_train_test_set(pdata):
    """Prepares the train and test set for both call options and put options"""
    no_options = len(pdata)
    no_calls = len(pdata[pdata['cpflag'] == "C"])
    no_puts = no_options-no_calls
    print("Options:", no_options, "Calls:", no_calls, "Puts:", no_puts)
    c_train_bound = math.floor(no_calls/5*4)
    print("Calls in train set (80%):", c_train_bound)
    p_train_bound = no_calls+math.floor(no_puts/5*4)
    print("Puts in train set (80%):", p_train_bound-no_calls)
    calls = pdata[:no_calls]
    puts = pdata[no_calls:]
    c_tr = calls[:c_train_bound]
    c_test = calls[c_train_bound:]
    p_tr = puts[:p_train_bound-no_calls]
    p_test = puts[p_train_bound-no_calls:]
    return(no_options, no_calls, no_puts, calls, puts, c_tr, c_test, p_tr, p_test)


def prepare_train_test_set_module(module):
    """Prepares the train and test set for one specific module in the modular neural network"""
    no_options = len(module)
    train_bound = math.floor(no_options/5*4)
    set_tr = module.sample(train_bound)
    set_tr_indices = set(set_tr.index)
    all_indices = set(module.index)
    set_test_indices = all_indices.difference(set_tr_indices)
    set_test = module.ix[list(set_test_indices)]
    return(set_tr, set_test)
