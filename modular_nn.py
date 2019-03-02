import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras import backend
import keras.activations
import pandas as pd
import nn_plotting
import black_scholes as bs
import prepare_df
import random


def divide_options_to_modules(calls, puts):
    """splits the initial option dataset to modules"""
    # thresholds
    m = [0.0, 0.97, 1.05, 100]  # moneyness
    t = [0.0, 0.165, 0.5, 10.0]  # time to maturity

    # Calls
    calls_mod = []
    for i in range(3):
        b1 = calls.moneyness > m[i]
        b2 = calls.moneyness < m[i+1]
        print("Moneyness", len(calls[b1 & b2]))
        for j in range(3):
            b3 = calls.normmat > t[j]
            b4 = calls.normmat < t[j+1]
            btotal = b1 & b2 & b3 & b4
            print("Both", len(calls[btotal]))
            calls_mod.append(calls[btotal])
    # Puts
    puts_mod = []
    for i in range(3):
        b1 = puts.moneyness > m[i]
        b2 = puts.moneyness < m[i+1]
        print("Moneyness", len(puts[b1 & b2]))
        for j in range(3):
            b3 = puts.normmat > t[j]
            b4 = puts.normmat < t[j+1]
            btotal = b1 & b2 & b3 & b4
            print("Both", len(puts[btotal]))
            puts_mod.append(puts[btotal])
    return(calls_mod, puts_mod)


def virtual_call_option(underlying, X_tr_v, y_tr_v, index, number_of_virtual, cpflag="C", garch=False):
    """Adds virtual option according to the condition C5"""
    step = 100
    date = underlying.date_cboe.iloc[index]
    vol5 = underlying.volatility5.iloc[index]
    vol20 = underlying.volatility20.iloc[index]
    vol60 = underlying.volatility60.iloc[index]
    vol100 = underlying.volatility100.iloc[index]

    vol_garch = underlying.vol_garch.iloc[index]
    for i in range(number_of_virtual):
        s = np.random.choice(list(range(1100, 1501)))
        x = np.random.choice(list(range(1000, 1625, 25)))

        S = s
        X = x
        if cpflag == "P":
            c = max(X-S, 0)
        else:
            c = max(S-X, 0)
        moneyness = S/X
        normmat = 0
        mid_strike = c/X
        new_y_row = np.array([mid_strike])

        if garch:
            new_X_row = np.array([normmat, moneyness, vol_garch])
        else:
            new_X_row = np.array(
                [normmat, moneyness, vol5, vol20, vol60, vol100])
        X_tr_v = np.vstack([X_tr_v, new_X_row])
        y_tr_v = np.hstack([y_tr_v, new_y_row])

    return(X_tr_v, y_tr_v)


def virtual_call_option_C6(underlying, X_tr_v, y_tr_v, index, 
                           number_of_virtual, cpflag="C"):
    """Adds virtual option according to the condition C6"""
    date = underlying.date_cboe.iloc[index]
    vol5 = underlying.volatility5.iloc[index]
    vol20 = underlying.volatility20.iloc[index]
    vol60 = underlying.volatility60.iloc[index]
    vol100 = underlying.volatility100.iloc[index]
    for i in range(number_of_virtual):
        s = np.random.choice(list(range(1100, 1501)))
        c = s
        moneyness = S/1  # X=1 (in order to avoid division by zero)
        normmat = np.random.choice(list(range(0, 3, 0.0033)))
        mid_strike = c/1
        new_y_row = np.array([mid_strike])
        new_X_row = np.array([normmat, moneyness, vol5, vol20, vol60, vol100])
        X_tr_v = np.vstack([X_tr_v, new_X_row])
        y_tr_v = np.hstack([y_tr_v, new_y_row])

    return(X_tr_v, y_tr_v)


def bsvirtual_call_option(underlying):
    """Randomly simulates options matching BS formula 
    (according to the Chapter 4.2.1)"""
    indices = list(underlying.index)
    index = random.sample(indices, 1)[0]
    S = random.sample(list(range(1100, 1501)), 1)[0]  # stock price - choose
    X = random.sample(list(range(1000, 1625, 25)), 1)[
        0]  # strike price - choose
    cpflag = "C"  # cpflag - choose
    moneyness = S/X
    normmat = random.sample(list(range(1, 1001)), 1)[0]/365  # choose
    date_cboe = underlying.date_cboe[index]
    risk_df = prepare_df.add_risk_free_rate_from_FED()
    try:
        discount_index = list(risk_df["date-rf"]).index(date_cboe)
        r = risk_df["discount-monthly"][discount_index]/100  # in percents
    except: #If the value is missing
        r = np.mean(list(risk_df["discount-monthly"]))/100
    vol5 = underlying["volatility5"][index]
    vol20 = underlying["volatility20"][index]
    vol60 = underlying["volatility60"][index]
    vol100 = underlying["volatility100"][index]

    c = bs.BS(S, X, r, normmat, vol100, cpflag)
    mid_strike = c/X
    # print(c)
    return(mid_strike, normmat, moneyness, vol5, vol20, vol60, vol100)


def bsvirtual_append(X_tr, y_tr, underlying):
    """Appends simulated options to the dataframe"""
    mid_strike, r, S, normmat, moneyness, vol5, vol20, 
    vol60, vol100 = bsvirtual_call_option(underlying)
    new_y_row = np.array([mid_strike])
    new_X_row = np.array(
        [r, S, normmat, moneyness, vol5, vol20, vol60, vol100])
    X_tr = np.vstack([X_tr, new_X_row])
    y_tr = np.hstack([y_tr, new_y_row])
    return(X_tr, y_tr)


def bsvirtual_append_lot(X_tr, y_tr, underlying):
    """Appends simulated options to the dataframe in loop (faster)"""
    listX = []
    listy = []
    for i in range(1000):
        mid_strike = 0
        while mid_strike == 0:  # because there was evaluation problem when mid_strike was 0
            mid_strike, normmat, moneyness, vol5, vol20, 
            vol60, vol100 = bsvirtual_call_option(underlying)
        listX.append([normmat, moneyness, vol5, vol20, vol60, vol100])
        listy.append(mid_strike)
    new_X_rows = np.array(listX)
    new_y_rows = np.array(listy)
    X_tr = np.vstack([X_tr, new_X_rows])
    y_tr = np.hstack([y_tr, new_y_rows])
    return(X_tr, y_tr)

# ====================== NEURAL NETWORKS ======================


def load_train_test_set(c_tr, c_test, varlist=['normmat', 'moneyness']):
    """Defaultly defined for call options"""
    X_tr = c_tr[varlist].values
    y_tr = c_tr['mid_strike'].values
    X_test = c_test[varlist].values
    y_test = c_test['mid_strike'].values
    return(X_tr, y_tr, X_test, y_test)


def activation_function(inp):
    return(backend.sigmoid(inp))


def custom_activation(x):
    return backend.exp(x)


def build_model(inp_size, hidden_layers=1, nodes=50, mtype="sigmoid", output="linear", drop=0.0):
    """Function that builds the neural network model with various hyperparameters
    activation function in hidden layers (mtype variable)
    output function (output variable)"""

    model = Sequential()
    model.add(Dense(nodes, input_dim=inp_size))
    if mtype == "sigmoid":
        for _ in range(hidden_layers):
            model.add(Dense(nodes, activation='sigmoid'))
            model.add(Dropout(drop))

    if mtype == "softplus":
        for _ in range(hidden_layers):
            model.add(Dense(nodes, activation='softplus'))
            model.add(Dropout(drop))

    if mtype == "special-culkin":
        model.add(LeakyReLU())
        model.add(Dense(nodes, activation='elu'))
        model.add(Dense(nodes, activation='relu'))
        model.add(Dense(nodes, activation='elu'))

    if output == "linear":
        model.add(Dense(1, activation='linear'))
    if output == "softplus":
        model.add(Dense(1, activation='softplus'))
    if output == "sigmoid":
        model.add(Dense(1, activation='sigmoid'))
    if output == "exp":
        model.add(Dense(1))
        model.add(Activation(custom_activation))

    model.compile(loss='mse', optimizer='adam')
    return(model)


def evaluation(y, y_pred):
    """evaluates the prediction errors"""
    MAE = np.mean(abs(y-y_pred))
    RMSE = np.sqrt(np.mean((y-y_pred)**2))
    MAPE = np.mean(abs((y-y_pred)/y))
    return(MAE, RMSE, MAPE)


def fit_and_predict_model(model, X_tr, y_tr, X_test, y_test, batch=64, epochs=50):
    """fits model and evaluates the errors"""
    model.fit(X_tr, y_tr, batch_size=batch, epochs=epochs,
              validation_split=0.25, verbose=2)
    y_tr_pred = model.predict(X_tr)[:, 0]
    y_test_pred = model.predict(X_test)[:, 0]
    MAE, RMSE, MAPE = evaluation(y_test, y_test_pred)
    return(y_tr_pred, y_test_pred, MAE, RMSE, MAPE)


def output_models_to_excel(models, filename):
    """Saves data to .xlsx file"""
    writer = pd.ExcelWriter(filename)
    models.to_excel(writer, 'Sheet1')
    writer.save()


def run_neural_network(underlying, X_tr, y_tr, X_test, y_test, varlist_vol, set_test, virtual=0, cpflag="C", garch=False):
    """Calculate the neural network prediction - defaultly works for call options"""

    inp_size = len(varlist_vol)
    virtual = 0
    filename = "models.xlsx"
    try:
        models = pd.read_excel(filename)
    except:  # Create new file if it does not exist
        models = pd.DataFrame(columns=['cpflag', 'activation', 'output', 'hidden_layers',
                                       'nodes', 'dropout', 'MAE', 'RMSE', 'MAPE', 'virtual', 'DM', 'DM-pvalue'])
        filename = "models_new.xlsx"

    dropouts = [0.0, 0.10, 0.20]
    outputs = ['linear', 'softplus', 'exponential']

    try:
        for h in range(2, 3):  # number of hidden layers
            for out in outputs:

                # for d in dropouts: # Dropout technique had not significant effect - thus omitted
                d = 0.0
                mtype = 'sigmoid'
                for i in range(3, 4):
                    nodes = i*10
                    model = build_model(
                        inp_size, nodes=nodes, hidden_layers=h, drop=d, output=out, mtype=mtype)
                    y_tr_pred, y_test_pred, MAE, RMSE, MAPE = fit_and_predict_model(
                        model, X_tr, y_tr, X_test, y_test)
                    models = models.append({"cpflag": cpflag, "activation": mtype, "output": out, "hidden_layers": h, "nodes": nodes, "dropout": d,
                                            "virtual": virtual, "MAE": MAE, "RMSE": RMSE, "MAPE": MAPE, "DM": a, "DM-pvalue": b, "size": len(X_tr)}, ignore_index=True)
                    nn_plotting.plot_prediction(y_tr, y_tr_pred, str(cpflag)+"_"+str(mtype)+"_h"+str(
                        h)+"_n"+str(nodes)+"_d"+str(d)+"_virtual"+str(virtual)+"_"+str(len(X_tr)))
                    nn_plotting.plot_errors(y_tr, y_tr_pred, str(cpflag)+"_"+str(mtype)+"_h"+str(
                        h)+"_n"+str(nodes)+"_d"+str(d)+"_virtual"+str(virtual)+"_"+str(len(X_tr)))
                    print(models)
        output_models_to_excel(models, filename)
    except KeyboardInterrupt:
        output_models_to_excel(models, filename)
    return(models)
