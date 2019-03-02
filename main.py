"""This is the main script.
It was used sequentially with jupyter notebooks.
If you are interested in code in more detail, 
consider using jupyter notebooks too."""

import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import prepare_df
import side_data_analysis
import black_scholes
import modular_nn

pdata = prepare_df.prepare_dataframe()

# side_data_analysis functions are not necessary for the main code
# it shows the basic overview about the data structure
side_data_analysis.plot_particular_option(pdata, 1400, "C")
side_data_analysis.plot_particular_option(pdata, 1300, "C")
side_data_analysis.plot_particular_option(pdata, 1200, "C")
side_data_analysis.plot_particular_option(pdata, 1100, "C")

side_data_analysis.plot_particular_option(pdata, 1400, "P")
side_data_analysis.plot_particular_option(pdata, 1300, "P")
side_data_analysis.plot_particular_option(pdata, 1200, "P")
side_data_analysis.plot_particular_option(pdata, 1100, "P")

# side_data_analysis.plot_particular_moneyness(pdata,"C")
# side_data_analysis.plot_particular_moneyness(pdata,"P")


# Adds columns to the option dataframe and creates dataframe for underlying asset
pdata = prepare_df.add_risk_free_rate_from_FED_to_pdata(pdata)
underlying = prepare_df.prepare_underlying_asset(pdata)
pdata = prepare_df.append_volatility_columns(pdata, underlying)

side_data_analysis.plot_close(underlying)
side_data_analysis.plot_returns(underlying)
side_data_analysis.plot_volatilities(underlying)


# Takes long time - adding Black Scholes results to the dataframe
pdata = black_scholes.compute_and_append_black_scholes_columns(pdata)
pdata = black_scholes.append_moneyness_columns(pdata)

side_data_analysis.plot_black_scholes_prediction(pdata)

no_options, no_calls, no_puts, calls, puts, c_tr, c_test, p_tr, p_test = prepare_df.prepare_train_test_set(
    pdata)
calls_mod, puts_mod = modular_nn.divide_options_to_modules(calls, puts)

# side_data_analysis.summary_table(calls_mod)
# side_data_analysis.summary_table(puts_mod)
# side_data_analysis.volatility_table(calls_mod)
# side_data_analysis.volatility_table(puts_mod)

# The following code calculates simulations for all different setups
# It is necessary to adjust the parameters in modular_nn file
# If uncommented - initially working with modular neural network architecture
"""
#from collections import Counter
#strikes_count=Counter(list(pdata.strike))
#print(strikes_count,max(strikes_count.keys()))

#calls_mod.append(calls)
#puts_mod.append(puts)
#all_mod = calls_mod+puts_mod
all_mod = [calls,puts]

cpflag="C"
a=0
for module in all_mod:
    a=a+1
    if a==2:
        cpflag="P"
    set_tr,set_test=prepare_df.prepare_train_test_set_module(module)    
    varlist_vol=['normmat','moneyness','volatility5','volatility20','volatility60','volatility100']
    #varlist_vol=['normmat','moneyness','vol_garch']
    garch=False    
    X_tr,y_tr,X_test,y_test=modular_nn.load_train_test_set(set_tr,set_test,varlist=varlist_vol)
    
    X_tr_v = np.copy(X_tr)
    y_tr_v = np.copy(y_tr)
    virtual=0
    for number_of_virtual in [0]:
        if number_of_virtual!=0:
            for i in range(len(underlying)):
                if i%100==0:
                    print(i,"/",len(underlying))
                X_tr_v,y_tr_v=modular_nn.virtual_call_option(underlying,X_tr_v,y_tr_v,i,number_of_virtual,cpflag=cpflag,garch=garch)
            print(len(X_tr_v),len(y_tr_v),len(X_tr),len(y_tr_v))    
            virtual = len(y_tr_v)-len(y_tr)
        models=modular_nn.run_neural_network(underlying,X_tr_v,y_tr_v,X_test,y_test,varlist_vol,set_test,virtual)
"""


# The following code simulates options according to the resolution tests (Chapter 4.2.1)
""" 
bsvirtual_X_tr=X_tr[0:0]
bsvirtual_X_test=X_test[0:0]
bsvirtual_y_tr=y_tr[0:0]
bsvirtual_y_test=y_test[0:0]

for i in range(40):
    #if i%100==0:
    print(i)
    bsvirtual_X_tr,bsvirtual_y_tr=modular_nn.bsvirtual_append_lot(bsvirtual_X_tr,bsvirtual_y_tr,underlying)
for i in range(50):
    print(i)
    bsvirtual_X_test,bsvirtual_y_test=modular_nn.bsvirtual_append_lot(bsvirtual_X_test,bsvirtual_y_test,underlying)

list1=[3] # This number depends on number of simulated options
for i in list1:
    models=modular_nn.run_neural_network(underlying,X_tr[0:(i+1)*10000],bsvirtual_y_tr[0:(i+1)*10000],X_test,bsvirtual_y_test,varlist_vol,c_test)
"""

# The following code can be used to save data in .xlsx format
"""
writer = pd.ExcelWriter("bsvirtual_X_tr.xlsx")
pd.DataFrame(bsvirtual_X_tr).to_excel(writer,'Sheet1')
writer.save()  
writer = pd.ExcelWriter("bsvirtual_y_tr.xlsx")
pd.DataFrame(bsvirtual_y_tr).to_excel(writer,'Sheet1')
writer.save()  
writer = pd.ExcelWriter("bsvirtual_X_test.xlsx")
pd.DataFrame(bsvirtual_X_test).to_excel(writer,'Sheet1')
writer.save()  
writer = pd.ExcelWriter("bsvirtual_y_test.xlsx")
pd.DataFrame(bsvirtual_y_test).to_excel(writer,'Sheet1')
writer.save()  
"""
