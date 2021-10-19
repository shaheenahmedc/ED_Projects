# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:46:36 2020

@author: shahe
"""

import numpy as np 
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def Root_Mean_Square_Scaled_Error(forecast_data, actual_forecast_horizon_data, actual_historic_data, OPLA_bool = False):
    '''
    This function calculates RMSSE, as in the M5 Forecasting competition competitor guide.
    Inputs:
    forecast_data = numpy array of forecast data for one SKU, of length h, with h being the forecast horizon 
    actual_forecast_horizon_data = numpy array of actual demand in the forecast horizon for one SKU, length h as above 
    actual_historic_data = numpy array of all historic data for one SKU, length n (number of historical observations) 
    OPLA_bool = if we're doing one-period look ahead (parameter fitting).
    '''
    assert (len(forecast_data) == len(actual_forecast_horizon_data)), "Lengths of forecast data and actual forecast horizon data don't match!"
    n = len(actual_historic_data)
    h = len(forecast_data)
    forecast_data = forecast_data.astype(float)
    actual_forecast_horizon_data = actual_forecast_horizon_data.astype(float)
    actual_historic_data = actual_historic_data.astype(float)

    numerator = np.sum((actual_forecast_horizon_data - forecast_data)**2) / h

    denominator = np.sum((np.diff(actual_historic_data)**2)) / (n - 1)
    #print (f'denominator = {denominator}')
    if (OPLA_bool & (np.isnan(denominator) or denominator == 0.0)):
        
        RMSSE = 0.0 
    else:
        RMSSE = np.sqrt(numerator / denominator) 

        
    #print (f'numerator = {numerator}')
    #print (f'denominator = {denominator}')
    
    
    return RMSSE 

def produce_sql_query_from_list_of_product_identifiers(SKUs_array):
    sql_list = "("
    for x in range(len(SKUs_array)):
        sql_list += "'NL-0000000000" + str(SKUs_array[x]) + "'" + ", " 
    sql_list = sql_list[:-2]
    sql_list += ")"

    script_after = f'SELECT MaterialNo, OrderQuantity, SalesOrderLineCreationDate FROM [ed].[SalesOrderLine] where MaterialNo in {sql_list}'
    return script_after

def write_script_to_file(script_as_string, script_filename):
    text_file = open(script_filename, "w")
    n = text_file.write(script_as_string)
    text_file.close()
   


    
def filter_out_string_entries_from_column_in_dataframe(entries_to_filter, dataframe, column_to_filter, filter_in_or_out_bool):
    '''
    A function to filter out, or only keep, a list of entries from a pandas datafrane

    Parameters
    ----------
    entries_to_filter : list[str]
        A list of entries to filter out(or in) of a dataframe.
    dataframe : pandas dataframe
        The dataframe to filter upon. 
    column_to_filter : str
        The column to filter entries out(or in).
    filter_in_or_out_bool : bool
        True = keep only entries in entries_to_filter.
        False = filter out entries in entries_to_filter. 

    Returns
    -------
    dataframe : pandas dataframe
        The filtered dataframe.
    number_of_skus_in_slim_snap : int
        The number of unique article codes remaining in the dataframe.

    '''
    if (filter_in_or_out_bool):
        dataframe = dataframe[dataframe[column_to_filter].isin(entries_to_filter)]
        number_of_skus_in_slim_snap = dataframe['Artikel Code'].nunique()
        return dataframe, number_of_skus_in_slim_snap
    else:
        dataframe = dataframe[~dataframe[column_to_filter].isin(entries_to_filter)]
        number_of_skus_in_slim_snap = dataframe['Artikel Code'].nunique()
        return dataframe, number_of_skus_in_slim_snap     
    

def append_MSEG_data_to_Slimstock_data(slimstock_dataframe, date, row, col, col_name_order_quantity):  
    bools_date_and_material_number_matching = slimstock_dataframe['Artikel Code'].isin([row['MaterialNo']]) & (row['PostingDate'] == pd.to_datetime(date))
    slimstock_dataframe.loc[bools_date_and_material_number_matching.values, col] = row[col_name_order_quantity]
    return slimstock_dataframe
    
def simple_exp_smoothing_wrapper(input_data, length_forecasts): 
    fitted_ses = SimpleExpSmoothing(input_data).fit(optimized=True)
    alpha_value = fitted_ses.params_formatted.loc['smoothing_level', 'param']
    initial_value = fitted_ses.params_formatted.loc['initial_level', 'param']
    forecast_from_fitted_ses = fitted_ses.forecast(length_forecasts)
    return forecast_from_fitted_ses, alpha_value, initial_value


def insert_SES_forecasts_to_results_dataframe(MSEG_dataframe, Results_dataframe, length_forecasts):
    MSEG_dataframe_grouped = MSEG_dataframe.groupby(['MaterialNo'])
    Results_dataframe.fillna(0.0, inplace=True)
    i = 0
    for name, group in MSEG_dataframe_grouped:
        i += 1
        print (f'Percentage complete: {100.0 * i /  len(MSEG_dataframe_grouped)}')
        group = group.sort_values(by='PostingDate') 
        group['OQ'].fillna(0.0, inplace=True)
      
        actual_historic_data_for_SES = group['OQ'].values

        SES_forecasts, tuned_alpha, tuned_init = simple_exp_smoothing_wrapper(actual_historic_data_for_SES, length_forecasts)
        cols_to_append_to = ['Forecast_1', 'Forecast_2', 'Forecast_3', 'Forecast_4', 'Forecast_5']
        
        assert (len(SES_forecasts) == len(cols_to_append_to))

        for j in range(len(SES_forecasts)):
            Results_dataframe.loc[Results_dataframe['Artikel_Code'] == name, cols_to_append_to[j]] =  SES_forecasts[j]
            
        Results_dataframe.loc[Results_dataframe['Artikel_Code'] == name, 'alpha_SES_tuned'] =  tuned_alpha
        Results_dataframe.loc[Results_dataframe['Artikel_Code'] == name, 'init_SES_tuned'] =  tuned_init

    return Results_dataframe  

def calculate_RMSSE_for_SKUs(MSEG_dataframe, Results_dataframe, forecast_method):

    MSEG_dataframe_grouped = MSEG_dataframe.groupby(['MaterialNo'])
    Results_dataframe.fillna(0.0, inplace=True)
    i = 0
    for name, group in MSEG_dataframe_grouped:
        i += 1
        print (f'Percentage complete: {100.0 * i /  len(MSEG_dataframe_grouped)}')
        group = group.sort_values(by='PostingDate') 
        group['OQ'].fillna(0.0, inplace=True)
        forecast_data = Results_dataframe.loc[(Results_dataframe['Artikel_Code'] == name), 'Forecast_1' : 'Forecast_5'] 
        # Note, we often have multiple rows in the Slimstock forecasting data for individual SKUs. 
        # Pass each row into error function. Deal with down the line. Should be fine to just add errors for each row, if we want to. 
        actual_data = Results_dataframe.loc[(Results_dataframe['Artikel_Code'] == name), 'Actual_1' : 'Actual_5'] 
        actual_historic_data_for_RMSSE = group['OQ'].values

        for ( forecast_id, forecast_row ), ( actual_id, actual_row ) in zip( forecast_data.iterrows(), actual_data.iterrows()):
            forecast_data_for_RMSSE = forecast_row.values
            actual_data_for_RMSSE = actual_row.values

            error_for_forecast_row = Root_Mean_Square_Scaled_Error(forecast_data_for_RMSSE, actual_data_for_RMSSE, actual_historic_data_for_RMSSE)
            Results_dataframe.loc[forecast_id, 'RMSSE'] = error_for_forecast_row
            Results_dataframe.loc[forecast_id, 'Forecast_method'] = forecast_method

    return Results_dataframe

def Croston(ts,extra_periods=1,alpha=0.4):
    d = np.array(ts) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods

    #level (a), periodicity(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    q = 1 #periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0]/p[0]
    # Create all the t+1 forecasts
    for t in range(0,cols):        
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            p[t+1] = alpha*q + (1-alpha)*p[t]
            f[t+1] = a[t+1]/p[t+1]
            q = 1           
        else:
            a[t+1] = a[t]
            p[t+1] = p[t]
            f[t+1] = f[t]
            q += 1

    # Future Forecast 
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]

    df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
    return df

def Croston_TSB(ts, extra_periods, parameter_vector = None):
    '''
    In order to tune TSB's parameters, we need to be able to input a parameter set to the the TSB model. 
    Thus we adjust the Croston_TSB function to have this parameter_vector input. 
    '''
    
    if (parameter_vector is None):
        alpha = 0.4
        beta = 0.4
    else:
        alpha = parameter_vector[0]
        beta = parameter_vector[1]
    
    d = np.array(ts) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods

    #level (a), probability(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0]*a[0]

    # Create all the t+1 forecasts
    for t in range(0,cols): 
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            p[t+1] = beta*(1) + (1-beta)*p[t]  
        else:
            a[t+1] = a[t]
            p[t+1] = (1-beta)*p[t]       
        f[t+1] = p[t+1]*a[t+1]

    # Future Forecast
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]
    #print (f'f = {f}')
    #print (f'f[len(ts):] = {f[len(ts):]}')
    forecasts = f[len(ts):]
    df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
    return forecasts


'''
def Croston_TSB(ts, extra_periods ,alpha=0.4,beta=0.4):
    d = np.array(ts) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d,[np.nan]*extra_periods) # Append np.nan into the demand array to cover future periods

    #level (a), probability(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0]*a[0]

    # Create all the t+1 forecasts
    for t in range(0,cols): 
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            p[t+1] = beta*(1) + (1-beta)*p[t]  
        else:
            a[t+1] = a[t]
            p[t+1] = (1-beta)*p[t]       
        f[t+1] = p[t+1]*a[t+1]

    # Future Forecast
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]
    #print (f'f = {f}')
    #print (f'f[len(ts):] = {f[len(ts):]}')
    forecasts = f[len(ts):]
    df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Period":p,"Level":a,"Error":d-f})
    return forecasts
'''
def insert_forecasts_to_results_dataframe(MSEG_dataframe, Results_dataframe, length_forecasts, forecast_method):
    MSEG_dataframe_grouped = MSEG_dataframe.groupby(['MaterialNo'])
    Results_dataframe.fillna(0.0, inplace=True)
    i = 0
    for name, group in MSEG_dataframe_grouped:
        i += 1
        print (f'Percentage complete: {100.0 * i /  len(MSEG_dataframe_grouped)}')
        group = group.sort_values(by='PostingDate') 
        group['OQ'].fillna(0.0, inplace=True)    
        actual_historic_data = group['OQ'].values
        forecasts = forecast_method(actual_historic_data, length_forecasts)
        cols_to_append_to = ['Forecast_1', 'Forecast_2', 'Forecast_3', 'Forecast_4', 'Forecast_5']   
        assert (len(forecasts) == len(cols_to_append_to))
        for j in range(len(forecasts)):
            Results_dataframe.loc[Results_dataframe['Artikel_Code'] == name, cols_to_append_to[j]] =  forecasts[j]       
    return Results_dataframe  


def create_forecast_method_df_and_forecast_and_calc_RMSSE(Results_df, MSEG_df, length_forecasts, forecast_method):
    print (len(Results_df))
    temp_df = Results_df[Results_df['Forecast_method'] == 'Slim'] # Get unique SKU set, forecast by Slim
    print (len(temp_df))
    temp_df = temp_df[['Artikel_Code', 'Actual_1', 'Actual_2', 'Actual_3', 'Actual_4', 'Actual_5']].copy()
    column_names = ['Artikel_Code', 
                    'Forecast_1', 
                    'Forecast_2', 
                    'Forecast_3', 
                    'Forecast_4', 
                    'Forecast_5', 
                    'Actual_1', 
                    'Actual_2',
                    'Actual_3',
                    'Actual_4', 
                    'Actual_5',
                    'RMSSE', 
                    'Forecast_method']
    temp_df = temp_df.reindex(columns=column_names)
    temp_df_with_forecasts = insert_forecasts_to_results_dataframe(MSEG_df, 
                                                                   temp_df, 
                                                                   length_forecasts,
                                                                   forecast_method)
    
    temp_df_with_forecasts_and_errors = calculate_RMSSE_for_SKUs(MSEG_df,temp_df_with_forecasts, 'TSB')    
    return temp_df_with_forecasts_and_errors

def insert_forecasts_to_results_dataframe_and_tune_params(MSEG_dataframe, Results_dataframe, length_forecasts, forecast_method, n_grid_search):
    MSEG_dataframe_grouped = MSEG_dataframe.groupby(['MaterialNo'])
    Results_dataframe.fillna(0.0, inplace=True)
    i = 0
    for name, group in MSEG_dataframe_grouped:
        i += 1
        print (f'Percentage complete: {100.0 * i /  len(MSEG_dataframe_grouped)}')
        group = group.sort_values(by='PostingDate') 
        group['OQ'].fillna(0.0, inplace=True)    
        actual_historic_data = group['OQ'].values
        actual_data = Results_dataframe.loc[(Results_dataframe['Artikel_Code'] == name), 'Actual_1' : 'Actual_5'] 
        
        # We're going to implement a rudimentary grid search here:
        best_param_set = 0
        best_RMSSE = 10.0**100
        best_forecast = 0
        param_values = np.random.uniform(low=0.0,high=0.5,size=(n_grid_search,2))

        for param_set in range(param_values.shape[0] - 1):
            # If you see 'opla' going forward, it's shorthand for 'one-period look ahead'
            Total_RMSSE_opla_for_single_param_vector = 0.0 
            #Loop through the historical data, producing a forecast for the next timestep ('opla')
            for data_point_index in range(len(actual_historic_data) - 1):
                #print (f'data_point_index = {data_point_index}')
                
                forecast_of_next_data_point = forecast_method(actual_historic_data[:data_point_index+1], 
                                                              1, 
                                                              param_values[param_set])

                next_data_point = actual_historic_data[data_point_index + 1]

                historic_data_for_opla = actual_historic_data[:data_point_index + 1]

                RMSSE_opla_for_data_point = Root_Mean_Square_Scaled_Error(np.array([forecast_of_next_data_point]),
                                                                            np.array([next_data_point]),
                                                                            historic_data_for_opla,
                                                                            OPLA_bool = True)

                Total_RMSSE_opla_for_single_param_vector += RMSSE_opla_for_data_point
            
            # For fitting the parameters of Croston/SES, do we use MSE or RMSSE? 
            # MSE will surely force the parameters to produce zero forecasts. 
            # Let's use RMSSE.             

            if (Total_RMSSE_opla_for_single_param_vector < best_RMSSE):
                best_RMSSE = Total_RMSSE_opla_for_single_param_vector 
                best_param_set = param_values[param_set]
                
                # Use the current best param set to actually produce the required multiple period look ahead forecasts
                forecasts_using_current_best_param_set = forecast_method(actual_historic_data, 
                                                                         length_forecasts, 
                                                                         best_param_set)
                best_forecast = forecasts_using_current_best_param_set
        
        # Need to sort this. How do we calculate MSE/RMSSE, using our existing RMSSE function, and 
        # how do we use our forecasting function for this? We haven't built functions to do one-period
        # look ahead, instead to forecast and calc RMSSE over a forecast horizon. Will need to edit them 
        # if we want to do manual param optimization
        cols_to_append_to = ['Forecast_1', 'Forecast_2', 'Forecast_3', 'Forecast_4', 'Forecast_5']   
        assert (len(best_forecast) == len(cols_to_append_to))
        Results_dataframe.loc[Results_dataframe['Artikel_Code'] == name, 'alpha_TSB'] =  best_param_set[0]       
        Results_dataframe.loc[Results_dataframe['Artikel_Code'] == name, 'beta_TSB'] =  best_param_set[1]     
        for j in range(len(best_forecast)):
            Results_dataframe.loc[Results_dataframe['Artikel_Code'] == name, cols_to_append_to[j]] =  best_forecast[j]  
  
    return Results_dataframe

def create_forecast_method_df_and_forecast_and_calc_RMSSE_with_tuned_params(Results_df, Ted_df, length_forecasts, forecast_method, n_grid_search, name_for_forecast_method_col):
    print (len(Results_df))
    temp_df = Results_df[Results_df['Forecast_method'] == 'Slim'] # Get unique SKU set, forecast by Slim
    print (len(temp_df))
    temp_df = temp_df[['Artikel_Code', 'Actual_1', 'Actual_2', 'Actual_3', 'Actual_4', 'Actual_5']].copy()
    column_names = ['Artikel_Code', 
                    'Forecast_1', 
                    'Forecast_2', 
                    'Forecast_3', 
                    'Forecast_4', 
                    'Forecast_5', 
                    'Actual_1', 
                    'Actual_2',
                    'Actual_3',
                    'Actual_4', 
                    'Actual_5',
                    'RMSSE', 
                    'Forecast_method']
    temp_df = temp_df.reindex(columns=column_names)
    temp_df_with_forecasts = insert_forecasts_to_results_dataframe_and_tune_params(Ted_df, 
                                                                   temp_df, 
                                                                   length_forecasts,
                                                                   forecast_method, 
                                                                   n_grid_search)
    
    temp_df_with_forecasts_and_errors = calculate_RMSSE_for_SKUs(Ted_df,temp_df_with_forecasts, name_for_forecast_method_col)    
    return temp_df_with_forecasts_and_errors