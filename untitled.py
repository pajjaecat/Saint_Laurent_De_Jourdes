def initVarsFor_ParEngines(pred_df, upscaling_coef, pred_cols_name):

    
    df_prodHT = pred_df.iloc[:,2:]
    df_prodHT.columns = ['P0013','P0018','P0100'] # Rename column 
    
    # Upscale P0100 based on its maximum value range
    df_prodHT.P0100 = upscaling_coef*df_prodHT.P0100/P0100_max

    df_prod_bt_total = pred_df.iloc[:,[1]]
    df_prod_bt_total.columns = ['Prod_BT']

    df_cons_total = pred_df.iloc[:,[0]]
    df_cons_total.columns = ['Cons']

    # Define a dict 
    dict_df_sgenLoad = dict({'df_prodHT':df_prodHT, 
                             'df_prod_bt_total':df_prod_bt_total, 
                             'df_cons_total':df_cons_total } )
    
    return df_prodHT, dict_df_sgenLoad



def Init_ParEngines_Parameters (df_prodHT, opf_status, dict_df_sgenLoad, parameters_dict, nb_exec ):

    if nb_exec == 0 : 
        # Clear the localspace of all engines
        dview.clear() 

        # # Import following modules on the local space of clients or engines
        with rc[:].sync_imports():
            import numpy, pandapower,pandas, Par_myFunctions

    # Share the total number of period in df_prodHT.index among all the created engines
    dview.scatter('period_part', df_prodHT.index)
    
    # Add all variables in the parameters_dict  to  the local space of each client
    dview.push(parameters_dict) 
                       
    # Send following Variables to local space of parallel engines
    dview['opf_status'] = opf_status  
    dview['dict_df_sgenLoad'] = dict_df_sgenLoad   
    
    
    
    
def block_pf_Opf(df_prodHT, opf_status): 
    
    # Run problem in parallel
    %px par_run_Results = [Par_myFunctions.run_powerflow_at(net_civaux, cur_period, net_civaux_hv_activated_bus, sum_max_main_network,  dict_df_sgenLoad, vm_mu_max, opf_status) for cur_period in period_part]
    
    results = dview.gather('par_run_Results')
    time.sleep(2) # Wait before continuing the execution

    if opf_status =='Both': 
        # Extract results
        extracted_results = mf.extract_par_results(results, df_prodHT)
        return extracted_results
    
    elif opf_status== False: 
        # Put data in dataframe
        max_vm_pu_rnn_df = pd.DataFrame(data=np.array(results), 
                                        index=df_prodHT.index, columns=['RNN'],)
        return max_vm_pu_rnn_df
    
    else: raise valueError ('<opf_status> must be either of [False, ''Both'']')


def block_prod(extracted_results, df_final, upscaling_coef):
    extracted_results.loc[per_index2[day_tot_per:], ['P0100']] = (np.minimum(df_final.loc[per_index2[day_tot_per:], 'P0100']*upscaling_coef/P0100_max,
                                                                             extracted_results.loc[per_index2[day_tot_per:], 'P0100']))
    
    return extracted_results