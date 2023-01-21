# List of function used in my code

import pandapower, pandas, numpy, ipyparallel, oriClass
from tqdm import tqdm  # Profiling

# Import all variables from module oriVariables
from oriVariables import (network_folder, excel_folder, 
                          default_auth_vm_pu_max, default_auth_vm_pu_min, default_ctrl_hvProd_max)
import checker 

pd = pandas
np = numpy 
pp = pandapower
ipp = ipyparallel



##############################         FUNCTIONS          #########################################

def readAndReshape_excelFile(f_name, folder_name=excel_folder, n_row2read=None):
    """
    Read and reshape in a one dimension array (that is returned) the excel file given by f_name


    Parameters: 
    ----------- 
    f_name: str
        Name of the file to load (with the correct extension)
    folder_name: str
        Location of the folder where the file is present
    n_row2read : Int (default=0) 
         Numbers of rows to read in the excel file.

    """

    filename = f"{folder_name}{f_name}"
    cols_to_read = range(2, 8)  # Define index of columns to read 
                                # 0 10 20 30 40 50 
                                # Where each of the six columns to read represent a period.
    input_data = pandas.read_csv(filename,
                                 header=None,
                                 sep=";",
                                 usecols=cols_to_read,
                                 nrows=n_row2read)

    return numpy.array(input_data).reshape(-1) / 1000  # /1000 To convert data (MW)


# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________

def check_bus_connection(network, bus_number, attr_list):
    """
    Check and print the connection between a bus number and all the elements in the lower network.

    Parameters:
    ----------
    network: pandapower network
        The network that has to be investigated
    bus_number: list of int
        The number of the concerned bus(ses)
    attr_list: list of String tuple
        Each tuple in the list represent the attribute of the attribute to look for
        Ex: attr_list[0] = ('bus', 'name') ==> network.bus.name must be accessed
    
    """

    for cur_bus in bus_number:  # For each bus
        for attribute in attr_list:  # For each tuple in the attibute list
            netsub = getattr(network, attribute[0])
            netsub_sub = getattr(netsub, attribute[1])

            if len(netsub[netsub_sub == cur_bus]) != 0:  # If there is some elements
                print(
                    f'----------******            Bus {cur_bus} net.{attribute[0]}.{attribute[1]}         ******-------')
                print(netsub[netsub_sub == cur_bus], '\n')
        print('\n')


# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________

def initLowerNet_at(network: pandapower.auxiliary.pandapowerNet,
                    curr_period: pandas._libs.tslibs.period,
                    sum_max_p_mw_upperNet: tuple,
                    dict_df_sgenLoad: dict):
    """
    Return a fixed float;

    Initialise the parameters of the network at the current period

    Parameters:
    ----------
    network: Pandapower network
        The lower level network concerned ;
    curr_period: Pandas period
        The current period to investigate;
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the upper level network (here, saint laurent 
        compared to the lower level network civaux)
        + Of all BT energy producers => sum_max_input[0]
        + of all Load in the network => sum_max_input[1] 
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year) of the 
        + df_prodHT         => HT producers in the subnetwork 
        + df_prod_bt_total  => BT producers in the subnetwork
        + df_cons_total     => Load demand subnetwork 


    """
    ##  TODO : Give only the data of the current period to
    ##  the function instead of that of the whole year

    # Initiate parameters to be used within funtion
    upNet_sum_max_lvProd = sum_max_p_mw_upperNet[0]
    upNet_sum_max_load = sum_max_p_mw_upperNet[1]

    df_prodHT = dict_df_sgenLoad['df_prodHT']
    df_prod_bt_total = dict_df_sgenLoad['df_prod_bt_total']
    df_cons_total = dict_df_sgenLoad['df_cons_total']

    # Initalise HT producers 
    network.sgen.p_mw[network.sgen.name.notna()] = df_prodHT.loc[curr_period].values

    # Initialize Bt producers
    prod_bt_total_1mw = df_prod_bt_total.loc[curr_period].values/upNet_sum_max_lvProd
    network.sgen.p_mw[network.sgen.name.isna()] = (network.sgen.max_p_mw[network.sgen.name.isna()] 
                                                   *prod_bt_total_1mw
                                                  )
    # Initialize Loads
    load_total_1mw = df_cons_total.loc[curr_period].values/upNet_sum_max_load
    network.load.p_mw = (network.load.max_p_mw*load_total_1mw)

    # Work with julia Power model since the load is zero
    # network.load.p_mw = (network.load.max_p_mw*df_cons_total.loc[curr_period].
    # values*0/upNet_sum_max_load)


# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________

def max_vm_pu_at(network: pandapower.auxiliary.pandapowerNet,
                 curr_period: pandas._libs.tslibs.period,
                 net_hv_activated_bus: list,
                 dict_df_sgenLoad: dict,
                 opf_status=False):
    """
    Return a fixed float;

    Return the maximum voltage over all the bus in the network for the current period.

    Parameters:
    ----------
    network: Pandapower network
        The network ;
    curr_period: Panda period
        The current period to investigate;
    net_hv_activated_bus: List
        List of all the higher voltage activated bus in the network
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year) of the 
        + df_prodHT         => HT producers in the subnetwork 
        + df_prod_bt_total  => BT producers in the subnetwork
        + df_cons_total     => Load demand subnetwork 
    ofp_status: Boolean = False
        Wether the maximum voltage is computed after a normal or optimal power flow or both
        + Normal  =>  **pandapower.runpp(net)**,  ofp_status = False
        + Optimal =>  **pandapower.runopp(net)**, ofp_status = True
                  
    
    """

    # Initiate parameters from input
    df_prodHT = dict_df_sgenLoad['df_prodHT']

    if opf_status:  # If status is true
        
        # Extract the name and the index of the controlled Higher voltage producer.
        # This supposed the there is a controllable column in the network, this controllable column 
        # is true for the controlled HV producer
        ctrld_hvProd_name = list(network.sgen[network.sgen.controllable].name)[0]
        ctrld_hvProd_ind = list(network.sgen[network.sgen.controllable].index)[0]
        
        # update 
        # For optimal flow, given that the sgen P0100 is contollable the optimization 
        # result is to draw the maximum power  with no regard to the actual power provided 
        # at each instant. To eliavate this problem we would rather initialize the maximum 
        # power of the said  producer with the actual prooduction. 
        network.sgen.at[ctrld_hvProd_ind, 'max_p_mw'] = df_prodHT[ctrld_hvProd_name][curr_period]  
        
        pandapower.runopp(network)  # Run network
        # pandapower.runpm_ac_opf(network) # Run network with Julia Power model:
        # Not converging for the moment, but Do converge when le load demand is low

    else:
        pandapower.runpp(network)  # Run network
        # pandapower.runpm_pf(network) # Run network with Julia Power model:
        # Not converging for the moment, but Do converge when le load demand is low

    # Return the maximum voltage over all the busses in the lower network for the current instant
    return network.res_bus.loc[net_hv_activated_bus, 'vm_pu'].max()


# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________

def run_powerflow(network: pandapower.auxiliary.pandapowerNet,
                  network_hv_activated_bus: list,
                  sum_max_p_mw_upperNet: tuple,
                  dict_df_sgenLoad: dict,
                  opf_status=False):
    """
    Return a list of maximum voltage on the network for each period given by the index 
    of element in 

    Initialise the parameters of the network

    Parameters:
    ----------
    network: Pandapower network
        The network to beimulation consider ;
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year) of the 
        + df_prodHT         => HT producers in the subnetwork 
        + df_prod_bt_total  => BT producers in the subnetwork
        + df_cons_total     => Load demand subnetwork 
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the bigger network (here, saint laurent 
        compared to the subnetwork civaux)
        + Of all BT energy producers => sum_max_input[0]
        + of all Load in the network => sum_max_input[1] 
    network_hv_activated_bus: list
        list of all Hv bus activated in the concerned network
    ofp_status: Boolean = False
        Wether the maximum voltage is computed after a normal or optimal power flow or both
        + Normal  =>  **pandapower.runpp(net)**,  ofp_status = False
        + Optimal =>  **pandapower.runopp(net)**, ofp_status = True
        + Both    =>  A normal power flow is run. Only when the result i.e. max_vm_pu > threshold, 
                      is the optimal power flow run.


    """

    # Creating empty list 
    list_max_vm_pu = []  # Maximum vm_pu at each period considered
    list_sgen_HT = []  # Actual HT generators power after optimal flow

    # Initiate parameters from inputs
    df_prodHT = dict_df_sgenLoad['df_prodHT']

    # Initialise the network and get the maximum value for each period
    for curr_period in tqdm(df_prodHT.index):

        if opf_status:  # Run optimal power flow
            max_vm_pu, sgen_pw_HT = run_powerflow_at(network, curr_period,
                                                     network_hv_activated_bus,
                                                     sum_max_p_mw_upperNet,
                                                     dict_df_sgenLoad, opf_status)
            list_max_vm_pu.append(max_vm_pu)
            list_sgen_HT.append(sgen_pw_HT)

        else:  # Run simple power flow
            list_max_vm_pu.append(run_powerflow_at(network, curr_period,
                                                   network_hv_activated_bus,
                                                   sum_max_p_mw_upperNet,
                                                   dict_df_sgenLoad, opf_status))

    # Return depends on ofp_status
    if opf_status:
        return list_max_vm_pu, list_sgen_HT
    else:
        return list_max_vm_pu





# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________

def run_powerflow_at(network: pandapower.auxiliary.pandapowerNet,
                     curr_period: pandas._libs.tslibs.period,
                     network_hv_activated_bus: list,
                     sum_max_p_mw_upperNet: tuple,
                     dict_df_sgenLoad: dict,
                     auth_vm_pu_max: float = default_auth_vm_pu_max,
                     opf_status=False, 
                     pred_model=None):
    """
    Return the maximum voltage on the network for the period 

    Initialise the parameters of the network

    Parameters:
    ----------
    network: Pandapower network
        The network to beimulation consider ;
    curr_period: Panda period
        The current period to investigate;
    dict_df_sgenLoad: dict 
        Dictionary containing data (as dataframe i.e indexed by each period of 
        the considered year) of the 
        + df_prodHT         => HT producers in the subnetwork 
        + df_prod_bt_total  => BT producers in the subnetwork
        + df_cons_total     => Load demand subnetwork 
    sum_max_p_mw_upperNet: tuple
        Sum of maximum power seen from the bigger network (here, saint laurent 
        compared to the subnetwork civaux)
        + Of all BT energy producers => sum_max_input[0]
        + of all Load in the network => sum_max_input[1] 
    network_hv_activated_bus: list
        list of all Hv bus activated in the concerned network
    auth_vm_mu_max: Threshold of maximum voltage allowed on the network. Only used when the last
        input `ofp_status` is 'Both';
    ofp_status: Boolean = False
        Wether the maximum voltage is computed after a normal or optimal power flow or both
        + Normal  =>  **pandapower.runpp(net)**,  ofp_status = False
        + Optimal =>  **pandapower.runopp(net)**, ofp_status = True
        + Both    =>  A normal power flow is run. Only when the result i.e. max_vm_pu > threshold, 
                      is the optimal power flow run.
    pred_model: String
        Which kind of prediction model to use for the all the variables to predict at current period
        + Pers  =>  Persistence model i.e. val(k)= val(k-1)


    
    """

    # Check variables congruence 
    checker.check_var_concordance(opf_status, pred_model) 
    
    # -- GT1
    if pred_model == 'Pers': # if the the prediction model is the persistance,
        curr_period = curr_period-1
        
        
    # Initialize the network. See the corresponding function for more explanation
    initLowerNet_at(network, curr_period, sum_max_p_mw_upperNet, dict_df_sgenLoad)

    # Get the maximum voltage magnitude of all activated bus to a list. See the 
    #                               corresponding function for more explanation
    if opf_status == True:  # Run optimal power flow *********************************************

        # get maximum value of vm_pu for the current period after optimal power flow
        cur_max_vm_pu = max_vm_pu_at(network, curr_period,
                                     network_hv_activated_bus,
                                     dict_df_sgenLoad, opf_status)

        # Get the value of HT producer after optimal flow. 
        sgen_pw_HT = list(network.res_sgen[network.sgen.name.notna()].p_mw)

        return cur_max_vm_pu, sgen_pw_HT

    elif opf_status == 'Both':  # Run normal and depending on the situation, also optimal power flow  *******
        # run normal power flow first 
        cur_max_vm_pu = max_vm_pu_at(network, curr_period, network_hv_activated_bus,
                                     dict_df_sgenLoad, False)
        max_vm_pu_pf = cur_max_vm_pu # Save the maximum voltage given by the power flow 
                                     # before optimizing
        # If the maximum voltage on buses is above the authorized threshold, run opf
        if cur_max_vm_pu > auth_vm_pu_max:
            cur_max_vm_pu = max_vm_pu_at(network, curr_period,
                                         network_hv_activated_bus,
                                         dict_df_sgenLoad, True)

        # Get the value of HT producer after optimal flow. 
        sgen_pw_HT = list(network.res_sgen[network.sgen.name.notna()].p_mw)

        # Depending on the prediction model parameter the return is different----------
        # For <pred_model = 'Pers'> given that at GT1 the <curr_period = curr_period-1> 
        #     one must reset curr_period to its initial value using <curr_period+1> 
        #     before ruturning the results
        if pred_model == 'Pers': return [max_vm_pu_pf, cur_max_vm_pu], sgen_pw_HT, curr_period+1
        else: return [max_vm_pu_pf, cur_max_vm_pu], sgen_pw_HT, curr_period

    elif opf_status == False :  # Run normal power flow  *******************************************************
        return max_vm_pu_at(network, curr_period, network_hv_activated_bus,
                            dict_df_sgenLoad, opf_status)
    
    else : 
        raise ValueError('<opf_status> must be either of [True, False, ''Both'']' )      





# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________
        
            
# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________            
            
def improve_persinstence(per_extracted_res_df: pandas.core.frame.DataFrame, 
                         prodHT_df: pandas.core.frame.DataFrame,
                         auth_vm_mu_max: float, 
                         h_start_end = ['11:00','14:00']):
# Implement : * Inject all the production as long as max_vm_pu_pf < vm_mu_max, i.e. 
# no voltage rise is detected 
    """
    Improve the results given by the persistence model. If a voltage rise is not predicted by 
    the persistence model at a certain period, the controllable sgens is allowed to inject all 
    its power into the grid. Otherwise the energy producer can inject at most the predicted power 
    by the persistence model. 


    Parameters
    ----------
    per_extracted_res_df: dataframe
        Result given by the persistence model. 
        Output of <<myFunction.extract_par_results(par_results_pers, *args).
    df_prodHT: Dataframe
        Dataframe containing data of all the HT producers in the network
    auth_vm_mu_max: 
        Threshold of maximum voltage allowed on the network. 


    Output:
    -------
    per_improved_res: dataframe
    Output the improve persistence model improve by the previoulsy described strategy
    """
    
    # Copy the results of the persistence model 
    per_improved_res_out = per_extracted_res_df.copy(deep=True)
    per_improved_res = per_extracted_res_df.copy(deep=True)
    
    # Convert index from period to timestamp
    per_improved_res.index = per_improved_res.index.to_timestamp()
    

    # Extract the part of the df one want to work with i.e. the period before h_start
    # and after h_end as -------'11:00'     '14:00'------ for the default value
    # the period defined between h_start and h_end is not considered since voltage rises 
    # are known to happen in that interval 
    working_df = per_improved_res.between_time(h_start_end[1],h_start_end[0])
    
    # Extract index of instances where no voltage rise is detected ignoring the last one 
    # because not present in the inital df df_prodHT
    var_index = working_df[working_df.max_vm_pu_pf<=auth_vm_mu_max].index.to_period('10T')[:-1]
    
    # remplace the prediction from the persistence model with the actual production since
    # no voltage rise is detected at these periods
    per_improved_res_out.P0100[var_index] = prodHT_df.P0100[var_index]
    

    return per_improved_res_out, var_index
    

    
 

    
# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________          
def prediction_bloc(rnn_model, fitting_scaler, history, scaler_features=None ):
    """
    Prediction bloc: Predict the values () of the next period based on the RNN (LSTM). 


    Prameters: 
    ----------
    rnn_model: Recurent neural network; 
        The model that will be used to predict the value at the next period. 
    fitting_scaler: Scaler
        Scaler parameters that are used to transform the training data set 
        fed to the RNN. 
    history: Non scaled history of the Electrical network: 
    scaler_features : Scaler to use for prediction when the number of 
        variables to predict is different from the number of features



    Output: List 
    (1): Prediction of the interest variable 
    (2): Period for wich the prediction is done

    
    """
    
    history_last_ind = history.index[-1]              # Get index of the last period of history
    in_shape = tuple([1]) + rnn_model.input_shape[1:] # define input shape for the RNN
    
    # Scaled the input  based on the fitting scaler 
    scaled_history = fitting_scaler.transform(history).reshape(in_shape)  
    
    pred = rnn_model.predict(scaled_history, verbose=False)  # prediction
    
    # inverse transform the prediction
    if scaler_features is None:   pred_inv_trans = fitting_scaler.inverse_transform(pred)  
    else : pred_inv_trans = scaler_features.inverse_transform(pred) 
    
    # Return the prediction of the RNN and the time period associated ()
    return pred_inv_trans, history_last_ind+1






# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________          
def predictionBin_bloc(rnn_model, fitting_scaler, history, sig_thresh=0.5):
    """
    Prediction bloc: Predict the values () of the next period based on the RNN (LSTM). 


    Prameters: 
    ----------
    rnn_model: Recurent neural network; 
        The model that will be used to predict the value at the next period. 
    fitting_scaler: Scaler
        Scaler parameters that are used to transform the training data set 
        fed to the RNN. 
    history: Non scaled history of the Electrical network:
    sig_thresh: Sigmoid threshold



    Output: List 
    (1): Prediction of the interest variable 
    (2): Period for wich the prediction is done

    
    """
    
    history_last_ind = history.index[-1]              # Get index of the last period of history
    in_shape = tuple([1]) + rnn_model.input_shape[1:] # define input shape for the RNN
    
    # Scaled the input  based on the fitting scaler 
    scaled_history = fitting_scaler.transform(history).reshape(in_shape)  
    
    pred = rnn_model.predict(scaled_history, verbose=False)  # prediction
    
    
    pred_bin = (pred>sig_thresh).astype(int) # convert prediction into a binary variablethe prediction
    
    # Return the prediction of the RNN and the time period associated ()
    return pred_bin[0][0], history_last_ind+1






# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________          
def robustPred(model_Vrise_dict, hvProd_noControl, P0100_opt_model1, 
               auth_vm_mu_max:float = default_auth_vm_pu_max, 
               n_models = None ):
    """
    Define Robust prediction bloc: 


    Prameters:
    ----------
    model_Vrise_dict: Dict
        Dictionary of the voltage rise for each model 
    hvProd_noControl : pandas dataframe
        Values of the controlled Generator P0100 when no controled is applied
    P0100_opt_model1 : pandas Dataframe. Partial output of function <<extract_par_results>>
        Optimal value of P0100 at the end of bloc PF/OPF of model1. This is the 
        command value to send to the said producer when the robustPred 
        predicts a voltage rise above the threshold vm_mu_max.
    auth_vm_mu_max: Threshold of maximum voltage on the network
    n_models: Int or string
        Int: Number of models which must agree on voltage rise above threshold before
        a command is set to P0100
        ** 1: At Least one of the models
        ** 2: At least two of the models
        ** 3: All three models
        String: 
            Name of the Model which voltage rise above threshold prediction is considered
                'Modelx' where x in {1,2,3}


    Output: 
    ---------
    new_p0100_df: panda dataframe
        y_optimal after combined model

    """
    

    # Extract model voltage rise from Input dictionary 
    model1_Vrise, model2_Vrise, model3_Vrise = (model_Vrise_dict['Model1'], 
                                                model_Vrise_dict['Model2'], 
                                                model_Vrise_dict['Model3'])
    
    mask_per2work = model1_Vrise.index # Get index of the considered period
    vect_int = np.vectorize(int)       # vectorized version of int

    # Create an empty dataframe i.e. binary threshold 
    bin_thresh_df = pd.DataFrame(index=mask_per2work)

    # add the binary output of three models to the created df
    bin_thresh_df[['Model3']] = model3_Vrise.values
    bin_thresh_df[['Model2']] = vect_int(model2_Vrise>auth_vm_mu_max)
    bin_thresh_df[['Model1']] = vect_int(model1_Vrise>auth_vm_mu_max)

    # Combined_output of all models
    bin_thresh_df[['Model_All']] = np.array(bin_thresh_df.sum(axis=1)).reshape((-1,1))

    
    # Create a new dataframe for the controlled SGEN based on its real values 
    new_p0100_df = hvProd_noControl.loc[mask_per2work, ['P0100']]

    
    if type(n_models) is str :# If n_model is a string
        if n_models in model_Vrise_dict.keys(): # Check if the sting input is in the model Dict
            vrise_true_mask = bin_thresh_df[n_models] == 1 # Create the mask using only the 
                       # period where the concerned model predict there is an voltage rise
        else: raise ValueError('Since <n_models> is a string it must be be either of', list(model_Vrise_dict.keys()))      
            
    elif type(n_models) is int: # If n_model is int 
        if n_models <= 3:
            # Create mask of instants where at least n models agrees on voltage rise above threshold 
            vrise_true_mask = bin_thresh_df.Model_All>= n_models 
        else: raise ValueError('Since <n_models> is an int it must be defined such that 0 < n_models <= 3 ')
    
    else: raise ValueError('<n_models> is the wrong type. Must either be an int or a string')

    # Use vrise_true_mask to insert predicted values given by model1 at the concerned instants 
    new_p0100_df[vrise_true_mask] = P0100_opt_model1.loc[mask_per2work].loc[vrise_true_mask, ['P0100']]
    
    return new_p0100_df, bin_thresh_df









def _upscale_HvLv_prod(prod_hv2upscale_df, prod_lv2upscale_df, 
                      ctrl_hvProd_max, upNet_sum_max_lvProd,
                      cur_hvProd_max:int=0, 
                      params_coef_add_bt: tuple=(None,None)
                      ):
    """
    Upscale  both the controled Higher voltage(HV) producer (P0100) in the lower network (civeaux) 
    and the total Lower voltage (LV) production. Check the parameter 'coef_add_bt_dist' to choose how 
    the upscaling is done on the LV production.
    THis mean the BT producer on the lower network receives only a fraction of the added BT production. 
    See function upscale_HvLv_prod() for the version of the function where the BT producer receive
    all the Added BT prod


    Parameters: 
    -----------
    prod_hv2upscale_df: pd.dataframe
        dataframe of the HV prod to upscale i.e. P0100
    prod_lv2upscale_df: pd.dataframe 
        dataframe of the total LV producers output (That must be increased) i.e. Prod_BT
    ctrl_hvProd_max: float
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    upNet_sum_max_lvProd: float
        Sum of maximum output of all lower voltage (LV) producers (MW) in the upper Network
        TODO: Get upNet_sum_max_lvProd from oriClass.InitNetwork() instance 
    cur_hvProd_max: Int (default=0) 
        Current value of maximum output Power of the HV producer (MW)
    params_coef_add_bt:tuple
        Parameters associated with how the upscaling of the total LV production is done. See doc
        # oriClass.InitNetworks(*args) for more information
        (1): coef_add_bt 
        (2): coef_add_bt_dist
        
    Output:
    -------
        The upscaled version of the HV producer and LV

    """

        
    ###  ----------- Upscale P0100 the controlable HT producer  ---------------  ###
    prodHT_P0100_1mw_df = prod_hv2upscale_df/ctrl_hvProd_max      # Rescale the total HT production for 1mw
    upscaled_prodHT_P0100_df = cur_hvProd_max*prodHT_P0100_1mw_df # Redefine the HT production based on the rescale


    # Get the parameters coef_add_bt and coef_add_bt_dist 
    coef_add_bt = params_coef_add_bt[0]
    coef_add_bt_dist = params_coef_add_bt[1]
    
    
    ###  ----------- upscale BT production ---------------  ###
    # Only the upscaling on the upper Network is done here. For the other cases, 
    # i.e. 'lowNet' and 'lowNet_rand', see oriClass.InitNetworks(*args)
    upscaled_prod_bt_total_df = prod_lv2upscale_df
    if coef_add_bt_dist == 'uppNet' :
    
        # Rescale the total BT production for 1mw
        prod_bt_total_1mw_df = prod_lv2upscale_df/upNet_sum_max_lvProd
        
        #Increase sum of output of BT  the coeef_add_bt if coef_add_bt_dist
        upscaled_prod_bt_total_df = prod_lv2upscale_df + coef_add_bt*prod_bt_total_1mw_df  

        
    upscaled_prod_bt_total_df.columns = ['Prod_BT']
    
    return upscaled_prodHT_P0100_df, upscaled_prod_bt_total_df





def createDict_prodHtBt_Load(df_pred_in, 
                             networks_in: oriClass.InitNetworks,
                             cur_hvProd_max:float,
                             ctrl_hvProd_max:float = default_ctrl_hvProd_max 
                            ) -> dict :
    """
    Create a dictionary that will be send to the local space of the parallele engines.
    
    Parameters: 
    -----------
    df_pred_in: pd.dataframe
        Dataframe (Predicted values) of Total lower voltage producer, load demand and all 
        the Hihger voltage producer in lower level network.
    network_in : oriClass.InitNetworks
        Networks initialized. An instance of oriClass.InitNetworks, especially the 
        output of the function setNetwork_params()
    cur_hvProd_max: float
        Current value of maximum output Power of the controlled HV producer (MW)
    ctrl_hvProd_max: (float)
        Maximum fixed output of the Controlled Higher voltage producer (MW)

    Outputs:
    --------
    dict_df_sgenLoad: dict of dataframe
        keys1: 'df_prodHT' 
            Dataframe containing the upscaled (based on cur_hvProd_max) pv power of the 
            Hihger voltage  producers in lower level network.
        keys2: 'df_prod_bt_total' 
            Dataframe of the upscaled (based on coef_add_bt) total pv power of all lower
            voltage producer in the lower network
        keys3: 'df_cons_total'
            Dataframe of the total load demand (consumption) in the lower level network
            

    """
    # Instancuate parameters
    upNet_sum_max_lvProd = networks_in.get_upperNet_sum_max_lvProdLoad()[0]
    params_coef_add_bt = networks_in.get_params_coef_add_bt()
    ctrl_hvProd_name = networks_in.get_ctrl_hvProdName()
    
    # Check if coef_add_bt_dist is authorized
    checker.check_coef_add_bt_dist(params_coef_add_bt[1])
    
    # Check wether the input dataframe columns are in the expected order
    checker.check_networkDataDf_columnsOrder(df_pred_in)
    
    df_pred = df_pred_in.copy(deep=True) # Create a copy of the input dataframe
    
    # If the last 2 digits of an elm of df_pred.columns is decimal, therefore the colums is 
    # that of a HV producer
    hvProd_columns = [elm for elm in df_pred.columns if elm[-4:].isdecimal()]   
    df_prodHT = df_pred[hvProd_columns]
    
    
    # Upscale HV production and the LV µ% production
    df_prodHT[[ctrl_hvProd_name]], df_prod_bt_total = _upscale_HvLv_prod(df_prodHT[[ctrl_hvProd_name]],
                                                                         df_pred[['Prod_BT']],
                                                                         ctrl_hvProd_max, upNet_sum_max_lvProd,
                                                                         cur_hvProd_max, params_coef_add_bt )

    # Define consumption df
    # TODO : Code a function to check the oreder of the input dataframe
    df_cons_total = df_pred.iloc[:,[0]]

    # Define a dict 
    dict_df_sgenLoad = dict({'df_prodHT':df_prodHT, 
                             'df_prod_bt_total':df_prod_bt_total, 
                             'df_cons_total':df_cons_total } )
    
    return dict_df_sgenLoad



def robustControl(df_out_block_pf_opf:pandas.core.frame.DataFrame ,
                  df_hvProd_noControl: pandas.core.frame.DataFrame,
                  cur_hvProd_max: float, 
                  ctrl_hvProd_max: int , 
                  vm_mu_max: float):
    
    """
    Implement Robust control by letting the controlled Hv Producer inject all its production 
    when no voltage rise above the predefined threshold is detected. Replacement is done in
    place i.e. in the df_out_block_pf_opf 
    
    Parameters
    -----------
    df_out_block_pf_opf: (Dataframe)
        Output of the block pf opf
    df_hvProd_noControl : Dataframe
        Dataframe of P0100 with no control
    cur_hvProd_max: float
        Current Value of maximum output Power of the HV producer (MW)
    ctrl_hvProd_max: (int)
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    vm_mu_max (Float)
        Threshold of voltage rise authorised
    """
    
    # Basically, we replace the value of the controled HvProd by its own 
    # value with No control when no voltage rise above the defined threshold is detected.
    
    # create new period index mask spaning from 08Am to 6PM
    per_index2 = df_out_block_pf_opf.index.to_timestamp().to_series().between_time('07:10',
                                                                         '18:50').index.to_period('10T')
    ctrld_hvProdName = df_out_block_pf_opf.columns[0]
    
    # Create a new df for hvProd
    hvProd_robust_df = pd.DataFrame(index=per_index2, columns=['hvProd_robust'])
    
     # Get into the dataframe data of hvProdWhen there is no control
    hvProd_robust_df.loc[per_index2,['hvProd_robust'] ] = (df_hvProd_noControl.loc[per_index2].P0100
                                                                  *cur_hvProd_max/ctrl_hvProd_max)
    
    # Get a mask for the periods where a voltage rise above the threshold is predicted 
    mask_vrise_per = df_out_block_pf_opf.loc[per_index2, 'max_vm_pu_pf']>vm_mu_max
       
    # Replace the values of periods given by the mask by the value of hvProd given by the persistence model
    hvProd_robust_df[mask_vrise_per] = df_out_block_pf_opf.loc[per_index2].loc[mask_vrise_per,[ctrld_hvProdName]]

    # Replace the values of hvProdin df_out_block_pf_opf
    df_out_block_pf_opf.loc[per_index2, [ctrld_hvProdName]] = hvProd_robust_df.loc[per_index2, 'hvProd_robust']  


    
    
    
def block_prod(df_out_block_pf_opf: pandas.core.frame.DataFrame, 
               df_hvProd_noControl: pandas.core.frame.DataFrame, 
               cur_hvProd_max: float, 
               ctrl_hvProd_max: int, 
               starting_index: int = 0 ):
    """
    Implement bloc prod i.e. make sure the controlled HV producer can't inject into the lower network 
    more than its actual production. Modify in place the input dataframe df_out_block_pf_opf
    
    Parameters:
    -----------
    df_out_block_pf_opf: (Dataframe)
        Output of the block pf opf that has been send to the robust persitence
    df_hvProd_noControl : Dataframe
        Dataframe of hvProdwith no control
    cur_hvProd_max: int
        Value of maximum output Power of the HV producer (MW)
    ctrl_hvProd_max: (int)
        Maximum fixed output of the Controlled Higher voltage producer (MW)
    starting_index: Starting index (optional), default to zero
        Important to use this starting index and set it to the lenght of a day in the case 
        of the RNN. This is due to the fact that the prediction needs a whole day of data 
        to be produced. Especially the first prediction must be that of the the first index
        of the second day of the testing set since the whole first day (of the testing set)
        data is used .    
    
    """
    ctrld_hvProdName = df_out_block_pf_opf.columns[0]
    
    
    per_index2 = df_out_block_pf_opf.index.to_timestamp().to_series().between_time('07:10',
                                                                                   '18:50').index.to_period('10T')
    
    df_hvProd_noControl_upscaled = (df_hvProd_noControl.loc[per_index2[starting_index:],ctrld_hvProdName]
                                    *cur_hvProd_max/ctrl_hvProd_max)
    
    df_P0100_controled = df_out_block_pf_opf.loc[per_index2[starting_index:], ctrld_hvProdName]
    
    df_out_block_pf_opf.loc[per_index2[starting_index:], [ctrld_hvProdName]] = (np.minimum(df_hvProd_noControl_upscaled,
                                                                                  df_P0100_controled)
                                                                      )
        

        
        
        
def setNetwork_params(upperNet_file:str, 
                      lowerNet_file:str, 
                      ctrld_hvProdName:str,
                      params_coef_add_bt:tuple=(None, None), 
                      params_vm_mu:tuple=(default_auth_vm_pu_max, default_auth_vm_pu_min)
                     ) -> oriClass.InitNetworks :
    
    """
    Load both the lower (network used for opimization) and upper network, after which a configuration 
    of the main  parameters to use for the simulations are done. 
    Namely:
 
    Parameters: 
    -----------
    upperNet_file:
        The upper Network file, with the approporiate extenxion (Must be present in the network_folder)
        Egg: 'ST LAURENT.p'
    lowerNet_file :
        The lower Network file, with the approporiate extenxion (Must be present in the network_folder)
        Egg:'CIVAUX.p'
    ctrld_hvProdName :
        Name of the controlled HV producer in the Lower Network. 
        Egg: 'P0100'
    params_coef_add_bt: 
        (1) coef_add_bt: float
            Value of the added output power for all the LV producers (MW) in the lower Network
        (2) coef_add_bt_dist: str
            How coef_add_bt is shared among the LV producers. 
            Three choices are possible
            (0): None (default) ==> No upscaling is done
            (1): 'uppNet' ==> coef_add_bt is added to the Sum of maximum output of all lower 
                 voltage (LV) producers (MW) in the upper Network. In consequence, the LV producers 
                 on the lower network receive only a fraction of coef_add_bt.
            (2): 'lowNet'==> coef_add_bt is added to the Sum of maximum output of all LV 
                 producers (MW) in the lower Network. In consequence, coef_add_bt is shared 
                 proportionnaly among all the LV producers on the lower network. 
            (3) 'lowNet_rand' ==> coef_add_bt is shared proportionnaly among a randomly selected 
                 set of the LV producers on the lower Network. The randomly selected set consist of 
                 half of all LV producers on the on the lower Network
    params_vm_mu: 
        (1) vm_mu_max: float
            Maximum authorised voltage rise on the Lower network
        (2) vm_mu_min: float
            Minimum authorised voltage rise on the Lower network

    Output:
    -------
    
    """
    
    # Extracts parameters
    coef_add_bt, coef_add_bt_dist = params_coef_add_bt
    vm_mu_max, vm_mu_min = params_vm_mu
    
    #Load lower and upper Network
    lowerNet=pp.from_pickle(f'{network_folder+lowerNet_file}')
    upperNet=pp.from_pickle(f'{network_folder+upperNet_file}')
    
    networks = oriClass.InitNetworks(upperNet, lowerNet, coef_add_bt, coef_add_bt_dist )  # Initialize networks
    
    networks.init_controled_hvProd(ctrld_hvProdName) # Initialize the controlled HVProd in the lowerNetwork
    
    lowerNet_hv_bus_df = networks.get_lowerNet_hv_bus_df(bus_voltage=20.6)  # Extract HV bus in the network
    
    uppNet_sum_max_lvProdLoad = networks.get_upperNet_sum_max_lvProdLoad() # To use later in functions 
    
    # Extract the actives HV buses in the lower Network
    lowerNet_hv_activated_bus = networks.get_lowerNet_hvActivatedBuses(lowerNet_hv_bus_df.index)
    
    # Set the voltage rise threshold on the lower Network
    networks.lowerNet_set_vrise_threshold(lowerNet_hv_activated_bus,vm_mu_min, vm_mu_max )

    # Add negative cost to usability of controlled Sgen so its usage can be maximised while 
    # respecting the constraints on the network
    ctrld_hvProd_index = lowerNet.sgen[lowerNet.sgen.name==ctrld_hvProdName].index
    cost_sgen_p0100 = pp.create_poly_cost(lowerNet, ctrld_hvProd_index,'sgen', cp1_eur_per_mw=-1)
    
    return networks