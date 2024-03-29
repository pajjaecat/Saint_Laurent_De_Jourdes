# List of function used in my code

import pandapower, pandas, numpy
from tqdm import tqdm  # Profiling

pd = pandas
np = numpy 
pp = pandapower
###############################          Variables       #########################################

 


# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________

def initialize_network_at(network: pandapower.auxiliary.pandapowerNet,
                          curr_period: pandas._libs.tslibs.period,
                          sum_max_main_network: tuple,
                          dict_df_sgenLoad: dict):
    """
Return a fixed float;

Initialise the parameters of the network at the current period

Parameters:
----------
network: Pandapower network
    The small network concerned ;
curr_period: Pandas period
    The current period to investigate;
sum_max_main_network: tuple
    Sum of maximum power seen from the bigger network (here, saint laurent 
    compared to the subnetwork civaux)
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
    sum_max_p_mw_StLaurent_prodBT = sum_max_main_network[0]
    sum_max_p_mw_StLaurent_load = sum_max_main_network[1]

    df_prodHT = dict_df_sgenLoad['df_prodHT']
    df_prod_bt_total = dict_df_sgenLoad['df_prod_bt_total']
    df_cons_total = dict_df_sgenLoad['df_cons_total']

    # Initalise HT producers 
    network.sgen.p_mw[network.sgen.name.notna()] = df_prodHT.loc[curr_period].values

    # Initialize Bt producers
    network.sgen.p_mw[network.sgen.name.isna()] = (network.sgen.
                                                   max_p_mw[network.sgen.name.isna()] *
                                                   df_prod_bt_total.loc[curr_period].
                                                   values / sum_max_p_mw_StLaurent_prodBT)
    # Initialize Loads
    network.load.p_mw = (network.load.max_p_mw * df_cons_total.loc[curr_period].
                         values / sum_max_p_mw_StLaurent_load)

    # Work with julia Power model since the load is zero
    # network.load.p_mw = (network.load.max_p_mw*df_cons_total.loc[curr_period].
    # values*0/sum_max_p_mw_StLaurent_load)


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
    + df_prodHT         => HV producers in the lower network
    + df_prod_bt_total  => Production of all 
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

    # Return the maximum voltage over all the busses in the network for the current instant
    return network.res_bus.loc[net_hv_activated_bus, 'vm_pu'].max()


# ___________________________________________________________________________________________________________________________________
# ----------------------------------------------------------------------------------------------------------------------------------
# ___________________________________________________________________________________________________________________________________

def run_powerflow(network: pandapower.auxiliary.pandapowerNet,
                  network_hv_activated_bus: list,
                  sum_max_main_network: tuple,
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
sum_max_main_network: tuple
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
                                                     sum_max_main_network,
                                                     dict_df_sgenLoad, opf_status)
            list_max_vm_pu.append(max_vm_pu)
            list_sgen_HT.append(sgen_pw_HT)

        else:  # Run simple power flow
            list_max_vm_pu.append(run_powerflow_at(network, curr_period,
                                                   network_hv_activated_bus,
                                                   sum_max_main_network,
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
                     sum_max_main_network: tuple,
                     dict_df_sgenLoad: dict,
                     auth_vm_pu_max=1.02,
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
sum_max_main_network: tuple
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
    check_var_concordance(opf_status, pred_model) 
    
    
    # -- GT1
    if pred_model == 'Pers': # if the the prediction model is the persistance,
        curr_period = curr_period-1
        
        
    # Initialize the network. See the corresponding function for more explanation
    initialize_network_at(network, curr_period,
                          sum_max_main_network, dict_df_sgenLoad)

    # Get the maximum voltage magnitude of all activated bus to a list. See the 
    #                               corresponding function for more explanation
    if opf_status == True:  # Run optimal power flow *********************************************

        # get maximum value of vm_pu for the current period after optimal power flow
        cur_max_vm_pu = max_vm_pu_at(network, curr_period,
                                     network_hv_activated_bus,
                                     dict_df_sgenLoad, opf_status)

        # Get the value of power on Hv producer after optimal flow. 
        # HT producer results are in res_sgen.p_mw[21:]
        # sgen_pw_HT = list(network.res_sgen[network.sgen.name.notna()].p_mw)
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
        # HT producer results are in res_sgen.p_mw[21:]
        sgen_pw_HT = list(network.res_sgen[network.sgen.name.notna()].p_mw)

        # Depending on the prediction model parameter the return is different----------
        # For <pred_model = 'Pers'> given that at GT1 the <curr_period = curr_period-1> 
        #     one must reset curr_period to its initial value using <curr_period+1> 
        #     before ruturning the results
        if pred_model == 'Pers': return [max_vm_pu_pf, cur_max_vm_pu], sgen_pw_HT, curr_period+1
        else: return [max_vm_pu_pf, cur_max_vm_pu], sgen_pw_HT, curr_period

    elif opf_status == False :  # Run normal power flow  *******************************************************
        return max_vm_pu_at(network, curr_period, network_hv_activated_bus,
                            dict_df_sgenLoad, False)
    
    else : 
        raise ValueError('<opf_status> must be either of [True, False, ''Both'']' )      
        
        

  
            
            
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

def prediction_bloc(rnn_model, fitting_scaler, history, ):
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
scaler_features: 



Output: List 
(1): Prediction of the interest variable 
(2): Period for wich the prediction is done

    
    """
    
    history_last_ind = history.index[-1]              # Get index of the last period of history
    in_shape = tuple([1]) + rnn_model.input_shape[1:] # define input shape for the RNN
    
    # Scaled the input  based on the fitting scaler 
    scaled_history = fitting_scaler.transform(history).reshape(in_shape)  
    
    pred = rnn_model.predict(scaled_history, verbose=False)  # prediction
    pred_inv_trans = fitting_scaler.inverse_transform(pred)  # inversse transform the prediction
    
    # Return the prediction of the RNN and the time period associated ()
    return pred_inv_trans, history_last_ind+1





def check_var_concordance(opf_status=False, pred_model=None):
    """
Check the congruence between the optimal power flow variable and the type of prediction model.

Parameters:
----------
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
    
    pred_model_values = ['Pers']
    
    # If the prediction model <pred_mod> is defined,  make sure that the <ofp_status> ='Both'
    if(pred_model is not None):
        if pred_model not in pred_model_values: # check if the pred_model value is an authorised
            raise ValueError('<pred_mod> must be either of', pred_model_values )      
            
        if opf_status != 'Both': # 
            raise ValueError('Given that <pred_mod> is defined, <ofp_status>  must be  set to <\'Both\'> ')

    