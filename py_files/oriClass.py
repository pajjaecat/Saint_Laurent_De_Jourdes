import pandapower, pandas, numpy, ipyparallel, os, joblib, seaborn, importlib
import matplotlib.pyplot as plt
import checker
pd = pandas
np = numpy 
pp = pandapower
ipp = ipyparallel
sbn = seaborn


# To remove later 
importlib.reload(checker)




# Variables 
Δt = 1/6


##############################         Class          #########################################
class CreateParEngines:
    
    def __init__(self, n_engines:int=0):
        """
        Create n parallel engines
        
        """
        cluster = ipp.Cluster(n=n_engines)     # Create cluster
        cluster.start_cluster_sync()            # Start cluster
        self.rc = cluster.connect_client_sync() # connect client
        self.rc.wait_for_engines(n_engines)     # Wait for engines to be available for all clients
        self.dview = self.rc[:]                 # Create a direct view of all engine or clients that would
                                                # be used to command each client

    def sendVar_to_localSpace(self, 
                              run_periodIndex: pandas.core.indexes.period.PeriodIndex, 
                              opf_status, 
                              dict_df_sgenLoad:dict, 
                              parameters_dict: dict, 
                              nb_exec: int ):
        """
        Send variables to the local space of each parallel engine. 
        
        
        Parameters
        -----------
        run_periodIndex:
            Total number of periods to run simulation for. The number of period each engine 
            will work on is therfore given by len(run_periodIndex)/n where n is the number of 
            engines
        opf_status : Optimal power flow status. 
            2 values are possible
            False (Boolean) ==> Run power flow
            'Both' (str)    ==> Run optimal power flow or power flow depending on the situation
        dict_df_sgenLoad: (dict)
        
        parameters_dict:
        
        nb_exec: execution number in the current loop
            0 ==>> Clear the localspace of all engines
        
        """
        # Raise Error if the OPF type is not well defined 
        if opf_status not in ['Both', False]:
             raise valueError ('<opf_status> must be either of [False, ''Both'']')
        
        self.run_periodIndex = run_periodIndex;
        self.pred_model = parameters_dict['pred_model']
        self.opf_status = opf_status;
        
        if nb_exec == 0 : 
            # Clear the localspace of all engines
            self.dview.clear() 

            # # Import following modules on the local space of clients or engines
            with self.rc[:].sync_imports():
                import numpy, pandapower,pandas, par_oriFunctions

        # Share the total number of period in df_prodHT.index among all the created engines
        self.dview.scatter('period_part', run_periodIndex)

        # Add all variables in the parameters_dict  to  the local space of each client
        self.dview.push(parameters_dict) 

        # Send following Variables to local space of parallel engines
        self.dview['opf_status'] = opf_status  
        self.dview['dict_df_sgenLoad'] = dict_df_sgenLoad
        
        
        
    def gather_results(self, par_result_name:str):
        """
        Gather the results of parallel computing and output a unique  
        variable
        
        Parameters
        ----------
        par_result_name: (string)
            Name use to call the parallel running in block pf_opf
        
        """
        # Che9k if the given argument is a string
        if type(par_result_name) is not str: 
            raise TypeError('gather_results() can only handle string as argument')
        
        self.gathered_results = self.dview.gather(par_result_name)
        
        return self.gathered_results
    
    
    
    def get_results_asDf(self):
        """
        Extract and save the result of the parallel computation in a dataframe that is output

        Output:
        ---------
        Dataframe as:
            max_vm_pu : Maximum voltage recorded over all the bus at the instant given 
            by the df.index
            Other columns : THe injected power of the respective HT producers.


        """

        # Get df_prodHT colums name [] from one of the engines 
        df_prodHT_colName = self.dview['dict_df_sgenLoad'][-1]['df_prodHT'].columns
        
        # Get all the elements from the parallel result in a list
        # elm[0]   : Maximum voltage on all the line 
        # elm[1][0]: Power injected into the network by the first HT producer 
        # ...
        # elm[1][n]: Power injected into the network by the last HT producer i.e. P0100 
        # elm[2]   : Period index associated to all the previous output variable

        # elm[0] can either be a list of [max_vm_pu_pf : max voltage  before opf
        #                                 max_vm_pu : maximum voltage after opf] 
        # or a single float which is  max_vm_pu : maximum voltage after opf. 
        # See the function run_powerflow_at (*args, ofp_status='both', pred_model= 'Pers')
        
        parallel_result = self.gathered_results
        
        if type(parallel_result[0][0]) is list: 
            sep_list = [(*elm[0], *elm[1], elm[2]) for elm in parallel_result]
            # Create a colums using 'vm_pu_max' and add the HT producers name
            colls = ['max_vm_pu_pf', 'max_vm_pu'] + df_prodHT_colName.to_list()
        else:
            sep_list = [(elm[0], *elm[1], elm[2]) for elm in parallel_result]
            # Create a colums using 'vm_pu_max' and add the HT producers name
            colls = ['max_vm_pu'] + df_prodHT_colName.to_list()


        data_input = np.array(np.array(sep_list)[:, :-1], dtype=float)
        index_list = np.array(sep_list)[:, -1]

        # create new  dataframe based on previous unpack data
        df = pd.DataFrame(data=data_input, index=index_list, columns=colls)

        # return the newly create dataFrame with the index sorted 
        return df.sort_index()



    def _dview(self):
        return self.dview

    def _run_periodIndex(self):
        return self.run_periodIndex
    
    def _pred_model(self):
        return self.pred_model
    
    def _opf_status(self):
        return self.opf_status
        


    def get_dview(self):
        return self._dview()

    def get_run_periodIndex(self):
        return self._run_periodIndex()
    
    def get_pred_model(self):
        return self._pred_model()
    
    def get_opf_status(self):
        return self._opf_status()

        
        
        

class InitNetworks:
    """
    Initialize both the upper and lower level  Networks.
    
    Parameters: 
    -----------
    upperNet: upper network
    lowerNet : Lower level Network 
    coef_add_bt : float (default=None) 
        Value of the added output power for all the LV producers (MW)
    coef_add_bt_dist: str(default=None)
        How the upscaling of the maximum output of all lower Voltage producers is done. 
        Three choices are possible
        (0): None ==> No upscaling is done
        (1): 'uppNet' ==> coef_add_bt is added to the Sum of maximum output of all lower 
             voltage (LV) producers (MW) in the upper Network. In consequence, the LV producers 
             on the lower network receive only a fraction of coef_add_bt.
        (2): 'lowNet'==> coef_add_bt is added to the Sum of maximum output of all LV 
             producers (MW) in the lower Network. In consequence, coef_add_bt is shared proportionnaly 
             among all the LV producers on the lower network. 
        (3) 'lowNet_rand' ==> coef_add_bt is shared proportionnaly among a randomly selected 
             set of the LV producers on the lower Network. The randomly selected set consist of 
             half of all LV producers on the on the lower Network

    """
    
    def __init__(self, 
                 upperNet:pp.auxiliary.pandapowerNet, 
                 lowerNet:pp.auxiliary.pandapowerNet, 
                 coef_add_bt:float=None, 
                 coef_add_bt_dist:str=None):
        
        self._upperNet = upperNet
        self._lowerNet  = lowerNet
        self._coef_add_bt = coef_add_bt
        self._coef_add_bt_dist = coef_add_bt_dist
        
        checker._check_network_order(self)
        checker._check_coef_add_bt_and_dist(self)
        
        lowerNet_sgenCopy = lowerNet.sgen.copy(deep=True) # Create a copy of the lower netSgens
        
        self._lowerNet_sgenLvCopy = lowerNet_sgenCopy[lowerNet_sgenCopy.name.isna()]# Extract LowerVoltage Producers
        
        # Get sum of maximum output of all the LV producers on the lower network before update
        self._lowerNet_nonUpdated_sum_max_lvProd = self._lowerNet_sgenLvCopy.max_p_mw.sum()
    
        self.lowerNet_update_max_p_mw() # Update or update the maximum output of LV producers given _coef_add_bt and 
                                         # _coef_add_bt_dist
        
        
    def lowerNet_update_max_p_mw(self):
        """ Update or upscale the maximum output of the all LV producers on the Lower net depending on
            self._coef_add_bt and self._coef_add_bt_dist
        """
        
        if self._coef_add_bt_dist == 'lowNet': # 
            updated_lowerNet_sum_max_lvProd = self._lowerNet_nonUpdated_sum_max_lvProd + self._coef_add_bt 
            self._lowerNet.sgen.max_p_mw[self._lowerNet.sgen.name.isna()] = (updated_lowerNet_sum_max_lvProd
                                    *self._lowerNet_sgenLvCopy.max_p_mw/self._lowerNet_nonUpdated_sum_max_lvProd)
        elif self._coef_add_bt_dist == 'lowNet_rand':
            #               TODO CODE                -----------------------------------
            pass
        elif self._coef_add_bt_dist == 'uppNet':
            # In this case, the The upscaling is done in oriFunctions.upscale_HvLv_prod(*args)
            pass
        
            


    def upperNet_sum_max_lvProdLoad(self):
        """ On the upper Network, Compute the sum of 
            (1) maximum output of all BT producers  
            (2) maximum of all Load demand 
        on the upper network """
        sum_max_lvProd = self._upperNet.sgen[self._upperNet.sgen.name.isna()].max_p_mw.sum()
        sum_max_load = self._upperNet.load.max_p_mw.sum()
        
        return sum_max_lvProd, sum_max_load
        
        
    def lowerNet_sum_max_lvProdLoad(self):
        """ Compute the sum of 
            (1) maximum output of all BT producers  
            (2) maximum of all Load demand 
        on the lower level network """
        sum_max_lvProd = self._lowerNet.sgen[self._lowerNet.sgen.name.isna()].max_p_mw.sum()
        sum_max_load = self._lowerNet.load.max_p_mw.sum()
        
        return sum_max_lvProd, sum_max_load
    
    
    def _lowerNet_hv_bus_df(self, bus_voltage):
        """ Extract the higher Voltage buses in the lower network: These are buses for which 
            the parameter \'bus_voltage\' equals 20.6 
        """
        return self._lowerNet.bus.groupby('vn_kv').get_group(bus_voltage)
         
        
    def _run_lowerNet(self):
        """ Run lower Network """
        pp.runpp(self._lowerNet)
        
        
    def lowerNet_set_vrise_threshold(self, lowerNet_hv_activated_bus:list, 
                                     min_vm_mu:float=0.95, 
                                     max_vm_mu:float=1.025):
        """ Set the minimum and the maximum authorised thresholds on all the activated bus on the 
            lower Network
            
            
        Parameters: 
        -----------
        lowerNet_hv_activated_bus: list
            List of all the activated buses on the lower Network
        min_vm_mu : float (default = 0.95)
            Minimum authorized voltage rise on the lower Network
        max_vm_mu : float (default = 1.025) 
            Maximum authorized voltage rise on the lower Network

        """
        self._lowerNet.bus.max_vm_pu[lowerNet_hv_activated_bus] = max_vm_mu
        self._lowerNet.bus.min_vm_pu[lowerNet_hv_activated_bus] = min_vm_mu
                       
            
            
    def init_controled_hvProd(self, controled_hvProdName:str):
        """ Initialize the controlled HV producer in the lower network
        
        Parameters:
        -----------
        controled_hvProdName:str
            Name of the controlled HV producer on the lower Network
        """
        
        # Check the existence of the controled_hvProdName
        checker._check_hvProdName_in_lowerNet(self, controled_hvProdName)
        
        # Add a controllable line to the static generators
        self._lowerNet.sgen['controllable'] = False 
        
        # Set the controled producer as a controllable sgen
        self._lowerNet.sgen['controllable'][self._lowerNet.sgen.name==controled_hvProdName] = True
        
        # Add Missing columns to be able to run an opf 
        self._lowerNet.sgen[['min_p_mw', 'min_q_mvar', 'max_q_mvar']]= 0.
        
        # Rename bus parameters because the names do not correspond to the 
        # parameters in pandapower 
        self._lowerNet.bus.rename({'max_vw_pu':'max_vm_pu'}, axis=1, inplace=True)
        
        # Delete useless parameters
        self._lowerNet.bus.drop(['max_vm', 'min_vm'], axis=1, inplace=True)    


            
    def get_lowerNet_hvActivatedBuses(self, lowerNet_hvBuses_list:list):
        """ Provide a list of all HV activated buses (vn_kv=20.6) on the lower Network
        
        Parameter: 
        ----------
        lowerNet_hv_bus_list: list
            List of all buses in the lower Network
        
        """
        self._run_lowerNet() # Run lower network

        # Extract a list of  all the activated bus on the run lower network
        lowerNet_activatedBuses_index = list(self._lowerNet.res_bus.vm_pu[
                                             self._lowerNet.res_bus.vm_pu.notna() ].index)
        
        # Extract the list of all HV activated Bus on the lower network
        self._lowerNet_hvActivatedBuses_list = [bus_index for bus_index in lowerNet_activatedBuses_index 
                                                if bus_index in lowerNet_hvBuses_list]
        return self._lowerNet_hvActivatedBuses_list 


    def _get_lowerNet_hvProducersName_df(self): 
        """ Get the name of all the Higher voltage producers on the lower net as a dataframe"""
        return self._lowerNet.sgen.name[self._lowerNet.sgen.name.notna()]
    
            
    def get_lowerNet_hvProducersName(self, return_index=False):
        """ 
        Get the name of all the Higher voltage producers on the lower net.
        
        Parameters
        -----------
        return_index: bool
            (1) False (default) 
                The indexes of the HV voltage on the sgen table are not returned
            (2) True
                The indexes of the HV voltage on the sgen table are returned with the HV producer's name
        """
        
        if return_index:
            return (list(self._get_lowerNet_hvProducersName_df().values), 
                        list(self._get_lowerNet_hvProducersName_df().index))
        else: 
            return list(self._get_lowerNet_hvProducersName_df().values)
    
    
    def get_params_coef_add_bt(self) -> tuple:
        """ Return the parameters coef_add_bt and coef_add_bt_dist """
        return self._coef_add_bt, self._coef_add_bt_dist
        
        
    def get_ctrl_hvProdName(self) -> str :
        """ Return the name of the controlled HV producer in the lower Network """
        return list(self._lowerNet.sgen[self._lowerNet.sgen.controllable].name)[0]
        
        
    ###  Define getter    ------------------------------------------------------------------
    def get_upperNet(self):
        return self._upperNet
    
    def get_lowerNet(self):
        return self._lowerNet
    
    def get_upperNet_sum_max_lvProdLoad(self):
        """ Compute the sum of 
            (1) maximum output of all BT producers  
            (2) maximum of all Load demand 
        on the upper network """
        return self.upperNet_sum_max_lvProdLoad()
    
    def get_lowerNet_sum_max_lvProdLoad(self):
        """ Compute the sum of 
            (1) maximum output of all BT producers  
            (2) maximum of all Load demand 
        on the lower level network """
        return self.lowerNet_sum_max_lvProdLoad()
    
    def get_lowerNet_hv_bus_df(self, bus_voltage:float=20.6):
        """ Extract higher Voltage bus in the lower network: These are 
            bus for witch vn_kv=20.6 
        """
        return self._lowerNet_hv_bus_df(bus_voltage)
    


        
        
class SensAnlysisResult:
    """
    Initiate the Sensitivity analysis with the folder_location
    
    Parameters:
    ------------
    folder_location: (Str)
        Location of the folder where the results of the sensitivity analysis are stored
    
    """
    
    def __init__(self, folder_location):
        self._folder_location = folder_location
        self._files_in_folder_list = os.listdir(self._folder_location) 
        self._plkFiles_in_folder_list = self._extractPlkFiles(self._files_in_folder_list)
        self._check_fileName_Format(self._plkFiles_in_folder_list)
        self._sortedPlkFiles_in_folder_list = self._sort_plkFiles_in_folder(self._plkFiles_in_folder_list)
        
        
    def _check_fileName_Format(self, plkFiles_in_folder_list):
        # Check if the files name are in the expected format. The Expected format ougth to be 
        # modelName_btRangeName_SimNumber.plk such that the split length ==3 
        len_splited = len(plkFiles_in_folder_list[0].split('_'))
        
        if len_splited > 3: 
            raise Exception('The *.plk files in '+ self._folder_location+'are not in the expected format. \n'
                            +'\t   Make sure they are named such as modelName_btRangeName_SimNumber.plk')
    
    def _extractPlkFiles(self, files_in_folder_list):
        # Extract only the plk files in folderlocation
        plk_files_list = [cur_file for cur_file in files_in_folder_list if cur_file.endswith('.plk')]
        return plk_files_list;
    

    def _sort_plkFiles_in_folder(self, plkFiles_in_folder_list):
        
        first_plkFile_in_folder_name = plkFiles_in_folder_list[0]
                
        # cur_file.split('_')[-1] is used to get the the last elm wich is somethink like 'n.plk' 
        #                                                                          where  n is an integer 
        files_indexAndExtenxion_list = [cur_file.split('_')[-1] for cur_file in plkFiles_in_folder_list] 
        
        # Extract the files index and sort them in ascending order
        # file_index.split('.')[0] is used to get the file index i.e,  || n where  n is an integer 
        sorted_index = np.sort( [int(file_index.split('.')[0]) for file_index in files_indexAndExtenxion_list
                                ] )
        
        # Separate first file in folder name
        self._prediction_model_name, bt_range_name, _ = first_plkFile_in_folder_name.split('_')  
        
        # Create a new list spaning from n to the total number of element in plkFiles_in_folder_list
        files_in_folder_list_out = [ self._prediction_model_name+'_'+bt_range_name+'_'+str(elm)+'.plk' 
                                    for elm in sorted_index ]
        
        return files_in_folder_list_out
          
        
        
    def in_dataFrame(self, start_date=None, end_date=None):
        """
        Transform all the saved results from multivariate simulation in a dataframe where each element 
        is the  curtailed energy for that particular simulation. The df's index represent the variation 
        of the maximum output of the controlled HT producer while the columns represent the added BT
        production in the network.
        
        
        Parameters:
        -----------
        start_date: (str) optional
            First day (included) to consider in the testing set for the simulation. 
            If the argument `is given, the curtailed energy (each element of the df) considers the 
            simulation starting from the given date. 
            else the whole data is considered
        end_date: (str) optional
            Last day (not included) to consider in the testing set for the simulation. 
            If the argument `is given, the curtailed energy (each element of the df) considers the 
            simulation up to the given date. 
            else the whole data is considered
        
        """
        res_dict = {} # Dictionary to save variables
        
        for curFileName in self._sortedPlkFiles_in_folder_list:
            cur_file_data = joblib.load(self._folder_location+curFileName)  # Load files
            cur_file_data_keys_list =list(cur_file_data.keys())            # Get keys names in the current file
            energy_curt_list  = []                                         # Create an energy list 

            for cur_key in cur_file_data_keys_list:  # for each element in the loaded dictionary
                data_df = cur_file_data[cur_key]['Power Sgen']
                
                # data_df.iloc[:,0] => Power injected when there is  no control
                # data_df.iloc[:,1] => Power injected using current controler
                if end_date is None:
                    power_curt = (data_df.iloc[:,0] - data_df.iloc[:,1]).sum()   
                else : 
                    mask_period_interest = (data_df.index>=start_date) & (data_df.index<=end_date)
                    power_curt = (data_df[mask_period_interest].iloc[:,0] 
                                  - data_df[mask_period_interest].iloc[:,1]).sum()   
                    
                energy_curt_list.append( power_curt*Δt ) 
                
            col_name = cur_file_data_keys_list[0].split()[0] 
            res_dict.update({col_name: energy_curt_list})
           
        # Create index name for the resulting dataframe
        df_index = [key.split()[1].split('=')[1] for key in cur_file_data_keys_list]
        
        # Crete resulting dataframe
        self._res_df = pd.DataFrame(res_dict, index=df_index)
        
#         # Rename column of  resulting dataframe
#         self._res_df.columns = [elm.split('=')[1] for elm in self._res_df.columns]
         
        return self._res_df
        
    
    def print_sorted_filesNames(self): 
        print(self._plkFiles_in_folder_list)

    
    def plot_heatmap(self, fig_params=None, 
                     contour_color='yellow', 
                     contour_level=np.arange(0,700,100), 
                     colmap='twilight', 
                     show_ylabel=False, 
                     show_cbar=True, 
                     show_contour=True, 
                     anotation = False,):
        
        # Check whether _res_df is already defined i.e self.in_dataFrame() 
        # is already executed once. If yes, thre is no exception, otherwise
        # execute the funcction
        try:
            getattr(self,'_res_df')   
        except AttributeError:
            self._res_df = self.in_dataFrame()
        
        # if fig_params is not given plot the heatmap in a new figure otherwise fig_params must 
        # be an axe from plt.subplots()
        # TODO Verify if fig_params is actualy an axe and issue an error in the contrary
        if fig_params == None:
            fig, axx = plt.subplots(figsize=(10,6), dpi=100)
        else:
            axx = fig_params
            

        x_contours = range(len(self._res_df.columns))
        y_contours = range(len(self._res_df.index))
        
        if show_contour: 
            cntr = axx.contour(x_contours, y_contours, 
                                    self._res_df.iloc[::-1,:],
                                    levels = contour_level,
                                    colors=contour_color,
                                    linewidths=1 ) 
            # Contour lables
            axx.clabel(cntr, fmt='%1.0f',inline_spacing=10,fontsize=10)

        # colorbar kwargs 
        clbar_kwargs = dict (label = "Curtailed Energy (MWh/Year)", anchor=(0,.5), shrink=0.7)
        
        # anotation kwargs
        annot_kw = dict(size=10)
        
        # actual plot()
        sbn.heatmap(self._res_df.iloc[::-1,:],
                    ax=axx, 
                    annot_kws=annot_kw,
                    fmt='.0f', 
                    lw=0.,
                    annot=anotation,
                    cbar = show_cbar,
                    cbar_kws=clbar_kwargs, 
                    cmap=colmap)
        
        if show_cbar & show_contour: 
            # axx collection[7] contains the colorbar
            axx.collections[7].colorbar.add_lines(cntr)

        axx.set( 
            xlabel ='BT Production increase Variation (Mwh)', 
            title = self._prediction_model_name );
        
        if show_ylabel: 
            axx.set( ylabel ='P0100 Maximum Prod (MWh)',  );  

            
            
            

class SensAnlysisResults(SensAnlysisResult):#This class inherits super properties from the  SensAnlysisResult
    """
    Initiate the Sensitivity analysis with the folder's (location) associated 
    with  each model to consider.
    
    """

    def __init__(self, models_folder_location, models_name, testSet_date, p_Hv_Lv_range):
        """
    Parameters:
    -----------
    models_folder_location: tuple of str
        The relative folder location of Each Model to consider 
    models_name: tuple of str
        The name of each model to consider
    testSet_date: (tuple of str)
        (0) test_set_starting date included
        (1) test set stopind date not includes
    p_Hv_Lv_range: (tuple of array)
        IMPORTANT: Make sure that these parameters are the same that are used in the 
        notebook Sensitivity analysis simulation 
        (0) P_Hv_max_range : Range of Maximum Power for the controlled Producer
        (1) P_Lv_max_range : Range of Maximum Power added for all the Lower voltage producer

        """

        self._models_folder_location = models_folder_location
        self._models_name = models_name
        self._testSet_startDate = testSet_date[0]
        self._testSet_endDate= testSet_date[1]
        checker._check_input_concordance(self)
        self._files_in_folder_list_dict = {}
        self._plkFiles_in_folder_list_dict = {}
        self._sortedPlkFiles_in_folder_list_dict = {}
        checker._check_numberOf_plk_Files_in_folders(self)
        self._P0100_max_range = p_Hv_Lv_range[0]   # Define the maximum power output P0100 
        self._bt_add_range  = p_Hv_Lv_range[1]

        
        # For each model or folder do : (1) List files in folder, (2) Extract plk Files, 
        #                               (3) Check files format,   (4) Sort plk files
        # and add the result in the corresponding dictionary
        for cur_model_name, cur_model_folder in zip(self._models_name, self._models_folder_location):
            
            self._files_in_folder_list_dict.update({cur_model_name: os.listdir(cur_model_folder)})
            self._plkFiles_in_folder_list_dict.update({cur_model_name: 
                                                      super()._extractPlkFiles(
                                                          self._files_in_folder_list_dict[cur_model_name] )
                                                     }) 
            super()._check_fileName_Format(self._plkFiles_in_folder_list_dict[cur_model_name])
            
            self._sortedPlkFiles_in_folder_list_dict.update({cur_model_name:
                                                            super()._sort_plkFiles_in_folder(
                                                                self._plkFiles_in_folder_list_dict[cur_model_name]) 
                                                           })
        
        
 
    def _read_files_at(self, bt_file_index:int, ht_file_index_list:list):
        """ 
        For each model_name in _sortedPlkFiles_in_folder_list_dict[model_name], read the file 
        at bt_file_index to extract both : 
            (1) the voltage rise dataframe ('maxV_rise_df') 
            (2) the Power injected dataframe with and without control ('Power Sgen') 
        for the simulation indexed by each elm in ht_file_index_list. Save each extracted variable   
        in the corresponding dictionary that are output
        """
        voltage_rise_df_dict = {}
        power_df_dict = {}
        show_exeption_message=True 
        
        for cur_ht_file_index in ht_file_index_list:
            for cur_model_name, cur_model_folder in zip(self._models_name, self._models_folder_location):
                plk_file_name = self._sortedPlkFiles_in_folder_list_dict[cur_model_name][bt_file_index]
                file2read_path = cur_model_folder+plk_file_name
                bt_file_dict = joblib.load(file2read_path)         # Load file
                bt_file_dict_keys_list = list(bt_file_dict.keys()) # get keys from the dict that is 
                                                                   #                the loaded file
                key2use = bt_file_dict_keys_list[cur_ht_file_index] 
                name2use = cur_model_name+' '+key2use
                voltage_rise_df_dict.update({name2use: bt_file_dict.get(key2use)['maxV_rise_df'] })
                
                try: # catch exception If the collumn 'Power Sgen' is not present in the read dataFrame 
                    power_df_dict.update({name2use: bt_file_dict.get(key2use)['Power Sgen'] })
                except KeyError: 
                    if show_exeption_message: # show exeption message only once
                        print(f'The collumn [\'Power Sgen\'] is not present in', 
                                       f'files located in {cur_model_folder}')
                        show_exeption_message=False
            
        return voltage_rise_df_dict, power_df_dict
    
    
    def _vRise_dict_as_dataFrame(self, voltageRise_df_dict, power_df_dict, v_rise_thresh):
        """ Transform the voltageRise_df_dict (dictionary of dataframe, each key being the result of 
        a simulation) in a dataframe that will be used for plotting,  create var vrise_count_dict 
        ( a dictionnary of the total number of voltage rise above the threshold for each keys in
        voltageRise_df_dict) and var caping_count_dict  ( a dictionnary of the total number of capping
        for each keys in voltageRise_df_dict)
            
        Parameters
        ----------
        voltageRise_df_dict: dict of dataframe
            Each key has the following format: '{model_name} BT+={bt} P0100={ht}' 
            where {model_name} is self._models_name 
                  {bt}         is bt argument in vRise_boxplot(**)
                  {ht}         is ht argument in vRise_boxplot(**)
        v_rise_thresh: optional, float
            Voltage rise threshold
        
        Output:
        -------
        df2use: dataframe
            Resulting Dataframe
        """
        
        df2use = pd.DataFrame(columns=['V_rise', 'Model', 'Power'])# Create an empty dataframe with following column
        
        self._vrise_count_dict = {}   # Create empty dictionnary
        self._capping_count_dict = {} # Create empty dictionnary
        
        for cur_key in voltageRise_df_dict.keys(): # For each key in voltageRise_df_dict
                                                   # the same keys are in power_df_dict
                
            # Extract dataframe containing the voltage rises and injected powers 
            # recorded for the selected simulation
            cur_vRise_df_model = voltageRise_df_dict[cur_key] 
            
            # create a mask that span # only the interested period
            mask = ( (cur_vRise_df_model.index>=self._testSet_startDate)  
                    & (cur_vRise_df_model.index<=self._testSet_endDate) ) 
            
            # filter dataframes using  mask
            cur_vRise_df_filtered = cur_vRise_df_model[mask]
            # Extract solely data where the voltage rise recorded is above the define threshold
            cur_vRise_df_filtered = cur_vRise_df_filtered[cur_vRise_df_filtered>=v_rise_thresh]
            # Update dictionaries
            self._vrise_count_dict.update({cur_key:len(cur_vRise_df_filtered.dropna()) })
            
            # The try except is used here because for some models the cur_key is not present 
            # in the power_df_dict  
            try: 
                cur_power_df_model = power_df_dict[cur_key]
                mask = ( (cur_power_df_model.index>=self._testSet_startDate)
                          & (cur_power_df_model.index<=self._testSet_endDate) ) 
                 
                cur_power_df_filtered = cur_power_df_model[mask]                   
                self._capping_count_dict.update({cur_key: 
                                                 self._compute_capping_count(cur_power_df_filtered) 
                                                })
            except KeyError: pass
            
            # Create an intermediate dataframe
            df = pd.DataFrame(cur_vRise_df_filtered.dropna().values, columns=['V_rise'])
            df[['Model', 'Power']] = cur_key.split()[0], cur_key.split()[1]+' '+cur_key.split()[2]
            df2use = pd.concat([df2use, df]) # Concatanate created dataframe with the exsisting dataframe  
            
        return df2use
        
        
        
    def _compute_capping_count(self, injected_power_df):
        """ Compute and return the number of capping that occured given 
        the input power dataframe """
        
        # injected_power_df.iloc[:,0] => Power injected when there is  no control
        # injected_power_df.iloc[:,1] => Power injected using controler
        
        # Create a lambda function named equality_check that will verifiy if for the current row,  
        # the power injected with controlled On is the same as when no control is applied. 
        # When both values are not equal (output True i.e. a capping or curtailement is occuring) 
        #                          equal (output False i.e. a no capping) 
        # equality_check = lambda cur_row: True if cur_row[1]!=cur_row[0] else False
        equality_check = lambda cur_row: cur_row[1]!=cur_row[0] 
        
        # apply equality check to each row of the df and sum the resulting df
        nb_event = (injected_power_df.apply(equality_check, axis=1)).sum()
        
        return nb_event 
        
        
        
        

        
    def vRise_boxplot(self, ht=0., bt=0., v_rise_thresh=1.025,
                     fig_params=None
                     ):
        """ Create a box plot of voltage rise above the defined threshold for simulations where the controlled
        HT producer maximum power is arg(ht) and the added maximum bt power is arg(bt)
        
        Parameters:
        -----------
        ht : (float, default = 0)
            Maximum Power of the controlled HT producer
           : (str,  'All')
           Range of all variation of the Maximum Power of the controlled HT producer
        bt : (float, default = 0) 
            Added Power to the BT producers in the network
        v_rise_thresh: (float, default = 1.025)
            Maximum authorised threshold
        fig_params : (Figure axis, optionnal)
            Figure axis where to plot the current box plots
            
        """
        self._bt = bt
        self._ht = ht
        checker._check_boxplot_inputs(self, ht, bt) # Check if boxplot inputs are authorised
        # Extract the number of the bt file to read
        bt_file_to_read_index = [file_ind for file_ind, curValue in enumerate(self._bt_add_range) 
                                 if curValue==bt][0]
        
        if type(ht) is float : # -----------------------   ht is a float           -----------------------
            # Extract the dict keys index of the HT simulation in the saved BT file
            ht_dict_keys_to_read_index = [file_ind for file_ind, curValue in enumerate(self._P0100_max_range) 
                                          if curValue==ht][0]
            
            # Read file and extract variable into a dictionary
            voltageRise_df_dict, power_df_dict = self._read_files_at(bt_file_to_read_index, 
                                                                     [ht_dict_keys_to_read_index])

            # convert the read dictionary in a dataframe to plot
            df_to_plot = self._vRise_dict_as_dataFrame(voltageRise_df_dict, power_df_dict, v_rise_thresh)
            
            if fig_params is not None:
                axx = fig_params
                if len(df_to_plot)!=0 :# Plot dataframe only if some data is present 
                    sbn.boxplot(x='Power', y='V_rise', data=df_to_plot, 
                                hue='Model', hue_order=list(self._models_name), 
                                width=0.5, fliersize=1, linewidth=0.8, ax=axx, )
            else:
                fig, axx = plt.subplots(1, figsize=(3,5))
                if len(df_to_plot)!=0 :# Plot dataframe only if some data is present 
                    sbn.boxplot(x='Power', y='V_rise', data=df_to_plot, 
                                hue='Model', hue_order=list(self._models_name), 
                                width=0.5, fliersize=1, linewidth=0.8, ax=axx, )

        elif type(ht) is str :# -----------------------  if ht is str (All)           -     ---------------
            ht_dict_keys_to_read_index = range(len(self._P0100_max_range)) #        
            voltageRise_df_dict, power_df_dict = self._read_files_at(bt_file_to_read_index, 
                                                                     list(ht_dict_keys_to_read_index))
            # convert the read dictionary in a dataframe to plot
            df_to_plot = self._vRise_dict_as_dataFrame(voltageRise_df_dict, power_df_dict, v_rise_thresh)
            
            if fig_params is None: # define a figure is an axis is not given as input
                fig, axx = plt.subplots(1, figsize=(15,6)) 
            else: axx = fig_params

            sbn.boxplot(x='Power', y='V_rise', data=df_to_plot, 
                        hue='Model', hue_order=list(self._models_name),  # Actual Plot
                        width=0.7, fliersize=1, linewidth=0.8, ax=axx)

            # Each elm in df_to_plot.Power.unique().tolist() gives BT+=bt_max P0100=ht_max
            # Hence the elm.split()[-1]                extracts P0100=ht_max and 
            #           elm.split()[-1].split('=')[-1] extracts ht_max 
            ht_max_labels_list = [elm.split()[-1].split('=')[-1] 
                                  for elm in df_to_plot.Power.unique().tolist()]

            axx.set_xticks(axx.get_xticks(), ht_max_labels_list)
            ticks_ = np.arange(v_rise_thresh,1.037,0.001)
            axx.set_yticks(ticks=ticks_,
                           labels=np.round(ticks_, decimals=3))

            axx.set_xlabel('HV Max Prod (MW)')
            axx.set_title(f'Lv Added Power = {bt} MW')
            
             
            
        
    def _count_dict_as_dataFrame(self, var_count_dict):
        """ Transform the variable indexed by var_count_dict  into a dataframe. 
        Note that var_count_dict is either <<_vrise_count_dict>> or <<_capping_count_dict>> 
        """
                
        # check the exixstence of var_count_dict in the local space of the instance
        checker._check_countDictAsDf_input_inLocalSpace(self, var_count_dict)
        
        # Get the self variable i.e. vRiseOrCapping_count_dict = self._vrise_count_dict 
        #                         or vRiseOrCapping_count_dict = self._capping_count_dict 
        vRiseOrCapping_count_dict = getattr(self, var_count_dict )
        
        # Extract all the ht_max in the self._vrise_count_dict to create a list 
        # cur_key is such that  BT+=bt_max P0100=ht_max
        # Hence the cur_key.split()[-1]                extracts P0100=ht_max  
        #       and cur_key.split()[-1].split('=')[-1] extracts ht_max that is converted in float
        list_index = [float(cur_key.split()[-1].split('=')[-1]) 
                      for cur_key in vRiseOrCapping_count_dict if cur_key.startswith(self._models_name[0])]

        df_to_plot =pd.DataFrame(index=list_index) # Create an empty dataframe with the index being list_index
        
        for cur_mod_name in self._models_name:
            # Extract the total number of voltage rise above threshold  associated with all the element in 
            # self._vrise_count_dict that start with the cur_modName
            vRiseOrPower_list = [vRiseOrCapping_count_dict[cur_key] 
                                 for cur_key in vRiseOrCapping_count_dict if cur_key.startswith(cur_mod_name)]
            try: # In case vRiseOrPower_list is empty
                df_to_plot[cur_mod_name] = vRiseOrPower_list
            except ValueError:
                df_to_plot[cur_mod_name] = [np.nan]*len(df_to_plot)
            
        return df_to_plot
    
    
            
    def countplot(self, count_dict_name, fig_params=None):
        """ Plot the total number of :
            voltage rise above the defined threshold 
            or the total number of capping commands 
        for the simulation defined by the parameters of the last call of *.vRise_boxplot(*args) 
            
        Parameters:
        -----------
        count_dict_name: str 
            'v_rise' : Plot of the total number of voltage rise above the defined threshold,
            'capping': Plot the total number of capping command sent to the energy producer
        fig_params : (Figure axis, optionnal)
            Figure axis where to plot the current box plots
             
        """

        # _vrise_count_dict and _capping_count_dict are 2 dict created in 
        #  _vRise_dict_as_dataFrame(*args)
        self._dict_vars = {'v_rise':'_vrise_count_dict', 
                           'capping':'_capping_count_dict'
                          } # Create a dict
        
        checker._check_countplot_inputs(self, count_dict_name)
        
        df_to_plot = self._count_dict_as_dataFrame(self._dict_vars[count_dict_name])
        
        if fig_params is None:
            x_len = len(df_to_plot)/2 # Define len of figure depending on the number of 
                                      # element in the df_to_plot
            fig_ = df_to_plot.plot(marker="*", ls='', figsize=(x_len,4))
        else:
            fig_ = df_to_plot.plot(marker="*", ls='', ax=fig_params)
            fig_.legend('_')
        fig_.set_xticks(df_to_plot.index)
        fig_.set(ylabel=('Count'), 
                 xlabel=('HV Max Prod (MW)'),
                 title = f'Bt+ = {self._bt} MW, {count_dict_name.capitalize()} ' )
        fig_.grid(axis='x', lw=0.2)


        
        