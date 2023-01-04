import pandapower, pandas, numpy, ipyparallel, os, joblib, seaborn

pd = pandas
np = numpy 
pp = pandapower
ipp = ipyparallel
sbn = seaborn

Δt=1/6
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
                import numpy, pandapower,pandas, Par_myFunctions

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



    def __dview__(self):
        return self.dview

    def __run_periodIndex__(self):
        return self.run_periodIndex
    
    def __pred_model__(self):
        return self.pred_model
    
    def __opf_status__(self):
        return self.opf_status
        


    def get_dview(self):
        return self.__dview__()

    def get_run_periodIndex(self):
        return self.__run_periodIndex__()
    
    def get_pred_model(self):
        return self.__pred_model__()
    
    def get_opf_status(self):
        return self.__opf_status__()

        
        
        

class InitNetwork:
    """
    Initiate both the Higher and lower level  Network.
    
    Parameters: 
    -----------
        higherNet: Higher level network
        LowerNet : Lower level Network 
    """
    
    def __init__(self, 
               higherNet:pp.auxiliary.pandapowerNet, 
               LowerNet:pp.auxiliary.pandapowerNet 
                ):
        
        self._higherNet = higherNet
        self._lowerNet  = LowerNet
        self.__check_network_order__()
        
        
        
    def __check_network_order__(self):
        # Make sure the first input i.e. the network is bigger than the second one
        if len(self._higherNet.bus) < len(self._lowerNet.bus): 
            raise Exception('The first input (higher net) has a lower number of bus\n'
                            'compared to the second input (lower network). ')
    


    def __sum_max_p_wm_higherNet__(self):
        """ Compute the total of BT prod and Load on higher level network """
        bt_max = self._higherNet.sgen[self._higherNet.sgen.name.isna()].max_p_mw.sum()
        load_max = self._higherNet.load.max_p_mw.sum()
        
        return bt_max, load_max
        
        
    def __sum_max_p_wm_lowerNet__(self):
        """ Compute the total of BT prod and Load on lower level network """
        bt_max = self._lowerNet.sgen[self._higherNet.sgen.name.isna()].max_p_mw.sum()
        load_max = self._lowerNet.load.max_p_mw.sum()
        
        return bt_max, load_max
    
    
    def __lowerNet_hv_bus_df__(self):
        """ Extract higher Voltage bus bus in the lower network: These are 
            bus for witch vn_kv=20.6 
        """
        return self._lowerNet.bus.groupby('vn_kv').get_group(20.6)
         
        
    ###  Define getter    ------------------------------------------------------------------
    def get_higherNet(self):
        return self._higherNet
    
    def get_lowerNet(self):
        return self._lowerNet
    
    def get_sum_max_p_wm_higherNet(self):
        return self.__sum_max_p_wm_higherNet__()
    
    def get_sum_max_p_wm_lowerNet(self):
        return self.__sum_max_p_wm_lowerNet__()
    
    def get_lowerNet_hv_bus_df(self):
        return self.__lowerNet_hv_bus_df__()
    


        
        
class SensAnlysisResults:
    """
    Initiate the Sensitivity analysis with the folder_location
    
    Parameters:
    ------------
    folder_location: (Str)
        Location of the folder where the results of the sensitivity analysis are stored
    
    """
    
    def __init__(self, folder_location):
        self.folder_location = folder_location
        self.files_in_folder_list = os.listdir(self.folder_location) 
        self.plkFiles_in_folder_list = self.__extractPlkFiles__()
        self.__check_fileName_Format__()
        self.sortedPlkFiles_in_folder_list = self.__sort_plkFiles_in_folder__()
        
        
    def __check_fileName_Format__(self):
        # Check if the files name are in the expected format. The Expected format ougth to be 
        # modelName_btRangeName_SimNumber.plk such that the split length ==3 
        len_splited = len(self.plkFiles_in_folder_list[0].split('_'))
        
        if len_splited > 3: 
            raise Exception('The *.plk files in '+ self.folder_location+'are not in the expected format. \n'
                            +'\t   Make sure they are named such as modelName_btRangeName_SimNumber.plk')
    
    def __extractPlkFiles__(self):
        # Extract only the plk files in folderlocation
        plk_files_list = [cur_file for cur_file in self.files_in_folder_list if cur_file.endswith('.plk')]
        return plk_files_list;
    

    def __sort_plkFiles_in_folder__(self):
        
        first_plkFile_in_folder_name = self.plkFiles_in_folder_list[0]
                
        # cur_file.split('_')[-1] is used to get the the last elm wich is somethink like 'n.plk' 
        #                                                                          where  n is an integer 
        files_indexAndExtenxion_list = [cur_file.split('_')[-1] for cur_file in self.plkFiles_in_folder_list] 
        
        # Extract the files index and sort them in ascending order
        # file_index.split('.')[0] is used to get the file index i.e,  || n where  n is an integer 
        sorted_index = np.sort( [int(file_index.split('.')[0]) for file_index in files_indexAndExtenxion_list
                                ] )
        
        # Separate first file in folder name
        self.prediction_model_name, bt_range_name, _ = first_plkFile_in_folder_name.split('_')  
        
        # Create a new list spaning from n to the total number of element in plkFiles_in_folder_list
        files_in_folder_list_out = [ self.prediction_model_name+'_'+bt_range_name+'_'+str(elm)+'.plk' 
                                    for elm in sorted_index ]
        
        return files_in_folder_list_out
          
        
        
    def in_data_frame(self, end_date=None):
        """
        Transform all the saved results from multivariate simulation in a dataframe where each element 
        is the  curtailed energy for that particular simulation. The df's index represent the variation 
        of the maximum output of the controlled HT producer while the columns represent the added BT
        production in the network.
        
        
        Parameters:
        -----------
        end_date: (str) optional
            Last day (not included) to consider in the testing set for the simulation. 
            If the argument `is given, the curtailed energy (each element of the df) considers the 
            simulation up to the given date. 
            else the whole data is considered
        
        """
        res_dict = {} # Dictionary to save variables
        
        for curFileName in self.sortedPlkFiles_in_folder_list:
            cur_file_data = joblib.load(self.folder_location+curFileName)  # Load files
            cur_file_data_keys_list =list(cur_file_data.keys())            # Get keys names in the current file
            energy_curt_list  = []                                         # Create an energy list 

            for cur_key in cur_file_data_keys_list:  # for each element in the loaded dictionary
                data_df = cur_file_data[cur_key]['Power Sgen']
                
                # data_df.iloc[:,0] => Power injected when there is  no control
                # data_df.iloc[:,1] => Power injected using current controler
                if end_date is None:
                    power_curt = (data_df.iloc[:,0] - data_df.iloc[:,1]).sum()   
                else : 
                    mask_period_interest = data_df.index <= end_date
                    power_curt = (data_df[mask_period_interest].iloc[:,0] 
                                  - data_df[mask_period_interest].iloc[:,1]).sum()   
                    
                energy_curt_list.append( power_curt*Δt ) 
                
                
            col_name = cur_file_data_keys_list[0].split()[0] 
            res_dict.update({col_name: energy_curt_list})
           
        # Create index name for the resulting dataframe
        df_index = [key.split()[1].split('=')[1] for key in cur_file_data_keys_list]
        
        # Crete resulting dataframe
        self.res_df = pd.DataFrame(res_dict, index=df_index)
        
#         # Rename column of  resulting dataframe
#         self.res_df.columns = [elm.split('=')[1] for elm in self.res_df.columns]
         
        return self.res_df
        
    
    def print_sorted_filesNames(self): 
        print(self.plkFiles_in_folder_list)

    
    def plot_heatmap(self, fig_params=None, 
                     contour_color='yellow', 
                     contour_level=np.arange(0,700,100), 
                     colmap='twilight', 
                     show_ylabel=False, 
                     show_cbar=True, 
                     show_contour=True, 
                     anotation = False,):
        
        # Check whether res_df is already defined i.e self.in_data_frame() 
        # is already executed once. If yes, thre is no exception, otherwise
        # execute the funcction
        try:
            getattr(self,'res_df')   
        except AttributeError:
            self.res_df = self.in_data_frame()
        
        # if fig_params is not given plot the heatmap in a new figure otherwise fig_params must 
        # be an axe from plt.subplots()
        # TODO Verify if fig_params is actualy an axe and issue an error in the contrary
        if fig_params == None:
            fig, axx = plt.subplots(figsize=(10,6), dpi=100)
        else:
            axx = fig_params
            

        x_contours = range(len(self.res_df.columns))
        y_contours = range(len(self.res_df.index))
        
        if show_contour: 
            cntr = axx.contour(x_contours, y_contours, 
                                    self.res_df.iloc[::-1,:],
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
        sbn.heatmap(self.res_df.iloc[::-1,:],
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
            title = self.prediction_model_name );
        
        if show_ylabel: 
            axx.set( ylabel ='P0100 Maximum Prod (MWh)',  );  
        
