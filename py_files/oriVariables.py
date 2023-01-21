""" Module containing all the variables that will be used in the ORI problem """


# Create an attribute list to use in functions
attr_list = [('bus', 'name'),
             ('load', 'bus'),
             ('switch', 'bus'),
             ('line', 'from_bus'),
             ('line', 'to_bus'),
             ('trafo', 'hv_bus'),
             ('trafo', 'lv_bus')]

# Set Define set of folders
network_folder = 'pickle_files/'
excel_folder = 'excel_files/'
py_folder = 'py_files/'

Δt = 1 / 6  # Time frequency 10mn ==> 1Hour/6


default_auth_vm_pu_max = 1.02
default_auth_vm_pu_min = 0.95
default_ctrl_hvProd_max = 4.0

train_split_date = '2021 12 31 23:50' # Date of training+Validation split data Lower bond 
trainVal_split_date = '2021 06 01'     # lower date to split training and validation data
testSet_end_date = '2022 06 02'
testSet_start_date = '2021 06 03'


