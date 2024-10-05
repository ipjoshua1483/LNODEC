import numpy as np
import os
#from main_6 import data_dict
from utils import plot_x_t_all, plot_x_t_all_truncated, plot_x_t_all_truncated_subplots, plot_x_t_all_truncated_individual_plots, plot_u_t
from datetime import datetime
timestamp = 'appj NODEC vs LNODEC inputs redone' #'appj NODEC vs LNODEC alpha 0.5' #'20240425 232428'

data_dict_upload = {
        'node_times': None,
        'node_states_last': None,
        'node_states_control': None,
        'node_inputs_control': None,
        'lya_times': None,
        'lya_states_last': None,
        'lya_states_control': None,
        'lya_states_lyapunov': None,
        'lya_inputs_lyapunov': None,
    }

for name in data_dict_upload.keys():
    file_name = os.getcwd() + '\\data\\' + timestamp + '\\' + name
    data_dict_upload[name] = np.load(file_name + '.npy')
print('Data loaded successfully')

timestamp = datetime.now()
timestamp_name = timestamp.strftime("%Y%m%d %H%M%S")

plot_x_t_all_truncated_subplots(
    data_dict_upload['node_times'], 
    data_dict_upload['node_states_control'],
    data_dict_upload['node_inputs_control'],
    data_dict_upload['lya_times'], 
    data_dict_upload['lya_states_lyapunov'],
    data_dict_upload['lya_inputs_lyapunov'],
    timestamp_name,
)

print('Plots saved successfully')