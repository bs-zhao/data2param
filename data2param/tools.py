import os
import pickle
import torch

def load_instance(path=None, dir_save=None, folder_save=None, name_save=None, device=None):

    # Handle path
    if path==None:
        if dir_save is None:
            dir_save = os.getcwd()     
        if folder_save is None:
            folder_save = "ParameterDecoder"
        if name_save is None:
            name_save = "parameter_decoder"
        path = os.path.join(dir_save, folder_save, f"{name_save}.pkl")    
    
    with open(path, 'rb') as f:
        instance = pickle.load(f)

    return instance
