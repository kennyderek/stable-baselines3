from typing import Dict, List
import numpy as np

def dict_obs_to_array(d_obs : Dict[int, np.ndarray], key_list : List[int]):
    '''
    Assumes len d_obs and key_list > 1
    Turns a dictionary d_obs of int->array into a single array a, where
    index i of a is d_obs[key_list[i]]
    '''
    obs_shape = d_obs[key_list[0]].shape # get the length of the observation
    a = np.zeros((len(key_list),) + obs_shape)
    for idx, mapping in enumerate(key_list):
        a[idx] = d_obs[mapping]
    return a

def array_to_dict_actions(arr_actions : np.ndarray, key_list : List[int]):
    '''
    Perform the reverse of dict_obs_to_array
    Turns an array of actions into a dictionary, where action i of a 
    '''
    d_actions = {}
    for idx, mapping in enumerate(key_list):
        d_actions[mapping] = arr_actions[idx]
    return d_actions