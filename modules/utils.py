from collections.abc import MutableMapping
import os
import torch
import numpy as np


def flatten_cfg(cfg):
    items = []
    for key, value in cfg.items():
        if isinstance(value, MutableMapping):
            items.extend(flatten_cfg(value).items())
        else:
            items.append((key, value))
    return dict(items)


def log(result, global_step, logger):
    for k, v in result.items():
        logger.add_scalar(k, v, global_step)


def outputdir_make_and_add(outputdir, title=None):
    #creates outputdir
    os.makedirs(outputdir,exist_ok=True)
    folder_num = len(next(os.walk(outputdir))[1]) #counts how many folders already there 
    if folder_num == 0:
        folder_num = 1
    elif folder_num == 1 and next(os.walk(outputdir))[1][0][0] == ".":
        folder_num = 1
    else:
        folder_num = max([int(i.split('-')[0]) for i in next(os.walk(outputdir))[1] if i[0] != '.'],default=0) + 1 # this looks for max folder num and adds one... this works even if title is used (because we index at 1) (dot check to ignore .ipynb) 
        #currently returns error when a subfolder contains anything other than a number (exept dot handle) 
        #so essentially this assumes the outputdir structure with numbers (and possible titles). will need to fix if i want to use it later for something else
    
    if title == None:
        outputdir += '/' + str(folder_num) #adds one
    else:
        outputdir += '/' + str(folder_num) + f'-({title})' #adds one and appends title
    os.makedirs(outputdir,exist_ok=True)
    return outputdir