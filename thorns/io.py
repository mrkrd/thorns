#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

__author__ = "Jörg Encke"

import struct
import pandas as pds
import copy

"""
Contains functions to read and write spike data in different formats
"""

def read_brainwaref32(filename, stimparams=None):
    """ Read the spiketimings as exported from BrainWare \
    (Tucker-Davis Technologies).
    
    Parameters
    ----------
    filename : string
        The branwaref32 file to import.
    stimparams : dict
        A dict with the parameter names used in the stimulation sequence.
        the key gives the parameter number as an integer while the value \
        is the new name as a string
    
    Returns
    -------
    Pandas.DataFrame
        A DataFrame containing the spike timings in the thonrns spike_train
        format.
    
    """
    
    dict_list = []

    with open(filename,'rb') as f:

        while True:
            #read next 4 byte (32bit)
            s = f.read(4)
            if not s: break
            f32 = struct.unpack('f', s)

            if f32[0] == -2.0:
                #read number of parameters:
                s = f.read(4)
                if not s: break
                length = struct.unpack('f', s)

                s = f.read(4)
                if not s: break
                n_param = int(struct.unpack('f', s)[0])
                #read n parameter values
                param = struct.unpack('f' * n_param, f.read(n_param * 4))
                c_dict = {}

                #create a base dictonary with all parameters
                for i,v in enumerate(param):
                    name = "param%i" % (i+1)
                    c_dict[name] = v
                c_dict['spikes'] = []
                c_dict['duration'] = length[0] * 1E-3 #ms -> s

            if f32[0] == -1.0:
                #c_dict has to be copied
                dict_list.append(copy.deepcopy(c_dict))

            else:
                if len(dict_list) > 0:
                    dict_list[-1]['spikes'].append(f32[0] * 1E-3) #ms -> s
    
    dataset = pds.DataFrame(dict_list)
    
    # Fill in column titles if given
    if stimparams != None:
        for k,v in stimparams.iteritems():
            name = "param%i" % k
            dataset = dataset.rename(columns={name: v})
            
    f.close()
    
    return dataset
