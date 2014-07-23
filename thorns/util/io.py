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

def read_brainwaref32(filename, stimparams = None):
    """ Read the spiketimings as exported from BrainWare \
    (Tucker-Davis Technologies).
    
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
                    name = "param%i" % i
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
    f.close()
    
    return dataset
