#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This demo generates sample periodic spike trains and plots a
period histogram.

"""

from __future__ import division, absolute_import, print_function

import numpy as np

import thorns as th



def main():

    fs = 10e3                   # Hz
    freq = 10                   # Hz


    ### Calculate spike probability over time (we want to have a
    ### periodic function for a pretty plot)
    t = np.arange(0, 1, 1/fs)
    spike_probability = 0.2 * np.abs(np.sin(2 * np.pi * freq * t))
    rand = np.random.rand(len(spike_probability))




    ### Generate spike trains
    spike_array = np.zeros_like(spike_probability)
    spike_array[rand < spike_probability] = 1
    spike_array = np.expand_dims(spike_array, axis=1)

    spike_trains = th.make_trains(
        spike_array,
        fs=fs,
    )




    ### Plot the histogram
    th.plot_period_histogram(
        spike_trains,
        freq=freq,
        nbins=128,
        density=True
    )

    th.show()






if __name__ == "__main__":
    main()
