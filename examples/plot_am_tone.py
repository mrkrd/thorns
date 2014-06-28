#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot amplitude modulated tone.

"""

from __future__ import division, absolute_import, print_function

__author__ = "Marek Rudnicki"

import os
import numpy as np
import matplotlib.pyplot as plt

import thorns.waves as wv

def main():

    fs = 48e3

    sound = wv.amplitude_modulated_tone(
        fs=fs,
        fm=100,
        fc=1e3,
        m=0.7,
        duration=0.1,
    )

    wv.plot_signal(sound, fs=fs)



    ### Save the plot, if doesn't exist
    fname = "am_tone.png"
    if not os.path.exists(fname):
        plt.tight_layout()
        plt.savefig(fname, dpi=80)



    wv.show()

if __name__ == "__main__":
    main()
