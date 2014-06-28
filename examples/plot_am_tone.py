#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot amplitude modulated tone.

"""

from __future__ import division, absolute_import, print_function

__author__ = "Marek Rudnicki"

import numpy as np

import thorns.waves as wv

def main():

    fs = 48e3

    sound = wv.amplitude_modulated_tone(
        fs=fs,
        fm=100,
        fc=2e3,
        m=1,
        duration=0.1,
        pad=0.01
    )

    wv.plot_signal(sound, fs=fs)
    wv.show()

if __name__ == "__main__":
    main()
