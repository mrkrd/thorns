#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Scirpt used to generate anf_zilany2014 dataset with auditory nerve
spike trains.

"""

from __future__ import division, absolute_import, print_function

__author__ = "Marek Rudnicki"

import numpy as np

import cochlea

import thorns as th
import thorns.waves as wv


def main():

    fs = 100e3
    cf = 1e3
    dbspl = 50
    tone_duration = 50e-3

    sound = wv.ramped_tone(
        fs=fs,
        freq=cf,
        duration=tone_duration,
        dbspl=dbspl,
    )

    anf_trains = cochlea.run_zilany2014(
        sound,
        fs,
        anf_num=(200,0,0),
        cf=cf,
        seed=0,
        species='cat',
    )

    anf_trains.to_pickle("anf_zilany2014.pkl")

    th.plot_raster(anf_trains)
    th.show()

if __name__ == "__main__":
    main()
