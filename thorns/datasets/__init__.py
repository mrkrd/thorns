# -*- coding: utf-8 -*-

"""Sample spike trasins.

"""

from __future__ import division, absolute_import, print_function

__author__ = "Marek Rudnicki"

from os.path import join, dirname

import numpy as np
import pandas as pd


def load_anf_zilany2014():
    """Load and return sample auditory nerve spike trains generated with
    Zilany et al. (2014) inner ear model.

    The simulus was a 50 ms ramped pure tone at 50 dB_SPL (without
    padding).


    Returns
    -------
    spike_trains
        Responses of high-spontaneous rate auditory nerve fibers with
        the characteristic ferquency (CF) of 1 kHz.

    """

    module_path = dirname(__file__)

    spike_trains = pd.read_pickle(
        join(module_path, "anf_zilany2014.pkl")
    )

    return spike_trains



def main():
    pass

if __name__ == "__main__":
    main()
