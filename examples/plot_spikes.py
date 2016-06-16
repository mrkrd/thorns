#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot a raster plot of auditory nerve spike trains generated using
Zilany et al. (2014) model.

"""

from __future__ import division, absolute_import, print_function

import thorns as th
from thorns.datasets import load_anf_zilany2014


def main():

    # Load spike trains
    spike_trains = load_anf_zilany2014()

    print(spike_trains.head())

    # Calculate vector strength
    cf, = spike_trains.cf.unique()
    onset = 10e-3

    trimmed = th.trim(spike_trains, onset, None)
    vs = th.vector_strength(trimmed, freq=cf)

    print()
    print("Vector strength: {}".format(vs))

    # Plot raster plot
    th.plot_raster(spike_trains)

    # Show the plot
    th.show()                   # Equivalent to plt.show()


if __name__ == "__main__":
    main()
