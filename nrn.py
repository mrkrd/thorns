#!/usr/bin/env python

from __future__ import division
__author__ = "Marek Rudnicki"

import numpy as np
from neuron import h
import waves as wv

def record_voltages(secs):
    vecs = []
    for sec in secs:
        vec = h.Vector()
        vec.record(sec(0.5)._ref_v)
        vecs.append(vec)
    return vecs


def plot_voltages(fs, vecs):
    import biggles

    p = biggles.FramedArray(len(vecs), 1)

    for i,vec in enumerate(vecs):
        p[i,0].add( biggles.Curve(wv.t(fs, vec), vec) )

    return p


def main():
    pass

if __name__ == "__main__":
    import biggles
    main()
