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

    all_values = np.concatenate( vecs )
    hi = all_values.max()
    lo = all_values.min()

    plot = biggles.Table(len(vecs), 1)
    plot.cellpadding = 0
    plot.cellspacing = 0
    for i,vec in enumerate(vecs):
        p = biggles.Plot()
        p.add( biggles.Curve(wv.t(fs, vec), vec) )
        p.yrange = (lo, hi)
        plot[i,0] = p


    p.add( biggles.LineX(0) )
    p.add( biggles.Label(0, (hi+lo)/2, "%.2f mV" % (hi-lo), halign='left') )

    p.add( biggles.LineY(lo) )
    p.add( biggles.Label((len(vec)/fs/2), lo, "%.1f ms" % (1000*len(vec)/fs), valign='bottom') )


    return plot


def main():
    pass

if __name__ == "__main__":
    import biggles
    main()
