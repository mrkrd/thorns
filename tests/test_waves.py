#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test thorns.waves module.

"""

from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import assert_equal

import thorns.waves as wv



def test_electrical_pulse_charge():

    durations = [1, 1, 1]
    amplitudes = [-1, 2, -1]

    fs = 100

    pulse = wv.electrical_pulse(
        fs=fs,
        amplitudes=amplitudes,
        durations=durations,
        charge=1
    )


    charge = np.sum(np.abs(pulse))/fs

    assert_equal(charge, 1)



def test_electrical_amplitudes_2():

    durations = [1, 0.5]

    amplitudes = wv.electrical_amplitudes(
        durations=durations,
        polarity=1,
    )


    assert_equal(amplitudes, [0.5, -1])


def test_electrical_amplitudes_3():

    durations = [0.5, 1, 0.5]
    ratio = 0.3
    polarity = 1

    amplitudes = wv.electrical_amplitudes(
        durations=durations,
        polarity=polarity,
        ratio=ratio
    )

    assert_equal(amplitudes, [0.3, -0.5, 0.7])
