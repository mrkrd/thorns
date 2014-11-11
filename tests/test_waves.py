#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test thorns.waves module.

"""

from __future__ import division, absolute_import, print_function

__author__ = "Marek Rudnicki"

import numpy as np
from numpy.testing import assert_equal

import thorns.waves as wv


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
