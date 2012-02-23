#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np

import unittest

import spikes as sp


class TestSpikes(unittest.TestCase):
    def setUp(self):

        dt = [('spikes',np.ndarray),
              ('duration',float)]

        a = [(np.array([1,2,4,8]), 10),
             (np.array([1,3,4,9]), 10)]
        self.a = np.array(a, dtype=dt)



        b = [(np.array([4,7, 13,18]), 20),
             (np.array([1,2, 14,15]), 20)]
        self.b = np.array(b, dtype=dt)

    def test_split_and_fold(self):

        silence, tones = sp.split_and_fold(
            spike_trains=self.b,
            silence_duration=10,
            tone_duration=2,
            pad_duration=2,
            remove_pads=False
        )

        # print silence
        print
        print '>>> tones', tones
        print

        # np.testing.assert_array_equal(




if __name__ == "__main__":
    unittest.main()

