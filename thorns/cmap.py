from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"


import matplotlib


double_phase = matplotlib.colors.LinearSegmentedColormap(
    'duble_phase_colormap',
    {
        'red': ((0.0, 0.0, 0.0),
                (0.5, 1.0, 0.7),
                (1.0, 1.0, 1.0)),
        'green': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 0.0),
                  (1.0, 1.0, 1.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.5, 1.0, 0.0),
                 (1.0, 0.5, 1.0))
    },
    256
)
double_phase.set_bad('b')
