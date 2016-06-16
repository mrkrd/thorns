# -*- coding: utf-8 -*-

"""Map demo.

"""

from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

import numpy as np
import thorns as th


def multiply(a, b):
    """Multiply two numbers."""

    # The output should be wrapped in a dict
    y = {'y': a * b}

    return y


def main():

    # Keys in the input dict MUST correspond to the key word
    # arguments of the function (e.g. multiply)
    xs = {
        'a': np.arange(3),
        'b': np.arange(3)
    }

    # Uncomment `backend` for parallel execution
    ys = th.util.map(
        multiply,
        xs,
        # backend='multiprocessing',
    )

    print(ys)


if __name__ == "__main__":
    main()
