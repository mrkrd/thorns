"""Utilities.

"""

from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

import os

from . dumpdb import dumpdb, loaddb, get_store
from . maps import map, cache
from . bisection import find_zero

def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
