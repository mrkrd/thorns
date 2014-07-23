"""Utilities.

"""

from __future__ import division, print_function, absolute_import

from thorns.util.dumpdb import (
    dumpdb,
    loaddb,
    get_store,
)

from thorns.util.maps import map, apply
from throns.util import io

from . bisection import find_zero

def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
