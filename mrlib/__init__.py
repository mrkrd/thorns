from __future__ import division

__author__ = "Marek Rudnicki"


import os
import sys


if 'MRpdb' in os.environ:
    import pdb, sys, traceback
    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        pdb.pm()
    sys.excepthook = info


if 'MRlog' in os.environ:
    import logging

    if os.environ['MRlog'] in ('d', 'debug'):
        level = 'DEBUG'

    else:
        level = os.environ['MRlog'].upper()

    logger = logging.getLogger()
    logger.setLevel(level)



from mrlib.dumpdb import (
    dumpdb,
    loaddb,
    get_store,
    mkstore
)

from mrlib.maps import (
    map,
    apply,
)

from mrlib.plotting import (
    plot,
    show,
)

from mrlib.waves import (
    resample,
)


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
