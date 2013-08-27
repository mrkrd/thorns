from __future__ import division

__author__ = "Marek Rudnicki"


import argparse
import os
import sys


parser = argparse.ArgumentParser()

parser.add_argument(
    '--backend',
)

parser.add_argument(
    '--machines',
    nargs='+',
    default=None
)

parser.add_argument(
    '--cache'
)

parser.add_argument(
    '--plot',
    nargs='?',
    const='show'
)

parser.add_argument(
    '--pdb',
    action='store_true'
)

parser.add_argument(
    '--dependencies',
    nargs='+',
    default=None
)

parser.add_argument(
    'files',
    nargs='*'
)



args = parser.parse_known_args()[0]



if args.pdb:
    import pdb, sys, traceback
    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        pdb.pm()
    sys.excepthook = info



from mrlib.dumpdb import (
    dumpdb,
    loaddb,
)

from mrlib.maps import (
    map,
    apply,
)

from mrlib.plot import (
    plot,
    show,
)

from mrlib.waves import (
    resample,
)


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
