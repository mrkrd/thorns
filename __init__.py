from __future__ import division

__author__ = "Marek Rudnicki"


import argparse
import logging
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log',
    dest='loglevel'
)
parser.add_argument(
    '--backend',
)
parser.add_argument(
    '--machines',
    nargs='+',
    default=[]
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


if args.loglevel is None:
    loglevel = logging.INFO
else:
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: %s' % loglevel)

logging.getLogger().setLevel(loglevel)

logging.debug(args)



from dumpdb import (
    dumpdb,
    loaddb
)
from maps import (
    map,
    apply
)
from plot import (
    plot,
    show
)


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
