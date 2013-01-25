from __future__ import division

__author__ = "Marek Rudnicki"


import argparse
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log',
    dest='loglevel'
)
parser.add_argument(
    '--backend',
    dest='backend'
)
parser.add_argument(
    '--machines',
    dest='machines',
    nargs='+',
    default=[]
)
parser.add_argument(
    '--cache',
    dest='cache'
)
parser.add_argument(
    '--plot',
    dest='plot',
    nargs='?',
    const='show'
)


args = parser.parse_known_args()[0]



if args.loglevel is None:
    loglevel = logging.INFO
else:
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: %s' % loglevel)

logging.getLogger().setLevel(loglevel)

logging.debug(args)


import thorns
import waves

from dumpdb import (
    dumpdb,
    loaddb
)
from maps import (
    map,
    apply
)
from plot import plot


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
