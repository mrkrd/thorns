#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-03-05 00:10:52 marek>

# Description:

from __future__ import division
import numpy as np
from time import sleep
import sys

import thorns.mproc as mp

fname = 'GLOBAL VAR'

def sender(task_queue):
    for i in range(10):
        task_queue.put(i)

def worker(task_queue, done_queue):
    # fname = sys.argv
    print fname
    for i in iter(task_queue.get, 'STOP'):
        print 'working on:', i
        r = i/2
        sleep(r)
        done_queue.put(r)

def receiver(done_queue):
    for i in iter(done_queue.get, 'STOP'):
        print 'received:', i

def main():
    mp.run_workers(sender, worker, receiver, nproc=2)

if __name__ == "__main__":
    main()
