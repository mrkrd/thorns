#!/usr/bin/env python

# Author: Marek Rudnicki
# Time-stamp: <2010-03-04 18:56:23 marek>

# Description:

from __future__ import division
import numpy as np

import multiprocessing


def run_workers(sender, worker, receiver, nproc=None):
    if nproc is None:
        nproc = multiprocessing.cpu_count()

    task_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    # Receiver process
    rcv_proc = multiprocessing.Process(target=receiver,
                                       args=(done_queue,))
    rcv_proc.start()


    # Worker processes
    work_procs = []
    for i in range(nproc):
        w = multiprocessing.Process(target=worker,
                                    args=(task_queue, done_queue))
        w.start()
        work_procs.append(w)


    # Sender process
    snd_proc = multiprocessing.Process(target=sender,
                                       args=(task_queue,))
    snd_proc.start()
    snd_proc.join()


    # Stop workers
    for i in range(nproc):
        task_queue.put('STOP')


    # Wait for workers to stop
    for w in work_procs:
        w.join()


    # Stop receiver process
    done_queue.put('STOP')



run_ants = run_workers



def main():
    pass

if __name__ == "__main__":
    main()
