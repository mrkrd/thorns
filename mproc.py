#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import numpy as np
import multiprocessing
import inspect

def run_workers(sender, worker, receiver, nproc=None, config=None):
    """
    config: object that will be passed to each process if each process
    function has it declared

    """
    if nproc is None:
        nproc = multiprocessing.cpu_count()

    task_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    # Receiver process
    nargs = len(inspect.getargspec(receiver).args)
    if nargs == 1:
        args = (done_queue,)
    elif nargs == 2:
        args = (done_queue, config)
    else:
        assert False, "Too many arguments for receiver()"
    rcv_proc = multiprocessing.Process(target=receiver,
                                       args=args)
    rcv_proc.start()


    # Worker processes
    work_procs = []
    nargs = len(inspect.getargspec(worker).args)
    if nargs == 2:
        args = (task_queue, done_queue)
    elif nargs == 3:
        args = (task_queue, done_queue, config)
    else:
        assert False, "Wrong number of arguments in worker()"
    for i in range(nproc):
        w = multiprocessing.Process(target=worker,
                                    args=args)
        w.start()
        work_procs.append(w)


    # Sender process
    nargs = len(inspect.getargspec(sender).args)
    if nargs == 1:
        args = (task_queue,)
    elif nargs == 2:
        args = (task_queue, config)
    else:
        assert False, "Too many arguments for sender()"
    snd_proc = multiprocessing.Process(target=sender,
                                       args=args)
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

    # Wait receiver process it to be done
    rcv_proc.join()

run_ants = run_workers



def main():
    pass

if __name__ == "__main__":
    main()
