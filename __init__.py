from __future__ import division

__author__ = "Marek Rudnicki"


import thorns
import waves
import dumpdb



def _func_wrap(data):
    global _func

    if isinstance(data, tuple):
        result = _func(*data)

    elif isinstance(data, dict):
        result = _func(**data)

    return result




def map(func, iterable):

    backend = 'multiprocessing'

    if backend == 'multiprocessing':
        import multiprocessing

        global _func
        _func = func

        pool = multiprocessing.Pool()
        results = pool.map(_func, iterable)

    elif backend == 'joblib':

        pass


    return results

