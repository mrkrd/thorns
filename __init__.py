from __future__ import division

__author__ = "Marek Rudnicki"


import thorns
import waves
import dumpdb


import functools


def _func_wrap(data):
    global _func

    if isinstance(data, tuple):
        result = _func(*data)

    elif isinstance(data, dict):
        result = _func(**data)

    return result



def _decor(func):
    @functools.wraps(func)
    def wrapped(data):
        if isinstance(data, tuple):
            result = func(*data)

        elif isinstance(data, dict):
            result = func(**data)

        return result

    return wrapped



def map(func, iterable):

    backend = 'm'

    global _func
    _func = func


    if backend == 'm':
        import multiprocessing

        wrapped = _decor(func)

        pool = multiprocessing.Pool()
        results = pool.map(wrapped, iterable)


    if backend == 'multiprocessing':
        import multiprocessing


        pool = multiprocessing.Pool()
        results = pool.map(_func_wrap, iterable)

    elif backend == 'joblib':
        import joblib

        results = joblib.Parallel(n_jobs=-1)(
            _func_wrap(i) for i in iterable
        )


    else:
        raise RuntimeError, "Unknown map() backend: {}".format(backend)



    return results

