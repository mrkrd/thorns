from __future__ import division

__author__ = "Marek Rudnicki"


import thorns
import waves
import dumpdb


import functools




class _MapWrap(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, data):
        if isinstance(data, tuple):
            result = self.func(*data)

        elif isinstance(data, dict):
            result = self.func(**data)

        else:
            raise RuntimeError, "Arguments must be stored as tuple or dict."

        return result





def map(func, iterable, backend='multiprocessing'):


    if backend == 'multiprocessing':
        import multiprocessing

        wrapped = _MapWrap(func)


        pool = multiprocessing.Pool()
        results = pool.map(wrapped, iterable)


    elif backend == 'joblib':
        import joblib


        def func_args_kwargs(func, data):
            if isinstance(data, tuple):
                out = (func, data, {})

            elif isinstance(data, dict):
                out = (func, (), data)

            else:
                raise RuntimeError, "Arguments must be stored as tuple or dict."

            return out



        results = joblib.Parallel(n_jobs=-1, verbose=5)(
            func_args_kwargs(func, i) for i in iterable
        )


    else:
        raise RuntimeError, "Unknown map() backend: {}".format(backend)



    return results

