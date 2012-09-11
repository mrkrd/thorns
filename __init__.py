from __future__ import division

__author__ = "Marek Rudnicki"


import thorns
import waves
import dumpdb


import inspect
import sys
import cPickle as pickle
import hashlib





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





def _calc_pkl_name(obj):
    pkl = pickle.dumps(obj, -1)
    h = hashlib.sha1(pkl).hexdigest()

    pkl_name = os.path.join(
        'tmp',
        'cache',
        h + '.pkl.gz'
    )

    return pkl_name


def _load_cache(fname):
    with gzip.open(fname, 'rb') as f:
        data = pickle.load(f)
    return data



class Mapper(object):
    def __init__(self, func, backend):
        self.func = func
        self.backend = backend
        self.results = None

    def apply(self, i, args):
        if self.backend == 'serial':
            if self.results is None:
                self.results = []






def map(func, iterable, backend='serial'):

    arguments = []
    results = []
    for i,args in enumerate(iterable):

        fname = _calc_pkl_name(args)

        if os.path.exists(fname):
            results.append( (i, _load_cache(fname)) )

        else:
            arguments.append( (i, args) )









    if backend == 'serial':
        import __builtin__

        results = __builtin__.map(
            _MapWrap(func),
            iterable
        )

    elif backend == 'multiprocessing':
        import multiprocessing

        pool = multiprocessing.Pool()

        results = pool.map(
            _MapWrap(func),
            iterable,
        )


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



        results = joblib.Parallel(n_jobs=-1, verbose=100)(
            func_args_kwargs(func, i) for i in iterable
        )


    elif backend == 'pp':
        import pp


        def wrap(func, data):
            if isinstance(data, tuple):
                result = func(*data)
            elif isinstance(data, dict):
                result = func(**data)
            else:
                raise RuntimeError, "Arguments must be stored as tuple or dict."
            return result


        job_server = pp.Server( ppservers=("*",), ncpus=0 )

        modules = []
        depfuncs = []
        for k,v in func.func_globals.items():
            if inspect.ismodule(v):
                modules.append( "import " + v.__name__ + " as " + k )

            if inspect.isfunction(v):
                depfuncs.append(v)

        jobs = []
        for i in iterable:
            job = job_server.submit(
                func=wrap,
                args=(func,i),
                modules=tuple(modules),
                depfuncs=tuple(depfuncs),
                globals=func.func_globals
            )
            jobs.append(job)

        results = [job() for job in jobs]

    else:
        raise RuntimeError, "Unknown map() backend: {}".format(backend)



    return results
