#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"


import cPickle as pickle
import hashlib
import os
import gzip


class _FuncWrap(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, data):
        if isinstance(data, tuple):
            result = self.func(*data)

        elif isinstance(data, dict):
            result = self.func(**data)

        else:
            print data
            raise RuntimeError, "Arguments must be stored as tuple or dict: {}".format(type(data))

        return result





def _calc_pkl_name(obj, cachedir):
    pkl = pickle.dumps(obj, -1)
    h = hashlib.sha1(pkl).hexdigest()

    pkl_name = os.path.join(
        cachedir,
        h + '.pkl.gz'
    )

    return pkl_name


def _load_cache(fname):
    with gzip.open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def _dump_cache(obj, fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print "dumping:", fname

    tmp_fname = fname + ".tmp"
    f = gzip.open(tmp_fname, 'wb', compresslevel=9)
    pickle.dump(obj, f, -1)
    os.rename(tmp_fname, fname)




def _serial_map(func, iterable):

    wrap = _FuncWrap(func)

    for i,args in iterable:
        result = wrap(args)
        yield i,result



def _multiprocessing_map(func, iterable):

    import multiprocessing

    wrap = _FuncWrap(func)

    pool = multiprocessing.Pool()


    results = []
    idx = []
    for i,args in iterable:
        results.append( pool.apply_async(wrap, (args,)) )
        idx.append( i )


    for i,result in zip(idx,results):
        yield i,result.get()






def map(func, iterable, backend='serial', cachedir='tmp/cache'):


    todos = []
    done = []
    for i,args in enumerate(iterable):
        fname = _calc_pkl_name(args, cachedir)

        if os.path.exists(fname):
            done.append( (i, _load_cache(fname)) )

        else:
            todos.append( (i, args) )



    if backend == 'serial':
        results = _serial_map(func, todos)
    elif backend == 'multiprocessing':
        results = _multiprocessing_map(func, todos)
    else:
        raise RuntimeError, "Unknown map() backend: {}".format(backend)


    for todo,result in zip(todos,results):
        i,args = todo
        fname = _calc_pkl_name(args, cachedir)
        _dump_cache(result[1], fname)
        done.append(result)


    done.sort()
    done = [d for i,d in done]


    return done
