#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"


import cPickle as pickle
import hashlib
import os
from itertools import izip
import argparse


class _FuncWrap(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, data):
        result = _apply_data(self.func, data)
        return result


def _func_wrap(func):
    def wrap(data):
        result = _apply_data(func, data)
        return result

    return wrap



def _apply_data(func, data):
    if isinstance(data, tuple):
        result = func(*data)

    elif isinstance(data, dict):
        result = func(**data)

    else:
        print(data)
        raise RuntimeError, "Arguments must be stored as tuple or dict: {}".format(type(data))

    return result




def _calc_pkl_name(obj, cachedir):
    pkl = pickle.dumps(obj, -1)
    h = hashlib.sha1(pkl).hexdigest()

    pkl_name = os.path.join(
        cachedir,
        h + '.pkl'
    )

    return pkl_name


def _load_cache(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def _dump_cache(obj, fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    print("MAP: dumping", fname)

    tmp_fname = fname + ".tmp"
    f = open(tmp_fname, 'wb')
    pickle.dump(obj, f, -1)
    os.rename(tmp_fname, fname)




def _serial_map(func, iterable, opts):

    wrap = _FuncWrap(func)

    for i,args in iterable:
        result = wrap(args)
        yield i,result



def _multiprocessing_map(func, iterable, opts):

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



def _playdoh_map(func, iterable, opts):

    import playdoh

    wrap = _func_wrap(func)

    idx,args = zip(*iterable)

    jobrun = playdoh.map_async(
        wrap,
        args,
        machines=opts['machines']
    )
    results = jobrun.get_results()

    for i,result in zip(idx,results):
        yield i,result



def _publish_progress(progress):
    dirname = 'work'
    fname = os.path.join(dirname, 'status_' + str(os.getpid()))

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    bar = "#" * progress['done'] + "." * (progress['all'] - progress['done'])
    msg = "{} / {}\n\n{}\n".format(
        str(progress['done']),
        str(progress['all']),
        bar
    )


    f = open(fname, 'w')
    f.write(msg)
    f.close()



def _get_options(backend='serial'):

    opts = {}

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--map-backend',
        dest='map_backend',
        nargs=1
    )

    parser.add_argument(
        '--machines',
        dest='machines',
        nargs='+',
        default=[]
    )


    ns = parser.parse_known_args()[0]
    print(ns)

    if ns.map_backend is None:
        opts['map_backend'] = backend
    else:
        opts['map_backend'] = ns.map_backend[0]

    if ns.machines is not None:
        opts['machines'] = ns.machines



    return opts




def map(func, iterable, backend='serial', cachedir='work/map_cache'):

    opts = _get_options(backend)

    status = {'done':0, 'all':0}
    todos = []
    done = []
    for i,args in enumerate(iterable):
        fname = _calc_pkl_name(args, cachedir)

        if os.path.exists(fname):
            print("MAP: loading", fname)
            done.append( (i, _load_cache(fname)) )
            status['done'] += 1
            status['all'] += 1

        else:
            todos.append( (i, args) )
            status['all'] += 1


    _publish_progress(status)


    if opts['map_backend'] == 'serial':
        results = _serial_map(func, todos, opts)
    elif opts['map_backend'] == 'multiprocessing':
        results = _multiprocessing_map(func, todos, opts)
    elif opts['map_backend'] == 'playdoh':
        results = _playdoh_map(func, todos, opts)
    else:
        raise RuntimeError, "Unknown map() backend: {}".format(opts['map_backend'])


    for todo,result in izip(todos,results):
        i,args = todo
        fname = _calc_pkl_name(args, cachedir)
        _dump_cache(result[1], fname)
        done.append(result)

        status['done'] += 1

        _publish_progress(status)


    done.sort()
    done = [d for i,d in done]


    return done
