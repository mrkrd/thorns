#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"


import cPickle as pickle
import hashlib
import os
from itertools import izip
import argparse
import time
import datetime
import numpy as np
import sys


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

    start = time.time()
    if isinstance(data, tuple):
        result = func(*data)

    elif isinstance(data, dict):
        result = func(**data)

    else:
        result = func(data)

    dt = time.time() - start

    return result, dt




def _calc_pkl_name(obj, cachedir):
    pkl = pickle.dumps(obj, -1)
    h = hashlib.sha1(pkl).hexdigest()

    pkl_name = os.path.join(
        cachedir,
        h + '.pkl'
    )

    return pkl_name


def _load_cache(fname):
    print("MAP: loading", fname)
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




def _serial_map(func, iterable, cfg):

    wrap = _FuncWrap(func)

    for i,args in iterable:
        result,dt = wrap(args)
        yield i,result,dt



def _multiprocessing_map(func, iterable, cfg):

    import multiprocessing

    wrap = _FuncWrap(func)

    pool = multiprocessing.Pool()


    results = []
    idx = []
    for i,args in iterable:
        results.append( pool.apply_async(wrap, (args,)) )
        idx.append( i )


    for i,result in izip(idx,results):
        ans,dt = result.get()
        yield i,ans,dt



def _playdoh_map(func, iterable, cfg):

    import playdoh

    wrap = _func_wrap(func)

    idx,args = zip(*iterable)

    jobrun = playdoh.map_async(
        wrap,
        args,
        machines=cfg['machines']
    )
    results = jobrun.get_results()

    for i,result in zip(idx,results):
        ans,dt = result
        yield i,ans,dt







def _publish_progress(status):
    dirname = 'work'
    sufix = os.path.splitext(
        os.path.basename(sys.argv[0])
    )[0]
    fname = os.path.join(dirname, 'status_' + sufix)


    if not os.path.exists(dirname):
        os.makedirs(dirname)

    bar = (
        "L" * status['loaded'] +
        "P" * status['processed'] +
        "." * (status['all'] - status['loaded'] - status['processed'])
    )
    msg = "{} + {} / {}\n\n{}\n".format(
        str(status['loaded']),
        str(status['processed']),
        str(status['all']),
        bar
    )


    f = open(fname, 'w')
    f.write(msg)
    f.close()



def _print_summary(status):

    print()
    print("Summary:")
    print()

    if status['times']:
        hist,edges = np.histogram(
            status['times'],
            bins=10,
        )

        for h,e in zip(hist,edges):
            dt = datetime.timedelta(seconds=e)
            height = h / hist.max() * 20
            print(dt, '|', '#'*int(height))

        print()

    print("Mapping time: {}".format(
        datetime.timedelta(seconds=(time.time() - status['start_time']))
    ))
    print()
    print("All: {}".format(status['all']))
    print("Loaded: {}".format(status['loaded']))
    print("Processed: {}".format(status['processed']))
    print()




def _get_options(backend, cache):

    cfg = {}

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--map-backend',
        dest='backend',
        nargs=1
    )
    parser.add_argument(
        '--machines',
        dest='machines',
        nargs='+',
        default=[]
    )

    cache_args = parser.add_mutually_exclusive_group()
    cache_args.add_argument(
        '--map-cache',
        dest='cache',
        action='store_const',
        const='yes',
        # default='yes'
    )
    cache_args.add_argument(
        '--no-map-cache',
        dest='cache',
        action='store_const',
        const='no',
        # default='yes'
    )
    cache_args.add_argument(
        '--refresh-map-cache',
        dest='cache',
        action='store_const',
        const='refresh',
        # default='yes'
    )


    ns = parser.parse_known_args()[0]
    print(ns)


    if ns.backend is None:
        cfg['backend'] = backend
    else:
        cfg['backend'] = ns.backend[0]


    if ns.machines is not None:
        cfg['machines'] = ns.machines


    if ns.cache is None:
        cfg['cache'] = cache
    else:
        cfg['cache'] = ns.cache

    return cfg






def map(func, iterable, backend='serial', cache='yes', cachedir='work/map_cache'):

    cfg = _get_options(
        backend=backend,
        cache=cache
    )

    status = {
        'all':0,
        'loaded':0,
        'processed':0,
        'times':[],
        'start_time':time.time()
    }

    todos = []
    done = []
    for i,args in enumerate(iterable):
        fname = _calc_pkl_name(args, cachedir)

        if (cfg['cache'] == 'yes') and os.path.exists(fname):
            done.append( (i, _load_cache(fname)) )
            status['all'] += 1
            status['loaded'] += 1

        else:
            todos.append( (i, args) )
            status['all'] += 1


    _publish_progress(status)


    if cfg['backend'] == 'serial':
        results = _serial_map(func, todos, cfg)
    elif cfg['backend'] == 'multiprocessing':
        results = _multiprocessing_map(func, todos, cfg)
    elif cfg['backend'] == 'playdoh':
        results = _playdoh_map(func, todos, cfg)
    else:
        raise RuntimeError, "Unknown map() backend: {}".format(cfg['backend'])


    for todo,result in izip(todos,results):
        i,args = todo
        j,r,dt = result
        assert i == j

        done.append( (i,r) )

        if cfg['cache'] in ('yes', 'refresh'):
            fname = _calc_pkl_name(args, cachedir)
            _dump_cache(result[1], fname)


        status['processed'] += 1
        status['times'].append(dt)

        _publish_progress(status)


    done.sort()
    done = [d for i,d in done]

    _print_summary(status)

    return done
