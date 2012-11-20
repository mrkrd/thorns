#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"


import cPickle as pickle
import hashlib
import os
import time
import datetime
import numpy as np
import sys
import logging
import inspect

import marlib as mr

logger = logging.getLogger(__name__)


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
        ans = func(*data)

    elif isinstance(data, dict):
        ans = func(**data)

    else:
        ans = func(data)

    dt = time.time() - start

    return ans,dt




def _calc_pkl_name(obj, cachedir):
    pkl = pickle.dumps(obj, -1)
    h = hashlib.sha1(pkl).hexdigest()

    pkl_name = os.path.join(
        cachedir,
        h + '.pkl'
    )

    return pkl_name


def _load_cache(fname):
    logger.info("Loading cache from {}".format(fname))
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def _dump_cache(obj, fname):
    dirname = os.path.dirname(fname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    logger.info("Dumping cache to {}".format(fname))

    tmp_fname = fname + ".tmp"
    with open(tmp_fname, 'wb') as f:
        pickle.dump(obj, f, -1)
    os.rename(tmp_fname, fname)




def _serial_map(func, iterable, cfg):

    wrap = _FuncWrap(func)

    for args in iterable:
        result = wrap(args)
        yield result




def _multiprocessing_map(func, iterable, cfg):

    import multiprocessing

    wrap = _FuncWrap(func)

    pool = multiprocessing.Pool()


    results = []
    for args in iterable:
        results.append( pool.apply_async(wrap, (args,)) )


    for result in results:
        yield result.get()




def _playdoh_map(func, iterable, cfg):

    import playdoh

    wrap = _func_wrap(func)

    jobrun = playdoh.map_async(
        wrap,
        iterable,
        machines=cfg['machines']
    )
    results = jobrun.get_results()

    for result in results:
        yield result



def _ipython_map(func, iterable, cfg):

    from IPython.parallel import Client

    fname = inspect.getfile(func)
    fname = os.path.abspath(fname)

    rc = Client()
    logger.info("IPython engine IDs: {}".format(rc.ids))


    print(fname)
    status = rc[:].run(fname) #, block=True)
    status.wait()

    # res = rc[:].apply(dir)
    # print(res.get())

    wrap = _FuncWrap(func)
    pool = rc.load_balanced_view()

    results = []
    for args in iterable:
        results.append( pool.apply_async(wrap, args) )


    for result in results:
        yield result.get()




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
    msg = "{0} + {1} / {2}\n\n{3}\n".format(
        str(status['loaded']),
        str(status['processed']),
        str(status['all']),
        bar
    )


    with open(fname, 'w') as f:
        f.write(msg)




def _print_summary(status):

    ### Header
    print()
    print("Summary:")
    print()


    ### Histogram
    hist,edges = np.histogram(
        status['times'],
        bins=10,
    )

    for h,e in zip(hist,edges):
        dt = datetime.timedelta(seconds=e)
        lenght = h / hist.max() * 20
        row = "{dt} | {s:<20} | {h}".format(dt=dt, s='#'*int(lenght), h=h)
        print(row)


    ### Counts
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

    if mr.args.backend is None:
        cfg['backend'] = backend
    else:
        cfg['backend'] = mr.args.backend


    if mr.args.machines is not None:
        cfg['machines'] = mr.args.machines


    if mr.args.cache is None:
        cfg['cache'] = cache
    else:
        cfg['cache'] = mr.args.cache

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


    cache_files = []
    hows = []
    todos = []
    for args in iterable:
        fname = _calc_pkl_name(args, cachedir)
        cache_files.append(fname)

        status['all'] += 1

        if (cfg['cache'] == 'yes') and os.path.exists(fname):
            hows.append('load')
        else:
            hows.append('process')
            todos.append(args)



    if cfg['backend'] == 'serial':
        results = _serial_map(func, todos, cfg)
    elif cfg['backend'] == 'multiprocessing':
        results = _multiprocessing_map(func, todos, cfg)
    elif cfg['backend'] == 'playdoh':
        results = _playdoh_map(func, todos, cfg)
    elif cfg['backend'] == 'ipython':
        results = _ipython_map(func, todos, cfg)
    else:
        raise RuntimeError("Unknown map() backend: {}".format(cfg['backend']))



    for how,fname in zip(hows,cache_files):

        _publish_progress(status)
        if how == 'load':
            result = _load_cache(fname)
            status['loaded'] += 1

        elif how == 'process':
            result = next(results)
            status['processed'] += 1

            if cfg['cache'] in ('yes', 'refresh'):
                _dump_cache(result, fname)

        else:
            raise RuntimeError("Should never reach this point.")


        ans,dt = result
        status['times'].append(dt)

        yield ans

    _publish_progress(status)
    _print_summary(status)
