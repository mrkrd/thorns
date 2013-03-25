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
import socket
import subprocess
import multiprocessing
import shutil
import tempfile
import string

import marlab as mr

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




def _calc_pkl_name(obj, func, cachedir):
    src = inspect.getsource(func)

    pkl = pickle.dumps((obj, src), -1)
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



def _serial_proc_map(func, iterable, cfg):


    for args in iterable:
        dirname = tempfile.mkdtemp()
        fname = os.path.join(
            dirname,
            'mar_maps_socket'
        )
        p = subprocess.Popen(
            ['python', '-m', 'mr.run_func', fname]
        )

        module_name = inspect.getfile(func)
        func_name = func.func_name
        data = (module_name, func_name, args)

        ### make a socket
        s = socket.socket(socket.AF_UNIX)
        s.bind(fname)
        s.listen(1)
        conn, addr = s.accept()


        ### send the function and data to the child
        f = conn.makefile('wb')
        pickle.dump(data, f, -1)
        f.close()


        ### receive results from the child
        f = conn.makefile('rb')
        result = pickle.load(f)
        f.close()


        ### cleanup
        conn.close()
        s.close()
        shutil.rmtree(dirname)


        yield result




def _multiprocessing_map(func, iterable, cfg):

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
    # fname = os.path.abspath(fname)

    rc = Client()
    logger.info("IPython engine IDs: {}".format(rc.ids))


    # print(fname)
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




def _publish_status(status, where='stdout'):


    name = os.path.splitext(
        os.path.basename(sys.argv[0])
    )[0]

    ### Bar
    bar = (
        "".join(status['bar']) +
        "." * (status['all'] - status['loaded'] - status['processed'])
    )

    ### Histogram
    histogram = ""
    hist,edges = np.histogram(
        status['times'],
        bins=10,
    )
    for h,e in zip(hist,edges):
        dt = datetime.timedelta(seconds=e)
        if hist.max() == 0:
            lenght = 0
        else:
            lenght = h / hist.max() * 20
        row = " {dt}  {h:>4} |{s:<20}\n".format(dt=dt, s='|'*int(lenght), h=h)
        histogram += row


    msg = """
{bar}

Loaded (O): {loaded}
Processed (#): {processed}

{histogram}
--------------------
Time: {time}

""".format(
    all=status['all'],
    loaded=status['loaded'],
    processed=status['processed'],
    bar=bar,
    histogram=histogram,
    time=datetime.timedelta(seconds=(time.time() - status['start_time']))
)



    if where == 'file':
        dirname = status['dir']
        fname = os.path.join(dirname, 'status_' + name)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(fname, 'w') as f:
            f.write(msg)


    elif where == 'stdout':
        print(msg)






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




def apply(func, workdir='work', **kwargs):

    results = map(func, [kwargs], workdir=workdir)
    result = list(results)[0]

    return result




def map(func, iterable, backend='serial', cache='yes', workdir='work'):

    cfg = _get_options(
        backend=backend,
        cache=cache
    )

    status = {
        'all':0,
        'loaded':0,
        'processed':0,
        'bar': [],
        'times':[],
        'start_time':time.time(),
        'dir': workdir
    }

    cachedir = os.path.join(workdir, 'map_cache')

    cache_files = []
    hows = []
    todos = []
    for args in iterable:
        fname = _calc_pkl_name(args, func, cachedir)
        cache_files.append(fname)

        status['all'] += 1

        if (cfg['cache'] == 'yes') and os.path.exists(fname):
            hows.append('load')
        else:
            hows.append('process')
            todos.append(args)


    if cfg['backend'] == 'serial':
        results = _serial_map(func, todos, cfg)
    elif cfg['backend'] in ('multiprocessing', 'm'):
        results = _multiprocessing_map(func, todos, cfg)
    elif cfg['backend'] == 'playdoh':
        results = _playdoh_map(func, todos, cfg)
    elif cfg['backend'] == 'ipython':
        results = _ipython_map(func, todos, cfg)
    elif cfg['backend'] == 'serial_proc':
        results = _serial_proc_map(func, todos, cfg)
    else:
        raise RuntimeError("Unknown map() backend: {}".format(cfg['backend']))



    for how,fname in zip(hows,cache_files):

        _publish_status(status, 'file')
        if how == 'load':
            result = _load_cache(fname)
            status['loaded'] += 1
            status['bar'].append('O')

        elif how == 'process':
            result = next(results)
            status['processed'] += 1
            status['bar'].append('#')

            if cfg['cache'] in ('yes', 'refresh'):
                _dump_cache(result, fname)

        else:
            raise RuntimeError("Should never reach this point.")


        ans,dt = result
        status['times'].append(dt)

        yield ans

    _publish_status(status, 'file')
    _publish_status(status, 'stdout')
