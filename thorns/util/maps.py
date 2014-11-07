#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements map function with various backends and
caching.

"""
from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals

__author__ = "Marek Rudnicki"
__copyright__ = "Copyright 2014, Marek Rudnicki, Jörg Encke"
__license__ = "GPLv3+"


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
import imp
import functools
import pandas as pd
import itertools
import warnings

logger = logging.getLogger('thorns')

is_inside_map = False


class _FuncWrap(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, data):
        func = self.func

        global is_inside_map

        is_inside_map = True
        start = time.time()
        ans = func(**data)
        dt = time.time() - start
        is_inside_map = False

        return ans,dt



def _pkl_name(obj, func, cachedir):
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


def _dump_cache(fname, obj):
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



def _serial_isolated_map(func, iterable, cfg):

    for args in iterable:
        dirname = tempfile.mkdtemp()
        fname = os.path.join(
            dirname,
            'mar_maps_socket'
        )
        p = subprocess.Popen(
            ['python', '-m', 'thorns.util.run_func', fname]
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
        yield result.get(9999999)





def _ipython_map(func, iterable, cfg):

    from IPython.parallel import Client

    rc = Client()
    rc[:].clear()



    ### Make modules for all dependencies on the engines
    for dep in cfg['dependencies']:
        mod_name = os.path.splitext(
            os.path.basename(dep)
        )[0]


        with open(dep) as f:
            code = f.read()

        code = code.encode('string_escape')

        rc[:].execute(
"""
import imp
import sys

_mod = imp.new_module('{mod_name}')
sys.modules['{mod_name}'] = _mod

exec '''{code}''' in _mod.__dict__

del _mod
""".format(code=code, mod_name=mod_name),
            block=True
        )




    ### Make sure all definitions surrounding the func are present on
    ### the engines (evaluate the code from the file of the func)
    fname = inspect.getfile(func)
    with open(fname) as f:
        code = f.read()


    logger.info("IPython engine IDs: {}".format(rc.ids))


    ## Need to escape all ' and " in order to embed the code into
    ## execute string
    # code = code.replace("\'", "\\\'")
    # code = code.replace("\"", "\\\"")

    code = code.encode('string_escape')


    ## The trick with `exec in {}' is done because we want to avoid
    ## executing `__main__'
    rc[:].execute(
"""
_tmp_dict = dict()
exec '''{code}''' in _tmp_dict
globals().update(_tmp_dict)
del _tmp_dict
""".format(code=code),
        block=True
    )
    # status.wait()

    # res = rc[:].apply(dir)
    # print(res.get())

    wrap = _FuncWrap(func)
    pool = rc.load_balanced_view()

    results = []
    for args in iterable:
        results.append( pool.apply_async(wrap, args) )


    for result in results:
        yield result.get()




def _publish_status(status, where='stdout', func_name=""):


    name = os.path.splitext(
        os.path.basename(sys.argv[0])
    )[0]

    ### Bar
    bar_len = 20

    bar = (
        "O" * int(round(bar_len * status['loaded']/status['all'])) +
        "#" * int(round(bar_len * status['processed']/status['all']))
    )
    bar += "." * (bar_len - len(bar))

    seconds = time.time() - status['start_time']

    msg = "[{bar}]  {loaded}|{processed}/{all}  {time}  ({func_name})".format(
        loaded=status['loaded'],
        processed=status['processed'],
        all=status['all'],
        bar=bar,
        time=datetime.timedelta(seconds=seconds),
        func_name=func_name
    )



    if where == 'file':
        dirname = status['dir']
        fname = os.path.join(dirname, 'status_' + name)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(fname, 'w') as f:
            f.write(msg)

    elif where == 'stdout':
        sys.stderr.write(msg)
        sys.stderr.write('\n')
        sys.stderr.flush()

    elif where == 'title':
        sys.stderr.write("\033]2;{}\007\r".format(msg))
        sys.stderr.flush()

    elif (where == 'notify') and (seconds > 5):
        try:
            import pynotify
            pynotify.init(name)
            notice = pynotify.Notification(name, msg)
            notice.show()
        except Exception:
            ### ImportError, GError (?)
            pass



def _get_options(backend, cache, dependencies):

    cfg = {}
    global is_inside_map

    if is_inside_map:
        cfg['backend'] = 'serial'
    elif backend is not None:
        cfg['backend'] = backend
    elif 'THmap' in os.environ:
        cfg['backend'] = os.environ['THmap']
    else:
        cfg['backend'] = 'serial'


    if 'THmachines' in os.environ:
        # TODO: must be parsed
        cfg['machines'] = os.environ['THmachines']
    else:
        cfg['machines'] = []


    if 'THdependencies' in os.environ:
        # TODO: must be parsed
        cfg['dependencies'] = os.environ['THdependencies']
    elif dependencies is not None:
        cfg['dependencies'] = dependencies
    else:
        cfg['dependencies'] = []


    if cache is not None:
        cfg['cache'] = cache
    elif 'THcache' in os.environ:
        cfg['cache'] = os.environ['THcache']
    else:
        cfg['cache'] = 'yes'


    cfg['publish_status'] = not is_inside_map

    return cfg






def cache(func, workdir='work'):
    """Wrap a function and cache its output.

    """
    @functools.wraps(func)
    def wrap(**kwargs):

        cachedir = os.path.join(workdir, 'map_cache')

        fname = _pkl_name(kwargs, func, cachedir)

        if os.path.exists(fname):
            result = _load_cache(fname)
        else:
            result = func(**kwargs)
            _dump_cache(fname, result)

        return result

    return wrap




def map(
        func,
        space,
        backend=None,
        cache=None,
        workdir='work',
        dependencies=None,
        kwargs=None
):
    """Apply func to every item of iterable and return a list of the
    results.  This map supports multiple backends, e.g. 'serial',
    'multiprocessing', 'ipcluster'.


    Parameters
    ----------
    func : function
        The function to be applied to the data.
    space : (list of dicts) or (dict of lists)
        Parameter space, where the keys of the dictonary(s) correspond
        to the keyward arguments of the function.
        In the case of a list of dicts, each entry of the list is applied
        to the function.
        In the case of a dict of lists, the parameter space is built
        by using all possible permutations of the list entries.
    backend : {'serial', 'ipcluster', 'multiprocessing', 'serial_isolated'}
        Choose a backend for the map.
    cache : bool or {'yes', 'no', 'redo'}
        If True, each result is loaded instead calculated again.
    workdir : str, optional
        Directory in which to store cache.
    dependencies : list, optional
        List of python files that will be imported on the remote site
        before executing the `func`.
    kwargs : dict, optional
        Extra parameters for the `func`.


    Returns
    -------
    pd.DataFrame
        Table with parameters (MultiIndex) and results.

    """
    cfg = _get_options(
        backend=backend,
        cache=cache,
        dependencies=dependencies
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
    all_kwargs_names = set()



    ### Convert a dict of lists into a list of dicts
    if isinstance(space, dict):
        all_values = itertools.product(*space.values())
        keys = space.keys()
        iterable = [dict(zip(keys, values)) for values in all_values]
    else:
        iterable = space



    ### Go through the parameter space and check what should be
    ### calculated (todos) and what recalled from the cache
    for args in iterable:
        all_kwargs_names.update(args)

        args = dict(args)
        if kwargs is not None:
            args.update(kwargs)

        fname = _pkl_name(args, func, cachedir)
        cache_files.append(fname)

        status['all'] += 1

        if (cfg['cache'] == 'yes') and os.path.exists(fname):
            hows.append('load')
        else:
            hows.append('process')
            todos.append(args)



    ### Submit the parameter space to one of the backends
    if cfg['backend'] == 'serial':
        results = _serial_map(func, todos, cfg)
    elif cfg['backend'] in ('multiprocessing', 'm'):
        results = _multiprocessing_map(func, todos, cfg)
    elif cfg['backend'] in ('ipython', 'ipcluster'):
        results = _ipython_map(func, todos, cfg)
    elif cfg['backend'] == 'serial_isolated':
        results = _serial_isolated_map(func, todos, cfg)
    else:
        raise RuntimeError("Unknown map() backend: {}".format(cfg['backend']))



    ### Generate reults by either using cache (how == 'load') or
    ### calculate using func (how == 'process')
    answers = []
    for how,fname in zip(hows,cache_files):

        if cfg['publish_status']:
            _publish_status(status, 'file', func_name=func.func_name)
            _publish_status(status, 'title', func_name=func.func_name)


        if how == 'load':
            result = _load_cache(fname)
            status['loaded'] += 1

        elif how == 'process':
            result = next(results)
            status['processed'] += 1

            if cfg['cache'] in ('yes', 'refresh', 'redo'):
                _dump_cache(fname, result)

        else:
            raise RuntimeError("Should never reach this point.")

        ans,dt = result
        status['times'].append(dt)

        answers.append(ans)


    ### Prepare DataFrame output
    iterable = pd.DataFrame(iterable)
    answers = pd.DataFrame(answers)

    out = pd.concat((iterable, answers), axis=1)
    out = out.set_index(list(all_kwargs_names))


    if cfg['publish_status']:
        _publish_status(status, 'file', func_name=func.func_name)
        _publish_status(status, 'title', func_name=func.func_name)
        _publish_status(status, 'stdout', func_name=func.func_name)
        _publish_status(status, 'notify', func_name=func.func_name)



    return out
