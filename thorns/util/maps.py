#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

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
import imp
import functools

logger = logging.getLogger(__name__)


class _FuncWrap(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, data):
        result = _apply_data(self.func, data)
        return result



def _func_wrap(func):
    @functools.wraps(func)
    def wrap(data):
        result = _apply_data(func, data)
        return result

    return wrap



def _apply_data(func, data):

    start = time.time()
    ans = func(**data)
    dt = time.time() - start

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



def _isolated_serial_map(func, iterable, cfg):


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
        yield result.get()




def _playdoh_map(func, iterable, cfg):

    import playdoh

    wrap = _func_wrap(func)

    jobrun = playdoh.map_async(
        wrap,
        iterable,
        machines=cfg['machines'],
        codedependencies=cfg['dependencies'],
    )
    results = jobrun.get_results()

    for result in results:
        yield result



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

        # TODO: fix the bug with "\n" in the source code.  Trying to
        # escape the backslash, but does not work here.  Working test
        # file in projects/python_demos/escaping
        # code = code.replace("\\", "\\\\")
        # code = code.replace("\'", "\\\'")
        # code = code.replace("\"", "\\\"")
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

    msg = "{func_name:<22} [{bar}]  {loaded}/{processed}/{remaining}  {time}".format(
        loaded=status['loaded'],
        processed=status['processed'],
        remaining=(status['all']-status['loaded']-status['processed']),
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
        print(msg)






def _get_options(backend, cache, dependencies):

    cfg = {}

    if 'MRmap' in os.environ:
        cfg['backend'] = os.environ['MRmap']
    elif backend is not None:
        cfg['backend'] = backend
    else:
        cfg['backend'] = 'serial'


    if 'MRmachines' in os.environ:
        # TODO: must be parsed
        cfg['machines'] = os.environ['MRmachines']
    else:
        cfg['machines'] = []


    if 'MRdependencies' in os.environ:
        # TODO: must be parsed
        cfg['dependencies'] = os.environ['MRdependencies']
    elif dependencies is not None:
        cfg['dependencies'] = dependencies
    else:
        cfg['dependencies'] = []


    if 'MRcache' in os.environ:
        cfg['cache'] = os.environ['MRcache']
    elif cache is not None:
        cfg['cache'] = cache
    else:
        cfg['cache'] = 'yes'


    return cfg






def apply(func, workdir='work', **kwargs):

    results = map(func, [kwargs], workdir=workdir)
    result = list(results)[0]

    return result




def map(
        func,
        iterable,
        backend='serial',
        cache='yes',
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
        The function to be applied to the data
    iterable : list of dicts
        Each dict is applied to the func. The keys of the dicts should
        correspond to the parameters of the func.
    cache : bool or {'yes', 'no', 'redo'}
        If True, each result is loaded instead calculated again.

    TODO: finish the docstring

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
    for args in iterable:
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


    if cfg['backend'] == 'serial':
        results = _serial_map(func, todos, cfg)
    elif cfg['backend'] in ('multiprocessing', 'm'):
        results = _multiprocessing_map(func, todos, cfg)
    elif cfg['backend'] == 'playdoh':
        results = _playdoh_map(func, todos, cfg)
    elif cfg['backend'] in ('ipython', 'ipcluster'):
        results = _ipython_map(func, todos, cfg)
    elif cfg['backend'] == 'serial_isolated':
        results = _isolated_serial_map(func, todos, cfg)
    else:
        raise RuntimeError("Unknown map() backend: {}".format(cfg['backend']))


    answers = []
    for how,fname in zip(hows,cache_files):

        _publish_status(status, 'file', func_name=func.func_name)
        if how == 'load':
            result = _load_cache(fname)
            status['loaded'] += 1

        elif how == 'process':
            result = next(results)
            status['processed'] += 1

            if cfg['cache'] in ('yes', 'refresh', 'redo'):
                _dump_cache(result, fname)

        else:
            raise RuntimeError("Should never reach this point.")

        ans,dt = result
        status['times'].append(dt)

        answers.append(ans)

    _publish_status(status, 'file', func_name=func.func_name)
    _publish_status(status, 'stdout', func_name=func.func_name)

    return(answers)
