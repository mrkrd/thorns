"""Utilities.

"""

from thorns.util.dumpdb import (
    dumpdb,
    loaddb,
    get_store,
)

from thorns.util.maps import (
    map,
    apply,
)


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
