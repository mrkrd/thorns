#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import random
import numpy as np


class Train(object):
    def __init__(self, data=[], **kwargs):
        if isinstance(data, Train):
            self._meta = dict(data._meta)
        else:
            self._meta = dict()

        self._data = np.array(data, dtype=float)
        self._meta.update(kwargs)



    def copy_meta(f):
        """
        This decorator assures that meta data is copied for the output
        train.

        """
        def w(self, *args, **kwargs):
            data = f(self, *args, **kwargs)

            # Copy the meta data from the original train
            if isinstance(data, np.ndarray):
                data = Train(data, **self._meta)

            return data

        return w


    @copy_meta
    def __getslice__(self, *args, **kwargs):
        return self._data.__getslice__(*args, **kwargs)
    def __setslice__(self, *args, **kwargs):
        self._data.__setslice__(*args, **kwargs)

    @copy_meta
    def __getitem__(self, key):
        if isinstance(key, str):
            item = self._meta[key]
        else:
            item = self._data.__getitem__(key)

        return item


    def __setitem__(self, *args, **kwargs):
        self._data.__setitem__(*args, **kwargs)

    def __len__(self):
        return self._data.__len__()

    def __iter__(self):
        return self._data.__iter__()

    def __str__(self):
        return self._data.__str__() + " " + self._meta.__str__()

    def __lt__(self, other):
        return self._data.__lt__(other)
    def __le__(self, other):
        return self._data.__le__(other)
    def __eq__(self, other):
        return self._data.__eq__(other)
    def __ne__(self, other):
        return self._data.__ne__(other)
    def __gt__(self, other):
        return self._data.__gt__(other)
    def __ge__(self, other):
        return self._data.__ge__(other)

    @copy_meta
    def __add__(self, other):
        return self._data.__add__(other)

    @copy_meta
    def __sub__(self, other):
        return self._data.__sub__(other)

    @copy_meta
    def __mul__(self, other):
        return self._data.__mul__(other)

    def __floordiv__(self, other):
        return self._data.__floordiv__(other)
    def __mod__(self, other):
        return self._data.__mod__(other)
    def __divmod__(self, other):
        return self._data.__divmod__(other, *args, **kwargs)
    def __pow__(self, other, *args, **kwargs):
        return self._data.__pow__(other)
    def __lshift__(self, other):
        return self._data.__lshift__(other)
    def __rshift__(self, other):
        return self._data.__rshift__(other)
    def __and__(self, other):
        return self._data.__and__(other)
    def __xor__(self, other):
        return self._data.__xor__(other)
    def __or__(self, other):
        return self._data.__or__(other)
    def __div__(self, other):
        return self._data.__div__(other)
    def __truediv__(self, other):
        return self._data.__truediv__(other)
    def __radd__(self, other):
        return self._data.__radd__(other)
    def __rsub__(self, other):
        return self._data.__rsub__(other)
    def __rmul__(self, other):
        return self._data.__rmul__(other)
    def __rdiv__(self, other):
        return self._data.__rdiv__(other)
    def __rtruediv__(self, other):
        return self._data.__rtruediv__(other)
    def __rfloordiv__(self, other):
        return self._data.__rfloordiv__(other)
    def __rmod__(self, other):
        return self._data.__rmod__(other)
    def __rdivmod__(self, other):
        return self._data.__rdivmod__(other)
    def __rpow__(self, other):
        return self._data.__rpow__(other)
    def __rlshift__(self, other):
        return self._data.__rlshift__(other)
    def __rrshift__(self, other):
        return self._data.__rrshift__(other)
    def __rand__(self, other):
        return self._data.__rand__(other)
    def __rxor__(self, other):
        return self._data.__rxor__(other)
    def __ror__(self, other):
        return self._data.__ror__(other)
    def __iadd__(self, other):
        return self._data.__iadd__(other)
    def __isub__(self, other):
        return self._data.__isub__(other)
    def __imul__(self, other):
        return self._data.__imul__(other)
    def __idiv__(self, other):
        return self._data.__idiv__(other)
    def __itruediv__(self, other):
        return self._data.__itruediv__(other)
    def __ifloordiv__(self, other):
        return self._data.__ifloordiv__(other)
    def __imod__(self, other):
        return self._data.__imod__(other)
    def __ipow__(self, other, *args, **kwargs):
        return self._data.__ipow__(other, *args, **kwargs)
    def __ilshift__(self, other):
        return self._data.__ilshift__(other)
    def __irshift__(self, other):
        return self._data.__irshift__(other)
    def __iand__(self, other):
        return self._data.__iand__(other)
    def __ixor__(self, other):
        return self._data.__ixor__(other)
    def __ior__(self, other):
        return self._data.__ior__(other)
    def __neg__(self):
        return self._data.__neg__(other)
    def __pos__(self):
        return self._data.__pos__(other)
    def __abs__(self):
        return self._data.__abs__(other)
    def __invert__(self):
        return self._data.__invert__(other)
    def __complex__(self):
        return self._data.__complex__(other)
    def __int__(self):
        return self._data.__int__(other)
    def __long__(self):
        return self._data.__long__(other)
    def __float__(self):
        return self._data.__float__(other)


class SpikeTrains(object):
    def __init__(self, spike_trains=[]):
        self._trains = [Train(train) for train in spike_trains]

    def __setslice__(self, *args, **kwargs):
        self._trains.__setslice__(*args, **kwargs)

    def __getslice__(self, *args, **kwargs):
        return self._trains.__getslice__(*args, **kwargs)

    def __len__(self):
        return self._trains.__len__()

    def __setitem__(self, *args, **kwargs):
        self._trains.__setitem__(*args, **kwargs)

    def __getitem__(self, key):

        if isinstance(key, str):
            # String gives us array of attributes
            item = np.array([train[key] for train in self._trains])

        elif isinstance(key, np.ndarray) and (key.dtype is np.dtype('bool')):
            # Indexing using Bool table (a la Numpy)

            assert len(key) == len(self._trains)
            idx = np.where(key)[0]

            item = SpikeTrains()
            for i in idx:
                item.append(self[i])

        else:
            # Integer indexing
            item = self._trains.__getitem__(key)

        return item

    def __iter__(self):
        return self._trains.__iter__()

    def __str__(self):
        s = "[\n"
        for t in self._trains:
            s = s + "  " + str(t) + "\n"
        s += "]"
        return s

    def append(self, train, **kwargs):
        train = Train(train, **kwargs)
        self._trains.append(train)

    def extend(self, L):
        L = [Train(el) for el in L]
        self._trains.extend(L)

    def where(self, **kwargs):
        mask = np.ones(len(self._trains), dtype=np.dtype('bool'))

        for key in kwargs:
            mask = mask & (self[key]==kwargs[key])

        return self[mask]


    def pop(self, random=False, *args):
        if random:
            i = random.randint(0, len(self._trains)-1)
            train = self._trains.pop(i)
        else:
            train = self._trains.pop(*args)

        return train




def main():
    t = Train([1,2,3], cf=12)
    print t['cf']


    st = SpikeTrains()
    st.append([1,2,3], cf=12000, type='hsr')
    st.append([4,5,6], cf=12222, type='msr')

    print st
    train = st[0]
    print train[(train > 2)]
    print st[0][2]



    import thorns as th
    print
    print "trimming"
    print th.trim(st, 0, 2)




if __name__ == "__main__":
    main()
