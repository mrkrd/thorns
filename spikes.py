#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import random
import numpy as np

class Meta(object):
    pass


class Train(object):
    def __init__(self, vec=[], **kwargs):
        if isinstance(vec, Train):
            self.meta = vec.meta
        else:
            self.meta = Meta()

        if not isinstance(vec, np.ndarray):
            vec = np.array(vec, dtype=float)
        self._train = vec

        for key in kwargs:
            setattr(self.meta, key, kwargs[key])


    def copy_meta(f):
        """
        This decorator assures that meta data is copied for the output
        train.

        """
        def w(self, *args, **kwargs):
            vec = f(self, *args, **kwargs)
            if isinstance(vec, np.ndarray):
                vec = Train(vec)
                vec.meta = self.meta
            return vec

        return w


    def __setslice__(self, *args, **kwargs):
        self._train.__setslice__(*args, **kwargs)

    @copy_meta
    def __getslice__(self, *args, **kwargs):
        return self._train.__getslice__(*args, **kwargs)

    def __len__(self):
        return self._train.__len__()

    def __setitem__(self, *args, **kwargs):
        self._train.__setitem__(*args, **kwargs)

    @copy_meta
    def __getitem__(self, *args, **kwargs):
        return self._train.__getitem__(*args, **kwargs)

    def __iter__(self):
        return self._train.__iter__()

    def __str__(self):
        return self._train.__str__()

    def __lt__(self, other):
        return self._train.__lt__(other)
    def __le__(self, other):
        return self._train.__le__(other)
    def __eq__(self, other):
        return self._train.__eq__(other)
    def __ne__(self, other):
        return self._train.__ne__(other)
    def __gt__(self, other):
        return self._train.__gt__(other)
    def __ge__(self, other):
        return self._train.__ge__(other)

    @copy_meta
    def __add__(self, other):
        return self._train.__add__(other)

    @copy_meta
    def __sub__(self, other):
        return self._train.__sub__(other)

    @copy_meta
    def __mul__(self, other):
        return self._train.__mul__(other)

    def __floordiv__(self, other):
        return self._train.__floordiv__(other)
    def __mod__(self, other):
        return self._train.__mod__(other)
    def __divmod__(self, other):
        return self._train.__divmod__(other, *args, **kwargs)
    def __pow__(self, other, *args, **kwargs):
        return self._train.__pow__(other)
    def __lshift__(self, other):
        return self._train.__lshift__(other)
    def __rshift__(self, other):
        return self._train.__rshift__(other)
    def __and__(self, other):
        return self._train.__and__(other)
    def __xor__(self, other):
        return self._train.__xor__(other)
    def __or__(self, other):
        return self._train.__or__(other)
    def __div__(self, other):
        return self._train.__div__(other)
    def __truediv__(self, other):
        return self._train.__truediv__(other)
    def __radd__(self, other):
        return self._train.__radd__(other)
    def __rsub__(self, other):
        return self._train.__rsub__(other)
    def __rmul__(self, other):
        return self._train.__rmul__(other)
    def __rdiv__(self, other):
        return self._train.__rdiv__(other)
    def __rtruediv__(self, other):
        return self._train.__rtruediv__(other)
    def __rfloordiv__(self, other):
        return self._train.__rfloordiv__(other)
    def __rmod__(self, other):
        return self._train.__rmod__(other)
    def __rdivmod__(self, other):
        return self._train.__rdivmod__(other)
    def __rpow__(self, other):
        return self._train.__rpow__(other)
    def __rlshift__(self, other):
        return self._train.__rlshift__(other)
    def __rrshift__(self, other):
        return self._train.__rrshift__(other)
    def __rand__(self, other):
        return self._train.__rand__(other)
    def __rxor__(self, other):
        return self._train.__rxor__(other)
    def __ror__(self, other):
        return self._train.__ror__(other)
    def __iadd__(self, other):
        return self._train.__iadd__(other)
    def __isub__(self, other):
        return self._train.__isub__(other)
    def __imul__(self, other):
        return self._train.__imul__(other)
    def __idiv__(self, other):
        return self._train.__idiv__(other)
    def __itruediv__(self, other):
        return self._train.__itruediv__(other)
    def __ifloordiv__(self, other):
        return self._train.__ifloordiv__(other)
    def __imod__(self, other):
        return self._train.__imod__(other)
    def __ipow__(self, other, *args, **kwargs):
        return self._train.__ipow__(other, *args, **kwargs)
    def __ilshift__(self, other):
        return self._train.__ilshift__(other)
    def __irshift__(self, other):
        return self._train.__irshift__(other)
    def __iand__(self, other):
        return self._train.__iand__(other)
    def __ixor__(self, other):
        return self._train.__ixor__(other)
    def __ior__(self, other):
        return self._train.__ior__(other)
    def __neg__(self):
        return self._train.__neg__(other)
    def __pos__(self):
        return self._train.__pos__(other)
    def __abs__(self):
        return self._train.__abs__(other)
    def __invert__(self):
        return self._train.__invert__(other)
    def __complex__(self):
        return self._train.__complex__(other)
    def __int__(self):
        return self._train.__int__(other)
    def __long__(self):
        return self._train.__long__(other)
    def __float__(self):
        return self._train.__float__(other)


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
            item = np.array([getattr(train.meta, key) for train in self._trains])

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
            s = s + " " + str(t) + "\n"
        s += "]"
        return s

    def append(self, train, **kwargs):
        if not isinstance(train, Train):
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

    def pop(self, *args, **kwargs):
        return self._trains.pop(*args, **kwargs)

    def pop_random(self):
        i = random.randint(0, len(self._trains)-1)
        return self._trains.pop(i)


def main():
    st = SpikeTrains()
    st.append([1,2,3], cf=12000, type='hsr')
    st.append([4,5,6], cf=12222, type='msr')

    import thorns as th
    print th.trim(st, 0, 2)

    print st
    train = st[0]

    print train[(train > 2)]
    print st[0][2]



if __name__ == "__main__":
    main()
