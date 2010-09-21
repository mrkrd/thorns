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


    def __setslice__(self, *args, **kwargs):
        self._train.__setslice__(*args, **kwargs)

    def __getslice__(self, *args, **kwargs):
        return self._train.__getslice__(*args, **kwargs)

    def __len__(self):
        return self._train.__len__()

    def __setitem__(self, *args, **kwargs):
        self._train.__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self._train.__getitem__(*args, **kwargs)

    def __iter__(self):
        return self._train.__iter__()

    def __str__(self):
        return self._train.__str__()




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

    print st
    print st[0]
    print st[0][2]



if __name__ == "__main__":
    main()
