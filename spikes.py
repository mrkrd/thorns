#!/usr/bin/env python

from __future__ import division

__author__ = "Marek Rudnicki"

import random
import numpy as np


class Spikes(object):
    """Class that respresents a single spike train.  Objects of this
    class have the same interface as np.ndarray's.  Meta data can be
    accessed using spikes['key'] notation.

    Raw spikes are in self.spikes.  Metadata is a dictionary in
    self.meta.

    """


    def __init__(self, spikes=[], **kwargs):

        ### Initialize spikes
        self.spikes = np.array(spikes, dtype=float)
        assert self.spikes.ndim == 1


        ### Initialize meta
        if isinstance(spikes, Spikes):
            self.meta = dict(spikes.meta)
        else:
            self.meta = dict()

        self.meta.update(kwargs)



    def copy_meta(func):
        """
        Decorator that copies metadata before returning the output
        train.

        """
        def wrapper(self, *args, **kwargs):
            output_train = func(self, *args, **kwargs)

            if isinstance(output_train, np.ndarray):
                output_train = Spikes(output_train, **self.meta)

            return output_train

        return wrapper



    @copy_meta
    def __getitem__(self, key):
        if isinstance(key, str):
            item = self.meta[key]
        else:
            item = self.spikes.__getitem__(key)
        return item

    def __setitem__(self, key, value):
        self.spikes.__setitem__(key, value)

    def __len__(self):
        return self.spikes.__len__()

    def __iter__(self):
        return self.spikes.__iter__()

    def __str__(self):
        return self.spikes.__str__() + " " + self.meta.__str__()

    def __lt__(self, other):
        return self.spikes.__lt__(other)
    def __le__(self, other):
        return self.spikes.__le__(other)
    def __eq__(self, other):
        return self.spikes.__eq__(other)
    def __ne__(self, other):
        return self.spikes.__ne__(other)
    def __gt__(self, other):
        return self.spikes.__gt__(other)
    def __ge__(self, other):
        return self.spikes.__ge__(other)

    @copy_meta
    def __add__(self, other):
        return self.spikes.__add__(other)

    @copy_meta
    def __sub__(self, other):
        return self.spikes.__sub__(other)

    @copy_meta
    def __mul__(self, other):
        return self.spikes.__mul__(other)






class Trains(object):
    def __init__(self, trains=[]):
        self.trains = [Spikes(spikes) for spikes in trains]



    def __len__(self):
        return self.trains.__len__()



    def __setitem__(self, key, value):
        self.trains.__setitem__(key, value)



    def __getitem__(self, key):

        if isinstance(key, str):
            ### String gives us an array of attributes
            item = np.array([spikes[key] for spikes in self.trains])

        elif isinstance(key, np.ndarray) and (key.dtype is np.dtype('bool')):
            ### Indexing using Bool table (a la Numpy)
            assert len(key) == len(self.trains)
            idx = np.where(key)[0]
            item = Trains([self[i] for i in idx])

        else:
            ### Integer indexing
            item = self.trains.__getitem__(key)

        return item



    def __iter__(self):
        return self.trains.__iter__()



    def __str__(self):
        s = "[\n"
        for t in self.trains:
            s = s + "  " + str(t) + "\n"
        s += "]"
        return s



    def append(self, spikes, **kwargs):
        spikes = Spikes(spikes, **kwargs)
        self.trains.append(spikes)



    def extend(self, L):
        trains = [Spikes(spikes) for spikes in L]
        self.trains.extend(trains)



    def where(self, **kwargs):
        mask = np.ones(len(self.trains), dtype=np.dtype('bool'))

        for key in kwargs:
            mask = mask & (self[key]==kwargs[key])

        return self[mask]



    def pop(self, random=False, *args):
        if random:
            i = random.randint(0, len(self.trains)-1)
            train = self.trains.pop(i)
        else:
            train = self.trains.pop(*args)

        return train




def main():
    import thorns as th
    from thorns.spikes import Spikes, Trains

    print
    print "=== Spikes ==="
    sp = Spikes([1,2,3], cf=12)
    print sp[1]
    print sp[0:2]
    print sp['cf']


    print
    print "=== Trains ==="

    t = Trains()
    t.append([1,2,3], cf=12000, type='hsr')
    t.append([4,5,6], cf=12222, type='msr')

    print "trains:", t
    print "spikes:", t[0]
    print "spike:", t[0][2]

    print
    print "trimming"
    print th.trim_spikes(t, 0, 2)




if __name__ == "__main__":
    main()
