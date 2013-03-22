#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

__author__ = "Marek Rudnicki"

import mar

def main():

    db = mar.loaddb()

    print()
    print(db.to_string())


if __name__ == "__main__":
    main()
