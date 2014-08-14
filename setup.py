#!/usr/bin/env python

from setuptools import setup, find_packages

with open('README.rst') as file:
    long_description = file.read()

setup(
    name = "thorns",
    version = "0.5",
    author = "Marek Rudnicki",
    author_email = "marek.rudnicki@tum.de",

    description = "Spike analysis software",
    license = "GPLv3+",
    url = "https://github.com/mrkrd/thorns",
    download_url = "https://github.com/mrkrd/thorns/tarball/master",

    packages = find_packages(),
    package_data = {
        "thorns.datasets": ["anf_zilany2014.pkl"],
    },
    long_description = long_description,
    classifiers = [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
    ],

    platforms = ["Linux", "Windows", "FreeBSD"],
    install_requires=["numpy", "pandas", "scipy"],
)
