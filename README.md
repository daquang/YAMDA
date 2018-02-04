![alt text](https://raw.githubusercontent.com/daquang/YAMDA/master/logo/logoYAMDA.jpg?token=ADRfUF9JObmq5QPrx0wpIk4d7D-8HnLgks5afSfawA%3D%3D "Fast GPU-enabled motif discovery")

A highly scalable GPU-accelerated *de novo* motif discovery software package

Daniel Quang, Yuanfang Guan, Stephen Parker; YAMDA: a highly scalable GPU-accelerated *de novo* motif discovery software package; (in preparation).

---

## Table of Contents
* [Installation](#installation)
    * [Dependencies]
    * [Docker]
    * [Optional]
* [Examples](#examples)
    * [ChIP-seq example]
    * [DGF example]
* [To-Do](#to-do)

---

## Installation

YAMDA relies on several open source Python packages.

### Dependencies
* [Python] (https://www.python.org) (2.7.12). The easiest way to install Python and all of the necessary dependencies is
to download and install [Anaconda] (https://www.continuum.io) (4.3.1). I listed the versions of Python and Anaconda I
used, but the latest versions should be fine. If you're curious as to what packages in Anaconda are used, they are:
[numpy] (http://www.numpy.org/) (1.10.4), [scipy] (http://www.scipy.org/) (0.17.0). Standard python packages are:
sys, os, errno, argparse, pickle and itertools.

### Optional
These are software packages and Python libraries that are not necessary to run YAMDA, but are nevertheless recommended.

* biopython
```
conda install -c anaconda biopython
```

## To-Do
* Docker installation
* Test YAMDA on RNA and protein sequences
* Python 2.7 compatibility
* Cythonize the seeding step
* Add ZOOPS model
* Add OOPS