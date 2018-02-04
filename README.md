![alt text](https://raw.githubusercontent.com/daquang/YAMDA/master/logo/logoYAMDA.jpg?token=ADRfUF9JObmq5QPrx0wpIk4d7D-8HnLgks5afSfawA%3D%3D "Fast GPU-enabled motif discovery")

A highly scalable GPU-accelerated *de novo* motif discovery software package

Daniel Quang, Yuanfang Guan, Stephen Parker; YAMDA: a highly scalable GPU-accelerated *de novo* motif discovery software package; (in preparation).

---

## Table of Contents
* [Installation](#installation)
    * [Required dependencies]
    * [Optional]
    * [Docker]
* [Examples](#examples)
    * [ChIP-seq example]
    * [DGF example]
* [To-Do](#to-do)

---

## Installation

YAMDA relies on several open source software packages. Links and version numbers for the packages used to develop and
test YAMDA are listed below. The best and easiest way to install all dependencies is with
[Anaconda](https://www.anaconda.com/) (5.0.1, Python 3.6 version). macOS High Sierra or Ubuntu 16.04 is recommended.

### Required dependencies
* [Python](https://www.python.org) (3.6.4). I chose Python 3.6 instead of Python 2.7 for initial YAMDA development
because the latter will  no longer be supported in 2020. YAMDA imports the following standard Python packages:
sys, os, errno, re, argparse, pickle, and itertools.
* [numpy]
* [scipy]
* [pyfaidx]
* [PyTorch] (0.3.0). Both GPU and CPU versions are supported.

### Optional dependencies
These are software packages and Python libraries that are not necessary to run YAMDA, but are nevertheless recommended.
They contain extra utilities that can extend the functionality of YAMDA or help preprocess data. Once again, I've put
links and version numbers of what I used, but any recent version of these packages should be fine.


* [http://meme-suite.org/ The MEME suite] (4.12.0). Appropriately enough, the MEME suite
* biopython
```
conda install -c anaconda biopython
```
* BEDTools. Since I also needed the pybedtools wrapper library, I installed BEDTools with the following conda command:
```
conda install -c bioconda pybedtools
```

### Docker

---

## Examples

---

## To-Do
* Docker installation
* Test YAMDA on RNA and protein sequences
* Python 2.7 compatibility
* Cythonize the seeding step
* Add ZOOPS model
* Add OOPS