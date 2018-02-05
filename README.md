![alt text](https://raw.githubusercontent.com/daquang/YAMDA/master/logo/logoYAMDA.jpg?token=ADRfUF9JObmq5QPrx0wpIk4d7D-8HnLgks5afSfawA%3D%3D "Fast GPU-accelerated motif discovery")

A highly scalable GPU-accelerated *de novo* motif discovery software package

Daniel Quang, Yuanfang Guan, Stephen Parker; YAMDA: a highly scalable GPU-accelerated *de novo* motif discovery 
software package; (in preparation).

---

## Table of Contents
* [Installation](#installation)
    * [Required dependencies](#required-dependencies)
    * [Optional dependencies](#optional-dependencies)
    * [Docker](#docker)
* [Examples](#examples)
    * [Making a masked genome FASTA]
    * [Extracting BED interval FASTA sequences]
    * [*De novo* motif discovery in ChIP-seq]
    * [*De novo* motif discovery in DGF]
* [To-Do](#to-do)

---

## Installation

YAMDA relies on several open source software packages. Links and version numbers for the packages used to develop and
test YAMDA are listed below; however, typically any recent version of these packages should be fine for running YAMDA. 
The best and easiest way to install all dependencies is with [Anaconda](https://www.anaconda.com/) (5.0.1, Python 3.6 
version). Anaconda uses pre-built binaries for specific operating systems to allow simple installation of Python and 
non-Python software packages. macOS High Sierra or Ubuntu 16.04 is recommended.

### Required dependencies
* [Python](https://www.python.org) (3.6.4). I chose Python 3.6 instead of Python 2.7 for initial YAMDA development
because the latter will  no longer be supported in 2020. YAMDA imports the following standard Python packages:
sys, os, errno, re, argparse, pickle, and itertools.
* [numpy](http://www.numpy.org/) (1.13.3). Python scientific computing library. Comes pre-packaged in Anaconda.
* [scipy](https://www.scipy.org/) (0.19.1). Python scientific computing library. Comes pre-packaged in Anaconda.
* [pyfaidx](https://github.com/mdshw5/pyfaidx) (0.5.2). Python wrapper module for indexing, retrieval, and in-place 
modification of FASTA files using a samtools compatible index. Easily installed in Anaconda with the following command 
line:
```
pip install pyfaidx
```
* [tqdm](https://pypi.python.org/pypi/tqdm) (4.19.5). Progress bar. Easily installed in Anaconda with the following 
command line:
```
pip install tqdm
```
* [PyTorch](http://pytorch.org/) (0.3.0). Tensor computation library from Facebook AI that forms the backbone of YAMDA. 
Both GPU and CPU versions are supported. It is recommended you check out the official 
[PyTorch website](http://pytorch.org) for foolproof methods of installation for specific operating systems and hardware 
configurations. Otherwise, the following tl;dr command line should work most of the time.
```
conda install pytorch torchvision -c pytorch 
```

### Optional dependencies
These are software packages and Python libraries that are not necessary to run YAMDA, but are nevertheless recommended.
They contain extra utilities that can extend the functionality of YAMDA or help preprocess data. Once again, I've put
links and version numbers of what I used, but any recent version of these packages should be fine.

* [The MEME suite](http://meme-suite.org/) (4.12.0). Appropriately enough, the MEME suite has many tools for 
processing FASTA and motif files. Among these are the fasta-shuffle-letters utility, which is useful for generating 
negative controls. MEME can also be installed easily enough through Anaconda:
```
conda install -c bioconda meme 
```
However, for my MacBook Pro, this command line yielded some errors. I had to download a more specific set of binaries 
for my specific operating system and version of Python, as follows:
```
wget https://anaconda.org/bioconda/meme/4.12.0/download/osx-64/meme-4.12.0-py36pl5.22.0_1.tar.bz2
conda install meme-4.12.0-py36pl5.22.0_1.tar.bz2
```
* [biopython](http://biopython.org/) (1.7.0). Required to read bgzipped FASTA files. Convenient if you like storing 
files compressed.
```
conda install -c anaconda biopython
```
* [BEDTools](http://bedtools.readthedocs.io/en/latest/) (0.7.10). Standard BEDTools suite is useful for extracing FASTA 
sequences from BED files. Since I also needed the pybedtools wrapper library, I installed BEDTools with the following 
conda command:
```
conda install -c bioconda pybedtools
```

### Docker
Coming soon.

---

## Examples
### Making a masked genome FASTA
### Extracting BED interval FASTA sequences
### *De novo* motif discovery in ChIP-seq
### *De novo* motif discovery in DGF

---

## To-Do
* Docker installation
* Test YAMDA on RNA and protein sequences
* Python 2.7 compatibility
* Cythonize seeding step
* Add ZOOPS model
* Add OOPS model