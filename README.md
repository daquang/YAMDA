<p align="center">
<img src="https://github.com/daquang/YAMDA/raw/master/logo/logo.jpg" width="300">
</p>
<!---
![YAMDA](https://github.com/daquang/YAMDA/raw/master/logo/logo.jpg "Fast GPU-accelerated motif discovery")
-->

A highly scalable GPU-accelerated *de novo* motif discovery software package

Please post in the Issues board or e-mail me (daquang@umich.edu) if you have any questions, suggestions, or complaints 
:)

---

## Table of Contents
* [Citation](#citation)
* [Installation](#installation)
    * [Required dependencies](#required-dependencies)
    * [Optional dependencies](#optional-dependencies)
    * [Docker](#docker)
* [Examples](#examples)
    * [Making a masked genome FASTA](#making-a-masked-genome-fasta)
    * [Extracting BED interval FASTA sequences](#extracting-bed-interval-fasta-sequences)
    * [Motif discovery in ChIP-seq](#motif-discovery-in-chip-seq)
    * [Motif discovery in DGF](#motif-discovery-in-dgf)
* [To-Do](#to-do)

---

## Citation

```
@article{doi:10.1093/bioinformatics/bty396,
author = {Quang, Daniel and Guan, Yuanfang and Parker, Stephen C J},
title = {YAMDA: thousandfold speedup of EM-based motif discovery using deep learning libraries and GPU},
journal = {Bioinformatics},
volume = {},
number = {},
pages = {bty396},
year = {2018},
doi = {10.1093/bioinformatics/bty396},
URL = {http://dx.doi.org/10.1093/bioinformatics/bty396},
eprint = {/oup/backfile/content_public/journal/bioinformatics/pap/10.1093_bioinformatics_bty396/1/bty396.pdf}
}
```

---

## Installation
Clone a copy of the YAMDA repository:

```
git clone https://github.com/daquang/YAMDA.git
```

Or download a stable release version (v0.1 should reproduce the paper's results exactly, but uses older libraries):
```
wget https://github.com/daquang/YAMDA/archive/0.1.tar.gz
```

YAMDA relies on several open source software packages. Links and version numbers for the packages used to develop and
test YAMDA are listed below; however, typically any recent version of these packages should be fine for running YAMDA. 
The best and easiest way to install all dependencies is with [Anaconda](https://www.anaconda.com/) (5.2, Python 3.6 
version). Anaconda uses pre-built binaries for specific operating systems to allow simple installation of Python and 
non-Python software packages. macOS High Sierra or Ubuntu 18.04 is recommended.

### Required dependencies
* [Python](https://www.python.org) (3.6.5). I chose Python 3.6 instead of Python 2.7 for initial YAMDA development
because the latter will  no longer be supported in 2020. YAMDA imports the following standard Python packages:
sys, os, errno, re, argparse, pickle, and itertools.
* [numpy](http://www.numpy.org/) (1.15.0). Python scientific computing library. Comes pre-packaged in Anaconda.
* [scipy](https://www.scipy.org/) (1.1.0). Python scientific computing library. Comes pre-packaged in Anaconda.
* [pyfaidx](https://github.com/mdshw5/pyfaidx) (0.5.4.1). Python wrapper module for indexing, retrieval, and in-place 
modification of FASTA files using a samtools compatible index. Easily installed in Anaconda with the following command 
line:
```
pip install pyfaidx
```
* [tqdm](https://pypi.python.org/pypi/tqdm) (4.24.0). Progress bar. Easily installed in Anaconda with the following 
command line:
```
pip install tqdm
```
* [PyTorch](http://pytorch.org/) (0.4.1). Tensor computation library from Facebook AI that forms the backbone of YAMDA. 
Both GPU and CPU versions are supported. It is recommended you check out the official 
[PyTorch website](http://pytorch.org) for foolproof methods of installation for specific operating systems and hardware 
configurations.

**tl;dr**, the following command line should work most of the time for installing PyTorch.
```
conda install pytorch torchvision -c pytorch 
```

### Optional dependencies
These are software packages and Python libraries that are not necessary to run YAMDA, but are nevertheless recommended.
They contain extra utilities that can extend the functionality of YAMDA or help preprocess data. Once again, I've put
links and version numbers of what I used, but any recent version of these packages should be fine.

* [The MEME suite](http://meme-suite.org/) (4.12.0). Appropriately enough, the MEME suite has many tools for 
processing FASTA and motif files. Among these are the fasta-shuffle-letters utility, which is useful for generating 
negative controls. MEME can also be installed easily enough from its main website or through Anaconda:
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
* [BEDTools](http://bedtools.readthedocs.io/en/latest/) (0.7.10). Standard BEDTools suite is useful for extracting FASTA 
sequences from BED files. Since I also needed the pybedtools wrapper library, I installed BEDTools with the following 
conda command:
```
conda install -c bioconda pybedtools
```

### Streamlined (can ignore this part if you already manually installed all dependencies)

#### Anaconda Install
```bash
cd /tmp && wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ./anaconda3.sh && bash ./anaconda3.sh -u -b -p /opt/anaconda3 && export PATH="/opt/anaconda3/bin:$PATH" && cd -;
```

#### Install Detailed
```bash
conda update -yn base conda && conda update -y --prefix /opt/anaconda3 anaconda && conda create -fmy -c defaults -c anaconda -c conda-forge -c bioconda -c pytorch -n YAMDA-env python=3.6.5 numpy=1.13.3 scipy=0.19.1 pyfaidx tqdm pytorch torchvision meme anaconda biopython pybedtools && source activate YAMDA-env;
```

#### Install Easy
```bash
conda env create -f environment.yml && . activate YAMDA-env;
```

#### Exit Env
`source deactivate`

#### Kill Env
`conda env remove --name YAMDA-env`

### Docker (can ignore this part if you do not intend on doing a Docker installation)
1. Install docker on whatever: https://www.docker.com/community-edition
```bash
cd /tmp && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo apt-get update && apt-cache policy docker-ce && sudo apt-get install -y docker-ce && cd -;
```
2. Install docker-compose on same whatever: https://docs.docker.com/compose/install
```bash
cd /tmp && sudo curl -L https://github.com/docker/compose/releases/download/1.18.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose && cd -;
```
3. Make docker image using the makefile `make yamda-dock`
4. To have docker run the CMD you put into the Dockerfile `sudo docker run yamda-dock`
5. To ssh into the image, for debugging and so on: `sudo docker run -it yamda-dock bash`
6. When in the image don't forget to `source activate YAMDA-env`
7. To kill the image and cleanup docker `make cleanup`

Docker is screwy about importing global variables in your environment, which you'll probably want now or later. So far to do it easily and relatively conveniently you need to enter the variable 4 times in 3 different places, twice in the Dockerfile, once in the .env file, and once in the docker-compose.yml file. I made an example VAR to make it clear how to do that. 

---

## Examples
In the examples folder, you will find the narrowPeak and masked FASTA files that are needed to reproduce results in the 
manuscript. For your convenience, I have included the major preprocessing steps that typically comprise a *de novo* 
motif discovery pipeline.

### Making a masked genome FASTA
Motif discovery for DNA usually performs better on a FASTA sequence set with all repetitive sequences masked. This is 
typically accomplished by first generating a masked genome where all repetitive sequence residues are replaced with 
capital N's. The following command lines will download masked hg19 chromosome FASTA files, assemble the individual 
files into a single FASTA file (hg19.fa.masked), and remove all intermediate files:
```
wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/chromFaMasked.tar.gz
tar zxvf chromFaMasked.tar.gz
cat chr*.fa.masked >> hg19.fa.masked
rm chr*.fa.masked chromFaMasked.tar.gz
```
### Extracting BED interval FASTA sequences
BEDTools' fastaFromBed utility is useful for extracting letter sequences from a reference fasta file based on feature 
coordinates. The following command lines demonstrate how to do this from an ENCODE narrowPeak file (H1 POU5F1) to 
generate 100 bp sequences centered on peak summits. For simplicity, we will use the same masked genome FASTA file 
generated in the previous example.
```
zcat Examples/wgEncodeAwgTfbsHaibH1hescPou5f1sc9081V0416102UniPk.narrowPeak.gz | awk -v "OFS=\t" '{print $1,$2 + $10 - 50,$2 + $10 + 50}' | fastaFromBed -bed stdin -fi hg19.fa.masked -fo Examples/H1_POU5F1_ChIP_HAIB.fa.masked
```
### Motif discovery in ChIP-seq
This example demonstrates motif discovery on the H1 POU5F1 ChIP-seq data. YAMDA requires a positive FASTA file
and a negative FASTA file. The latter is typically a dinucleotide-shuffled control version of the positive file. The 
fasta-shuffle-letters utility from the MEME-suite is useful for this purpose. 

```
fasta-shuffle-letters -kmer 2 -s 0 Examples/H1_POU5F1_ChIP_HAIB.fa.masked H1_POU5F1_ChIP_HAIB_shuffled.fa.masked
```

The run_em.py script executes the motif discovery program on the FASTA pairs. Use `python run_em.py -h` to get a 
detailed description of the script's arguments. Note that to run this example, you do not necessarily need to run the 
previous examples because all the necessary files have already been prepackaged with this repository. 

```
python run_em.py -r -e -i Examples/H1_POU5F1_ChIP_HAIB.fa.masked -j Examples/H1_POU5F1_ChIP_HAIB_shuffled.fa.masked -oc H1_POU5F1_output 
```

The output folder H1_POU5F1_output contains the following files:

* model.pkl. A saved/pickled version of the learned mixture model.
* motifs.txt. The discovered motif(s) in Minimal MEME format. This file can be further processed with MEME utilities 
such as meme2images and TOMTOM.
* positive_seqs.fa. A FASTA of the positive sequences with all instances of the discovered motif(s) erased.
* negative_seqs.fa. A FASTA of the negative sequences with all instances of the discovered motif(s) erased.

### Motif discovery in DGF
This example demonstrates motif discovery on the K562 Digital Genomic Footprinting dataset. This is the same example 
from [EXTREME](https://github.com/uci-cbcl/EXTREME).

Motif discovery in DGF is similar to motif discovery in ChIP-seq; however, due to the rarity of motifs in DGF datasets, 
we found that it helps to erase all overlapping instances of repetitive sequences such as AAAAAA/TTTTTT and 
CCCGCCC/GGGCGGG:


```
python erase_annoying_sequences.py -i Examples/K562_DNase.fa -o Examples/K562_DNase_eraseannoying.fa
fasta-shuffle-letters -kmer 2 -dna -seed 0 Examples/K562_DNase_eraseannoying.fa Examples/K562_DNase_eraseannoying_shuffled.fa
```

Now we can run the YAMDA algorithm on the FASTA file:
```
python run_em.py -f 0.1 -r -e -maxs 20000 -i Examples/K562_DNase_eraseannoying.fa -j Examples/K562_DNase_eraseannoying_shuffled.fa -oc K562_DNase_output
```

The -f argument is one of the most difficult, yet perhaps most important, arguments. The closest corresponding argument 
in MEME is wnsites. The closer the -f argument is to zero, the stronger the bias towards motifs with exactly the 
expected number of sites. The default value of 0.1 works well for most ChIP-seq and some DGF datasets, but in cases of 
even rarer motifs smaller values (e.g. 0.025) is necessary.

---

## To-Do
Here is a list of features I plan to add. They will be added according to demand.
* Test YAMDA on RNA and protein sequences
* Python 2.7 compatibility
* Cythonize seeding step and reduce its memory overhead
* Add more examples (e.g. SELEX data)
* Add ZOOPS (zero or one occurrence per sequence) and OOPS (one occurrence per sequence) models. YAMDA currently only supports the TCM (two component model), whereas
MEME supports all three. ZOOPS and OOPS may offer faster and more accurate performance for certain datasets, such as ChIP-seq.

In addition, I promise to update YAMDA as library dependencies are updated.