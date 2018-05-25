FROM continuumio/anaconda3
ENV DEBIAN_FRONTEND noninteractive
WORKDIR /yamda
COPY . /yamda

ARG YAMDA_EXAMPLE_VAR
ENV YAMDA_EXAMPLE_VAR ${YAMDA_EXAMPLE_VAR}

RUN dpkg-reconfigure -f noninteractive tzdata && apt-get update --fix-missing && apt-get install -fy apt-utils && apt-get upgrade -y && apt-get autoremove && apt-get autoclean && dpkg --configure -a;
RUN conda update -yn base conda && conda update -y --prefix /opt/conda anaconda;
RUN conda env create -f environment.yml && . activate YAMDA-env;

CMD ["python", "run_em.py", "-r", "-e", "-i", "Examples/H1_POU5F1_ChIP_HAIB.fa.masked", "-j", "Examples/H1_POU5F1_ChIP_HAIB_shuffled.fa.masked" "-oc", "H1_POU5F1_output"]
