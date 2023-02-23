FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget git curl libbz2-dev && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tar.xz && \
    tar -xvf Python-3.9.7.tar.xz && \
    cd Python-3.9.7 && \
    ./configure --prefix=/opt/python3 && \
    make altinstall && \
    ln -s /opt/python3/bin/python3.9 /usr/bin/python3.9 && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

ENV PATH="/root/.cargo/bin:${PATH}"
RUN python3.9 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN git clone https://github.com/jerryji1993/DNABERT
WORKDIR ./DNABERT

COPY ./src /DNABERT/src

RUN python3.9 -m pip install --editable .
WORKDIR ./examples
RUN echo "tensorboardX\ntensorboard\nscikit-learn >= 0.22.2\nseqeval\npyahocorasick\nscipy\nstatsmodels\nbiopython\npandas\npybedtools\nsentencepiece\ntqdm\nFlask\npyfaidx" > requirements.txt
RUN python3.9 -m pip install -r requirements.txt

COPY ./server.py /DNABERT/examples
COPY ./test_server.py /DNABERT/examples

RUN mkdir /DNABERT6
RUN mkdir /DNABERT_storage
COPY ./weights /DNABERT6
COPY ./run_finetune.py /DNABERT/examples

CMD python3.9 server.py
