
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.07-py3
FROM ${FROM_IMAGE_NAME}

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX"

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    software-properties-common build-essential autotools-dev \
    libaio-dev \
    nfs-common pdsh \
    cmake g++ gcc \
    curl wget vim tmux emacs less unzip \
    htop iftop iotop ca-certificates openssh-client openssh-server \
    rsync iputils-ping net-tools sudo \
    llvm-dev ffmpeg libsm6 libxext6  \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

##############################################################################
# Add jack user
##############################################################################
# Add a jack user with user id 1000
RUN useradd --create-home --uid 1000 --shell /bin/bash jack \
 && usermod -aG sudo jack \
 && echo "jack ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to non-root user
USER jack

##############################################################################
# Setup Conda and install environment
##############################################################################
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p ~/miniconda \
 && rm  ~/miniconda.sh \
 && ~/miniconda/bin/conda clean -ayq

ENV PATH ~/miniconda/envs/internvl/bin:$PATH
ENV CONDA_DEFAULT_ENV internvl

RUN ~/miniconda/bin/conda init bash \
 && source ~/.bashrc \
 && ~/miniconda/bin/conda create -y -n internvl python=3.10 \
 && ~/miniconda/bin/conda clean -a -y \
 && echo "conda activate internvl" >> ~/.bashrc

###############################################################################
## Setup 'postal' library from source
###############################################################################
RUN cd ~ \
 && git clone https://github.com/openvenues/libpostal \
 && cd libpostal \
 && ./bootstrap.sh \
 && ./configure \
 && sudo make \
 && sudo make install \
 && echo "LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

###############################################################################
## Set working repository
###############################################################################
COPY . /workspace/InternVL
#RUN git clone https://github.com/Jenesys-HQ/InternVL.git && \
#    cd /workspace/InternVL && \
#    git checkout AIR-221/setup-docker-container # TODO remove this line after testing

###############################################################################
## Install requirements
###############################################################################
RUN pip uninstall transformer-engine -y \
 && pip install -r /workspace/InternVL/requirements/internvl_chat.txt \
 && pip install flash-attn --no-build-isolation \
 && pip install -r /workspace/InternVL/requirements/internvl_chat_eval.txt

RUN sudo chown -R jack:jack /workspace/InternVL

WORKDIR /workspace/InternVL/internvl_chat