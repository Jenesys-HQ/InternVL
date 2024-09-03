ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.07-py3
FROM ${FROM_IMAGE_NAME}

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX"

RUN cd /opt && \
    pip install --upgrade pip && \
    pip list | \
    awk '{print$1"=="$2}' | \
    tail +3 > pip_constraints.txt

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common build-essential autotools-dev \
    libaio-dev \
    nfs-common pdsh \
    cmake g++ gcc \
    curl wget vim tmux emacs less unzip \
    htop iftop iotop ca-certificates openssh-client openssh-server \
    rsync iputils-ping net-tools sudo \
    llvm-dev ffmpeg libsm6 libxext6


#RUN pip install --upgrade pip && \
#    pip install \
#    triton \
#    ninja \
#    hjson \
#    py-cpuinfo \
#    mpi4py

##############################################################################
# PyYAML build issue
##############################################################################
RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
    rm -rf /usr/lib/python3/dist-packages/PyYAML-*

##############################################################################
# AWS CLI Setup
##############################################################################
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip

##############################################################################
# Conda Package Installer
##############################################################################
RUN mkdir -p /miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh && \
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 && \
    rm -rf /miniconda3/miniconda.sh && \
    /miniconda3/bin/conda init bash

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Add jack user
##############################################################################
# Add a jack user with user id 1000
RUN useradd --create-home --uid 1000 --shell /bin/bash jack && \
    usermod -aG sudo jack && \
    echo "jack ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Ensure 'jack' has access to necessary directories
RUN chown -R jack:jack /miniconda3 ${STAGE_DIR}

# Switch to non-root user
USER jack

##############################################################################
# Create and activate conda environemnt
##############################################################################
#RUN conda create -n internvl python=3.10
#RUN conda activate internvl
#
###############################################################################
## Install DeepSpeed as 'jack' user
###############################################################################
#RUN git clone https://github.com/microsoft/DeepSpeed.git ${STAGE_DIR}/DeepSpeed && \
#    cd ${STAGE_DIR}/DeepSpeed && \
#    git checkout master && \
#    ./install.sh --pip_sudo && \
#    rm -rf ${STAGE_DIR}/DeepSpeed
#
##RUN python -c "import deepspeed; print(deepspeed.__version__)"
#
###############################################################################
## Set working repository
###############################################################################
#RUN git clone https://github.com/Jenesys-HQ/InternVL.git
#RUN cd /workspace/InternVL
#
#RUN pip install -r /workspace/InternVL/requirements/internvl_chat.txt
#RUN pip uninstall transformer-engine -y
#
#RUN python -c "import torch; print(torch.__version__)"