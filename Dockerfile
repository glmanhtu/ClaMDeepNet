
FROM nvidia/cuda:8.0-devel-ubuntu16.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \
        python-tk && \
        bsdmainutils && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
RUN git clone --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    pip install flask
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && \
    perl -i -p -e 's/(\s*)(-gencode[=\w,]*\s*)\\(\s*)/ \2 /g' Makefile &&\
    sed -i -E 's/(NVCC_GENCODE\s*\?=)(-|\w|\s|=|,|\\)+/\1 -gencode=arch=compute_52,code=sm_52/' Makefile && \
    make -j install && cd .. && \
    cp Makefile.config.example Makefile.config && \
    perl -i -p -e 's/(\s*)(-gencode[=\w,_ ]*\s*)\\(\s*)/ \2 /g' Makefile.config && \
    sed -i -E 's/(CUDA_ARCH\s*:=)(-|\w|\s|=|,|\\)+/\1 -gencode=arch=compute_52,code=sm_52/' Makefile.config && \
    sed -i '/^# WITH_PYTHON_LAYER := 1/s/^# //' Makefile.config && \
    sed -i 's/\/usr\/local\/cuda/\/usr\/local\/cuda-8.0/g' Makefile.config && \
    sed -i '/^PYTHON_INCLUDE/a    /usr/local/lib/python2.7/dist-packages/numpy/core/include/ \\' Makefile.config && \
    mkdir build && cd build && \
    cmake -DUSE_NCCL=1 .. && \
    make -j"$(nproc)" all

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN apt-get update
RUN pip install mako
RUN pip install lmdb
RUN apt-get install python-opencv

WORKDIR /workspace
RUN mkdir classify-images
COPY . /workspace/classify-images/