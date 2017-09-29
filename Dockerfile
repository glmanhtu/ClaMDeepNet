FROM ubuntu:16.04
LABEL maintainer caffe-maint@googlegroups.com

COPY libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb /tmp/
COPY libcudnn6_6.0.21-1+cuda8.0_amd64.deb /tmp

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
        tar \
        wget \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cuda && apt-get clean

RUN dpkg -i /tmp/libcudnn6_6.0.21-1+cuda8.0_amd64.deb
RUN dpkg -i /tmp/libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb

RUN sh -c "echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda_hack.conf"
RUN ldconfig /usr/local/cuda/lib64

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: use ARG instead of ENV once DockerHub supports this
# https://github.com/docker/hub-feedback/issues/460
RUN git clone --branch caffe-0.15 --depth 1 https://github.com/NVIDIA/caffe.git . && \
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    cp Makefile.config.example Makefile.config && \
    perl -i -p -e 's/(\s*)(-gencode[=\w,_ ]*\s*)\\(\s*)/ \2 /g' Makefile.config && \
    sed -i -E 's/(CUDA_ARCH\s*:=)(-|\w|\s|=|,|\\)+/\1 -gencode=arch=compute_60,code=sm_60/' Makefile.config && \
    sed -i '/^# WITH_PYTHON_LAYER := 1/s/^# //' Makefile.config && \
    sed -i '/^# USE_CUDNN := 1/s/^# //' Makefile.config && \
    sed -i 's/\/usr\/local\/cuda/\/usr\/local\/cuda-8.0/g' Makefile.config && \
    sed -i '/^PYTHON_INCLUDE/a    /usr/local/lib/python2.7/dist-packages/numpy/core/include/ \\' Makefile.config && \
    mkdir build && cd build && \
    cmake .. && \
    make -j"$(nproc)"  && \
    make -j"$(nproc)" test && \
    make runtest

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

RUN apt-get update
RUN pip install mako
RUN pip install lmdb
RUN apt-get install python-opencv

WORKDIR /workspace