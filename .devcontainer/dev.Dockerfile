
FROM python

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y \
        nano git git-lfs \
        build-essential \
        python3-dev \
        python3-pip \
        ffmpeg \
        graphviz

RUN pip3 install \
    numpy \
    numpy-quaternion \
    matplotlib \
    pyopengl \
    pygltflib \
    pytransform3d[all]