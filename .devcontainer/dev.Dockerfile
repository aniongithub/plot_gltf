
FROM python

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y \
        git git-lfs \
        build-essential

RUN pip3 install \
    numpy \
    numpy-quaternion \
    matplotlib \
    pyopengl \
    pygltflib \
    pytransform3d[all]