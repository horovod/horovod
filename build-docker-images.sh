#!/bin/bash

set -e
set -x

function build_one()
{
    py=$1
    device=$2

    tensorflow_pkg='tensorflow'
    if [[ ${device} == 'cpu' ]]; then
        tensorflow_pkg='tensorflow-cpu'
    fi

    mxnet_pkg='mxnet'
    if [[ ${device} == 'gpu' ]]; then
        mxnet_pkg='mxnet-cu101'
    fi

    tag=horovod-build-py${py}-${device}:$(date +%Y%m%d-%H%M%S)
    docker build -f Dockerfile.${device} -t ${tag} --build-arg python=${py} .
    horovod_version=$(docker run --rm ${tag} pip show horovod | grep Version | awk '{print $2}')
    tensorflow_version=$(docker run --rm ${tag} pip show ${tensorflow_pkg} | grep Version | awk '{print $2}')
    pytorch_version=$(docker run --rm ${tag} pip show torch | grep Version | sed 's/+/ /g' | awk '{print $2}')
    mxnet_version=$(docker run --rm ${tag} pip show ${mxnet_pkg} | grep Version | awk '{print $2}')
    final_tag=horovod/horovod:${horovod_version}-tf${tensorflow_version}-torch${pytorch_version}-mxnet${mxnet_version}-py${py}-${device}
    docker tag ${tag} ${final_tag}
    docker rmi ${tag}
}

# clear upstream images, ok to fail if images do not exist
# docker rmi $(cat Dockerfile.cpu | grep FROM | awk '{print $2}') || true
# docker rmi $(cat Dockerfile.gpu | grep FROM | awk '{print $2}') || true

# build for cpu and gpu
build_one 3.7 cpu
# build_one 3.7 gpu

# print recent images
docker images horovod/horovod
