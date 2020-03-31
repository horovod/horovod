#!/bin/bash

set -e
set -x

function build_one()
{
    py=$1
    device=$2

    tensorflow_pkg='tensorflow'
    if [[ ${device} == 'gpu' ]]; then
        tensorflow_pkg='tensorflow-gpu'
    fi

    tag=horovod-build-py${py}-${device}:$(date +%Y%m%d-%H%M%S)
    docker build -f Dockerfile.${device} -t ${tag} --build-arg python=${py} --no-cache .
    horovod_version=$(docker run --rm ${tag} pip freeze | grep ^horovod= | awk -F== '{print $2}')
    tensorflow_version=$(docker run --rm ${tag} pip freeze | grep ^${tensorflow_pkg}= | awk -F== '{print $2}')
    pytorch_version=$(docker run --rm ${tag} pip freeze | grep ^torch= | awk -F== '{print $2}' | awk -F+ '{print $1}')
    mxnet_version=$(docker run --rm ${tag} pip freeze | grep ^mxnet | awk -F== '{print $2}')
    final_tag=horovod/horovod:${horovod_version}-tf${tensorflow_version}-torch${pytorch_version}-mxnet${mxnet_version}-py${py}-${device}
    docker tag ${tag} ${final_tag}
    docker rmi ${tag}
}

# clear upstream images, ok to fail if images do not exist
docker rmi $(cat Dockerfile.gpu | grep FROM | awk '{print $2}') || true
docker rmi $(cat Dockerfile.cpu | grep FROM | awk '{print $2}') || true

# build for py2 and py3, cpu and gpu
build_one 3.6 gpu
build_one 3.6 cpu

# print recent images
docker images horovod/horovod
