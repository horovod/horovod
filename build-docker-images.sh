#!/bin/bash

set -e
set -x

function build_one()
{
    py=$1
    tag=horovod-build-py${py}:$(date +%Y%m%d-%H%M%S)
    docker build -t ${tag} --build-arg python=${py} --no-cache .
    horovod_version=$(docker run ${tag} pip freeze | grep ^horovod= | awk -F== '{print $2}')
    tensorflow_version=$(docker run ${tag} pip freeze | grep ^tensorflow-gpu= | awk -F== '{print $2}')
    pytorch_version=$(docker run ${tag} pip freeze | grep ^torch= | awk -F== '{print $2}')
    mxnet_version=$(docker run ${tag} pip freeze | grep ^mxnet | awk -F== '{print $2}')
    final_tag=horovod/horovod:${horovod_version}-tf${tensorflow_version}-torch${pytorch_version}-mxnet${mxnet_version}-py${py}
    docker tag ${tag} ${final_tag}
    docker rmi ${tag}
}

# clear upstream image, ok to fail if image does not exist
docker rmi $(cat Dockerfile | grep FROM | awk '{print $2}') || true

# build for py2 and py3
build_one 2.7
build_one 3.5

# print recent images
docker images horovod/horovod
