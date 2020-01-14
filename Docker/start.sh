#!/bin/bash

docker run --detach \
    --hostname localhost \
    --publish 8080:8080 \
    --name "predictorAlg1" \
    --restart always \
    taip/predictorAlg1