#!/usr/bin/env bash

cd ..
echo $PWD
docker build -f docker/Dockerfile -t loeiten/fruit_classifier:latest .
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker docker push loeiten/fruit_classifier:latest
