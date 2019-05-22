# In the repo root, run
# docker build -f docker/Dockerfile -t fruit_classifier:latest .
# to build this image

# After successful build, it can be started with
# docker run --rm -d -p 5001:5000 fruit_classifier:latest

# You can also check the image by
# docker run --rm -it --entrypoint "/bin/bash" fruit_classifier:latest

# Use ubuntu as the base layer
FROM ubuntu:18.04

# Intall apt dependencies
RUN apt-get -yqq update
RUN apt-get -yqq upgrade
RUN apt-get -yqq install python3
RUN apt-get -yqq install python3-pip
RUN apt-get -yqq install libsm6
RUN apt-get -yqq install libxext6
RUN apt-get -yqq install libxrender-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /fruit-classifier/requirements.txt

# Set the working directory to fruit-classifier
WORKDIR /fruit-classifier

# Install the requirements through pip
RUN pip3 install -r /fruit-classifier/requirements.txt

# Copy the code to fruit-classifier
COPY . /fruit-classifier

# The ENTRYPOINT specifies a command that will always be executed when
# the container starts
ENTRYPOINT [ "python3" ]

# The CMD specifies arguments that will be fed to the ENTRYPOINT
CMD [ "-m", "app" ]