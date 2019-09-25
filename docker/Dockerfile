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
RUN echo "hello"
RUN apt-get -yqq update
RUN apt-get -yqq upgrade
RUN apt-get -yqq install python3
RUN apt-get -yqq install python3-pip
# Needed for open-cv
RUN apt-get -yqq install libsm6
RUN apt-get -yqq install libxext6
RUN apt-get -yqq install libxrender-dev
# Needed for chromedriver
RUN apt-get -yqq install wget
RUN apt-get -yqq install curl

# Install google chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -yqq update
RUN apt-get install -y google-chrome-stable

# Install chromedriver
RUN apt-get install -yqq unzip
# Use version 2.39 due to https://github.com/RobCherry/docker-chromedriver/issues/15
# RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/2.39/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /fruit-classifier/

# Set display port to avoid crash
ENV DISPLAY=:99

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