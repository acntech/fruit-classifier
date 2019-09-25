# Fruit classifier

[![Build Status](https://travis-ci.org/acntech/fruit-classifier.svg?branch=master)](https://travis-ci.org/acntech/fruit-classifier.svg)
[![codecov](https://codecov.io/gh/acntech/fruit-classifier/branch/master/graph/badge.svg)](https://codecov.io/gh/acntech/fruit-classifier)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)

> This is repo is part of an assignment series.
> See [assignments](assignments) for details

This assignment is loosely based on
[this tutorial](https://www.pyimagesearch.com/2017/12/18/keras-deep-learning-raspberry-pi/)
from `pyimagesearch`.

## Usage

1. Download the requirements in [requirements.txt](requirements.txt)
2. Download the [chromedriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)
   based on your operating system, and store it to the root directory
   of this repository
3. Scrape images with `python -m fruit_classifier.data_scraping`
4. Clean the data with `python -m fruit_classifier.preprocessing`
5. Train with `python -m fruit_classifier.train`
6. Predict with `python -m fruit_classifier.predict -i <path_to_image>`
 
   Unix-example (in windows replace `/` with `\`): 
   `python -m fruit_classifier.predict -i "test/data/raw/bananas/1. banana-1.png"`
