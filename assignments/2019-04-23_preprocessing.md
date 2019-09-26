# Preprocessing

In this assignment, we get to know the fruit-classifier. We will see
if different preprocessing steps can make the classifier better

## Getting started

1. Download `python3` ([anaconda](https://www.anaconda.com/distribution/) is preferred)
2. Make sure you have an editor you are comfortable with 
([pyCharm community](https://www.jetbrains.com/pycharm/download/)
 is recommended)
3. Download the requirements in `requirements.txt` (you do this by
 running the command `pip install -r requirements.txt` from the root
  directory of this repository)
4. Download the 
[chromedriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) 
 based on your operating system and your chrome version, and
 store it to the root directory of this repository (if you are on a
  windows operating system you may have to rename `chromedriver.exe`
  to `chromedriver`, or to change `chromedriver` in `agruments` in 
  `fruit_classifier/data_scraping/__main__.py`)
5. Scrape images with 
`python -m fruit_classifier.data_scraping`
6. Clean the data with 
`python -m fruit_classifier.preprocessing`
7. Train with `python -m fruit_classifier.train`
8. Predict with `python -m fruit_classifier.predict -i <path_to_image>`

## Assignments

The steps in [Getting started](#getting-started) can be time
 consuming enough for most. However, if you are up for some even more
 time consuming challenges you can

1. Rewrite the preprocessing module. What steps can be done to
 improve accuracy?
2. The code suffers from technical debt and lack of tests. Can you
 make it better?
3. We're impressed you even made it to this point. 
Now you can try to locate the classes in
the image (also known as object detection). You can use tools like
[labelImg](https://github.com/tzutalin/labelImg) to do this. 
**NOTE**:
We only ask you to label, if you are really eager you could also try
to change the model so that you get a bounding box, but note that
this is not a task which is done in 4 hours (see for example [the
original YOLO paper](https://arxiv.org/abs/1506.02640) if you are
interested)