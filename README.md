# Fruit classifier

> This is repo is one of the assignments given at the AI workshop on 
pre-processing given the 23rd of April 2019

This assignment is loosely based on 
[this tutorial](https://www.pyimagesearch.com/2017/12/18/keras-deep-learning-raspberry-pi/)
from `pyimagesearch`.


## Task

In this assignment you have to do distinguish various kinds of fruit.
No dataset is given, so you either have to bring your own dataset or 
scrape something of the internet.
In any case you will need to clean and pre-process the data prior to 
model ingestion.

Furthermore, you should make your own test set (with your mobile 
camera) based on fruit in your local vicinity (or any other object you 
choose to classify). Note that a couple of photos for each class should 
suffice for this toy-model. 

### Pre-processing

In the pre-processing part of the assignment your task is to enhance 
the code in `fruit_classifier/preprocessing.`

## Usage

1. Download the requirements in [requirements.txt](requirements.txt)
2. Download the [chromedriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)
   based on your operating system
3. Scrape images with `python -m fruit_classifier.datascraping`
4. Clean the data with `python -m fruit_classifier.preprocessing`
5. Train with `python -m fruit_classifier.train`
6. Predict with `python -m fruit_classifier.predict <path_to_image>`


## Troubleshooting
**Question**: I've done all the assignments and have literally 
nothing to do

**Answer**: Congratulations! Now you can try to locate the class in 
the image (also known as object detection). You can use tools like 
[labelImg](https://github.com/tzutalin/labelImg) to do this. *NOTE*: 
We only ask you to label, if you are really eager you could also try 
to change the model so that you get a bounding box, but note that 
this is not a task which is done in 4 hours (see for example [the 
original YOLO paper](https://arxiv.org/abs/1506.02640) if you are 
interested).

---

**Question**: I've gone through the painful experience of labeling 
data, what do I do now?

**Answer**: You may have noticed that there are several places where 
the code can be improved. Why don't you give it a try and make a pull
request with your improvements.
