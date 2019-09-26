# Sacred experiments

This assignment is all about bookkeeping your experiments. One way of
doing this is by using 
[sacred](https://github.com/IDSIA/sacred) visualized by
[omniboard](https://github.com/vivekratnavel/omniboard).

> NOTE: This repo has aimed at separating the experiments from the
> rest of the logic in the code. The setup here may therefore divert
> from what is found in the 
> [sacred documentation](https://sacred.readthedocs.io/en/stable/) 

It's worth noting that a lot of hyperparameter tuning can be done
automatically these days, a lot of experimenting is done prior to the
tuning (in order not to make the tuning very expensive). 

In addition, we're usually tuning more than just the model. Consider
a pipe where different means of preprocessing can be done before
training the model, or that you'd like a quick overview of what model
type makes the most sense (linear classifier vs. neural net etc.). 
In such a pipe there are so many combinations that it would be
impractical to search through them all. What you usually want is a
rough idea of what works, and what range you should do your parameter
search in.

And finally, you usually make changes to your code as you go. You may
be asking yourself "Did I run that experiment which went so welll
before or after I added the new preprocessing with the new exciting
library?".

## Getting started

We will run [sacred](https://github.com/IDSIA/sacred) and
[omniboard](https://github.com/vivekratnavel/omniboard) 
through containers. The easiest way to do this is through 
[docker-compose](https://docker-curriculum.com/#docker-compose).

In the root directory of this repository run

```bash
cd docker
docker-compose up -d
```

This will start the containers, setup a network which the containers
can communicate through and setup directories shared with the host
machine and the containers.

In order to enter the `fruit_classifier` container type

```bash
docker exec -i -t docker_fruit_classifier_1 /bin/bash
```

You should now be inside of the shell of the container, and you can
run terminal commands. In order to run the first experiment, type

```bash
python3 -m experiments
```

Head over to [omniboard](http://0.0.0.0:9000) to check the results of
your run.

Once you are done playing around with this exercise, head back to the
`docker` directory of this repository (on the host machine) and type

```bash
docker-compose down -v
```

## Assignments

Apart from the assignments below, feel free to try your software
 engineering skills by assigning yourself to one of the tasks in the 
 [issue-tracker](https://github.com/acntech/fruit-classifier/issues)

1. Make a new experiment in the `experiment_files` directory where you
 change any parameter but `model_type`
2. Make a new model architecture in `fruit_classifier/models/models.py`,
 and add it to the factory in `fruit_classifier/models/models.py`
 **NOTE**: You will have to build the `Dockerfile` again, and map the
  new image in the `docker-compose.yml` file
3. Try to host the `mongodb` database on your cloud provider, and
 share it with your friends
4. Rewrite the code so that it's possible to change the parameters in 
`fruit-classifier/fruit_classifier/preprocessing/preprocessing_utils.py`
5. Add more diagnostics (metrics, plots etc.) to the recipe in 
`fruit-classifier/experiments/recipe.py`
