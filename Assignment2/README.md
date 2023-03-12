# Assignment 2 
## two-layer Neural Network 

Dataset:  CIFAR10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html))

In this assignment, A two-layer neural network with a ReLu activation function is applied. 



## Mini-Batch Gradient Descent.

In this assignment, we applied mini-batch gradient descent, and batch size is set with `--batch_size`.

## Regularization
In this assignment, L2-Regularization, is applied with `--lmda`

## Loss
In this assignment, Cross-Entropy Loss are used.


## Learning Rate

### cyclical learning rates

In assignment 2, I implemented cyclical learning rate as my learning rate scheduler. One should set the hyper-parameter `--min_eta`, `--max_eta` and `--n_s` to the trainer.
- `--min_eta` is the minimum learning rate on cyclical learning rate.
- `--max_eta` is the maximum learning rate on the cyclical learning rate.
- `--n_s` is the number of cycle during training.

## Weight Initialization

Enable He Initialization with `--He`.


## Dropout

In this assignment, I also applied Dropout while training. If `--Dropout` with True, and 50% of nodes will be dropped during training.