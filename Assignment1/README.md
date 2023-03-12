# Assignment 1 
## One-layer neural network 

Dataset:  CIFAR10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html))

In this assignment, A one-layer neural network is applied. 



## Mini-Batch Gradient Descent.
Mini-batch gradient descent is a technique that leverages the benefits of both batch gradient descent and stochastic gradient descent. During each epoch of the training process, the dataset is divided into smaller batches, and the model is updated using each batch.

In this assignment, we applied mini-batch gradient descent, and batch size is set with `--batch_size`.

## Regularization
In this assignment, L2-Regularization, is applied with `--lmda`
## Loss

In this assignment, Cross-Entropy Loss and Hinge Loss are tested.

Hinge Loss is implemented in `SVMtrainer.py`  and cross-entropy loss is implemented in `trainer.py`.


## Learning Rate

### Learning Rate Decay

In assignment 1, I implemented a simple learning rate decay. If `--lr_schedule` is enable, then the learning rate would decay with a factor of 0.999 at the beginning of each epoch. 

## Weight Initialization
Using a suitable weight initialization technique can aid in training a high-quality model by minimizing the risk of gradient vanishing during back-propagation, ensuring consistent variance of input and output across all layers. In this assignment, I opted for the Xavier Initialization to initialize the weight. 

Enable Xavier Initialization with `--Xavier`.


## Ensemble Learning (Homogenous)
In this assignment, I trained several models with different setting of hyper-parameters, then collected the prediction of each model and vote for the final result. There are two different ways to vote for the results: Hard Voting and Soft Voting.


### Hard Voting

The idea behind Hard Voting is easy: We count the time each class has the highest probability and consider the class with highest counting time as the result. 
 

| Hard Voting | Class 1 | Class 2 | Class 3 |
|---|---|---|---|
| Model 1 | 60% |  30% | 10% |
| Model 2 |  0% |  90% | 10% |
| Model 3 | 20% |  20% | 60% |
| Model 4 | 20% |  50% | 30% |
| Model 5 | 60% |  30% | 10% |
| Model 6 | 60% |  30% | 10% |
| Result| 3  |  2 |  1 |


For example, class 1 has highest probability in three models, while class 2 and class 3 have highest probability in two models and one model respectively. Therefore, we say the answer would be Class 1.

### Soft Voting

As for soft voting, we summed up the probability of all the model and found out that Class 2 has highest summation, hence we declared that the answer is Class 2.

| Soft Voting | Class 1 | Class 2 | Class 3 |
|---|---|---|---|
| Model 1 | 60% |  30% | 10% |
| Model 2 |  0% |  90% | 10% |
| Model 3 | 20% |  20% | 60% |
| Model 4 | 20% |  50% | 30% |
| Model 5 | 60% |  30% | 10% |
| Model 6 | 60% |  30% | 10% |
| Result| 220%  |  230% | 130%|



