# Assignment 1 
## One-layer neural network 

Dataset:  CIFAR10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html))

In this assignment, A one-layer neural network is applied. 

## Gradient Descent
Gradient Descent is an optimization algorithm which is commonly used to train the machine learning algorithm and neural networks. Learning from the training data, the model aims to minimize / maximize the cost function.

Gradient Descent finds the minimum in an iterative fashion by moving in the direction of steepest descent. Ideally, it will converge to a global minimum, where for a convex optimization problem or neural network problem, it is fine to look for a local minimum as a global minimun. 

### Batch Gradient Descent
Batch gradient descent computes the loss for all training data points and updates the model after evaluating all examples. Saying it in a simple way, it updates the model once per epoch using the complete training dataset.

Batch gradient descent is suitable for convex problems or error manifolds that are relatively smooth. In such cases, we can directly update the weights towards the minimum. However, when dealing with large training datasets, batch gradient descent has certain drawbacks: 

1. Computing gradient is both time and space consuming. 
2. It takes an age to find the local minimum. (slow convergence)
3. It doesn't generalize well compare to stochastic gradient descent.

### Stochastic Gradient Descent 
Unlike Batch Gradient Descent, Stochastic gradient descent update the model with only one training data at each update step. In practice, Stochastic Gradient Descent converges a lot faster than batch gradient descent, and it provides better generalization.

On the other hand, Stochastic Gradient Descent provides a much noisy estimate of gradient since it only consider one training data at each update. The cost fluctuates over the whole training dataset and it never reaches the minimum but keep dancing around it. 

### Mini-Batch Gradient Descent.
Mini-batch gradient descent is a technique that leverages the benefits of both batch gradient descent and stochastic gradient descent. During each epoch of the training process, the dataset is divided into smaller batches, and the model is updated using each batch. This approach allows us to strike a balance between the efficiency of stochastic gradient descent and the accuracy of batch gradient descent.

## Loss

In this assignment, we test 2 different kinds of loss: Cross-Entropy Loss and Hinge Loss. 


## Learning Rate
Selecting an appropriate learning rate is a crucial factor in model optimization. A lower learning rate may result in slower convergence, but it produces a smoother update trajectory. Conversely, a higher learning rate can lead to divergence during optimization and create an inefficient zig-zag update path.

### Learning Rate Decay

In assignment 1, I implemented a simple learning rate decay. If `lr_schedule` is enable, then the learning rate would decay with a factor of 0.999 at the beginning of each epoch. 

## Weight Initialization
Using a suitable weight initialization technique can aid in training a high-quality model by minimizing the risk of gradient vanishing during back-propagation, ensuring consistent variance of input and output across all layers. In this assignment, I opted for the Xavier Initialization to initialize the weight. 

### Xavier Initialization


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



