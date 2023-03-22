# DD2424 Deep Learning for Data Science 

This repository is the python version of assignment on DD2424 (2019 spring). The purpose of this repository is to revise myself all the fundamental knowledge of deep learning. 


## Four Assignments
Each assignment includes not only the requirement of the assignment, but also some important knowledge / topic of deep learning.
1. Assignment 1: A one-layer neural network applied to CIFAR10
    - SGD / minibatch.
    - Learning rate decay
    - Softmax Cross-Entropy loss
    - Hinge loss 
    - Xavier Initialization
    - Ensemble Learning
2. Assignment 2: A two-layer neural network applied to CIFAR10
    - He Initialization
    - ReLu Activation Function
    - Cyclical Learning Rate 
3. Assignment 3: A k-layer neural network with batch normalization 
applied to CIFAR10
    - Batch Normalization
    - Optimizer : Adam.
4. Assignment 4: A vanilla RNN to synthesize English text characters trained on The Goblet of Fire by J.K. Rowling
    - Optimizer : Adagrad, RMSProp.
    - Gradient Clipping.

## Datasets

In assignment 1, 2 and 3, the dataset CIFAR10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html))  is used. Download CIFAR-10 python version, and unzip it. In the assignments, only parts of the dataset is used.

For Assignment 1:

    - training data: data_batch_1.
    - validation data: data_batch_2.
    - test data: test_batch.

For Assignment 2 and 3:

    - training data: data_batch_1, data_batch_3, data_batch_4,data_batch_5.
    - validation data: data_batch_2.
    - test data: test_batch.
data_batch_1, data_batch_2 and test_batch should be put under the directory of ```./Dataset/cifar```.

For Assignment 4:
    - The Goblet of Fire by J.K. Rowling.
    - The txt file of the book should be put under the directory of ```./Dataset/JK-Rowling```.

## Model Training

### Assignment 1 

To train the model, please run :

```
python trainer.py --batch_size BATCHSIZE --epoch EPOCH --lr LR --lmda LAMBDA --outdir <OUTDIR_PATH> --Xavier True --shuffle False --lr_scheduler True
```

For example :

```
python trainer.py --batch_size 100 --epoch 40 --lr 0.01 --lmda 0.001 --outdir ./result --Xavier True --shuffle False ---lr_scheduler True
```

#### Assignment 1: Bonus

Train a model with a multi-class hinge loss.

Run 

```
python SVMtrainer.py --batch_size BATCHSIZE --epoch EPOCH --lr LR --lmda LAMBDA --outdir <OUTDIR_PATH> --Xavier True --shuffle False --lr_scheduler True
```

For example 

```
python SVMtrainer.py --batch_size 200 --epoch 20 --lr 0.001 --lmda 0.0001 --outdir ./result --Xavier True --shuffle False --lr_scheduler True
```
#### Assignment 1: Ensemble Learning

Run

```
python ensemble.py --epch EPOCH --num_ensemble NUM_ENSEMBLE --outdir <OUTDIRPATH> --softVote False 
```

where **NUM_ENSEMBLE** is the number of model we'd like to train for the voting. The details of the algorithm can be found in ```./Assignment1/README.md```

For example:

```
python3 ensemble.py --epch 20 --num_ensemble 20 --outdir ./result --softVote False 

```
### Assignment 2

Run 
```
python3 trainer.py --batch_size BATCH_SIZE --epoch EPOCH --min_lr MIN_LR --max_lr MAX_LR --n_s N_S --lmda  LAMBDA  --layer_num NUM_LAYER --He True --outdir <OUTDIRPATH> --Dropout True
```

For example:

```
python3 trainer.py --batch_size 200 --epoch 20 --min_lr 1e-5 --max_lr 1e-3 --n_s 5 --lmda 0.001 --layer_num 40 --He True --Dropout True --outdir ./result/ --He True
```

### Assignment 3 

Run 
```
python3 trainer.py --batch_size BATCH_SIZE --epoch EPOCH --lmda  LAMBDA  --num_layer NUM_LAYER --num_nodes NODE1 NODE2 ... --min_lr MIN_LR --max_lr MAX_LR --n_s N_S --Initialize INIT  --outdir <OUTDIRPATH> --Dropout True --Adam True
```

For example:

```
python3 trainer.py --batch_size 200 --epoch 20 --lmda 0.001 --Initialize He --outdir ./result --num_layer 5 --num_nodes 50 40 30 20  --min_lr 1e-4 --max_lr 1e-2 --n_s 2 --Dropout True --Adam True
```

### Assignement 4

Run 

```
python3 trainer.py --epoch EPOCH --lr LR --sig SIG --m M --seq_len SEQUENCE_LENGTH --text_len TEXT_LEN --optimizer adagrad
```

For example 

```
python3 trainer.py --epoch 10 --lr 0.001 --sig 0.01 --m 100 --seq_len 25  --text_len 200 --optimizer adagrad
```