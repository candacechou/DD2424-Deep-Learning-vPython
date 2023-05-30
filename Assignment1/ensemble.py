
import pickle
import numpy as np
import argparse 
import os
import matplotlib.pyplot as plt
from trainer import load_batch, NNmodel
from scipy.stats import mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type = int, default = 10,help="number of epoch")
    parser.add_argument('--num_ensemble', type = int, default = 10,help="number of emsemble")
    parser.add_argument('--outdir', default='./result', type= str, help = " the directory to save the output plot")
    parser.add_argument('--softVote', default= 1, type= bool, help = "Voting scheme : soft for True, hard for False")
    args = parser.parse_args()
    
    config = {}
    config["batch_size"] = [1,2,4,8,16,32,64,128,256]
    config["lr"] = [10e-1,10e-2,10e-3,10e-4,10e-5]
    config["lmda"] = [0,10e-1,10e-2,10e-3,10e-4,10e-5]
    config["initial"] = ["Xavier", None]
    config["lr_scheduler"] = [True, False]
    
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    
    train_filename = ["../Dataset/cifar/data_batch_1","../Dataset/cifar/data_batch_3","../Dataset/cifar/data_batch_4","../Dataset/cifar/data_batch_5"]
    val_filename = "../Dataset/cifar/data_batch_2"
    test_filename = "../Dataset/cifar/test_batch"

    filenames = {}
    filenames["train"] = train_filename
    filenames["val"] = val_filename
    
    testX, testy,testY = load_batch(test_filename)
    
    ensemble_test = []
    
    for i in range(args.num_ensemble):
        batch_size = np.random.choice(config["batch_size"])
        lr = np.random.choice(config["lr"])
        lmda = np.random.choice(config["lmda"])
        initial = np.random.choice(config["initial"])
        lr_scheduler = np.random.choice(config["lr_scheduler"])
        print(f"training {i}th ensemble model... batch size {batch_size}, lr {lr} and L2-regularization {lmda}")
        trainer = NNmodel(filenames,
                    batch_size = batch_size, 
                    epoch = args.epoch,
                    lr = lr,
                    lmda = lmda,
                    initial = initial,
                    outdir = args.outdir,
                    lr_decay = lr_scheduler)

        trainer.train()
        testP = trainer.evaluateClassifier(testX)
        if  args.softVote:
            ensemble_test.append(testP)
        else:
            ensemble_test.append(np.argmax(testP, axis = 0))
    
    if args.softVote:
        ensemble_test = np.sum(ensemble_test,axis=2)
        prediction = np.argmax(ensemble_test, axis = 0)
    else:
        ensemble_test = np.array(ensemble_test)
        prediction = mode(ensemble_test,axis = 0)[0]
    
    TEST_ACC = trainer.computeAccuracy(testy,prediction)

    
    print("test accuracy: ",TEST_ACC,"%")
   

    
if __name__ == "__main__":
    main()