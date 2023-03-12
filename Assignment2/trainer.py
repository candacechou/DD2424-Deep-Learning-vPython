
import pickle
import numpy as np
import argparse 
import os
import copy
import matplotlib.pyplot as plt
eps = 1e-8
def load_batch(filename):
    
    """
    X: np.narray(D,N)
    y: label
    Y: one-hot encoding matrix
    """
    if isinstance(filename,list):
        for i, file in enumerate(filename):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                X_batch = dict[b"data"] / 255
                X_batch = X_batch.T
                y_batch = dict[b"labels"]
                Y_batch = np.eye(10)[y_batch].T
            if i == 0 :
                X = X_batch
                Y = Y_batch
                y = y_batch
            else:
                X = np.concatenate((X,X_batch),axis =1)
                Y = np.concatenate((Y,Y_batch),axis =1)
                y += y_batch
    else:
        with open(filename,'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            X = dict[b"data"] / 255
            X = X.T
            y = dict[b"labels"]
            Y = np.eye(10)[y].T

    return X, y, Y

class NNmodel():
    def __init__(self, filename,
                       batch_size = 16,
                       epoch = 20,
                       min_lr = 1e-5,
                       max_lr = 1e-2,
                       n_s = 20,
                       lmda = 0,
                       dropout = True,
                       layer_num = 20,
                       He = True,
                       outdir = "./output"):

        self.batch_size = batch_size
        self.epoch = epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lmda = lmda
        self.layer_num = layer_num
        self.dropout = dropout
        
        
        ### dataset 
        self.k = 10
        self.datasets = {}
        self._prepareDataset(filename)

        ### some parameters
        self.d = self.datasets["Xval"].shape[0]
        self.n = self.datasets["Xtrain"].shape[1]
        ### model
        self.He = He
        self._weightInitialization()

        ### save 
        self.train_loss = []
        self.train_acc = []
        self.train_cost = []
        self.val_loss = []
        self.val_acc = []
        self.val_cost = []

        self.out_dir = outdir

        ### Cyclincal Learning rate
        self.n_s = n_s * int(self.n / self.batch_size) / 2
        self.l = 0
        self.lr = 0
        self.lr_list = []

    def _prepareDataset(self,filenames):
        
        print("prepare training dataset")
        self.datasets["Xtrain"], self.datasets["ytrain"],self.datasets["Ytrain"] = load_batch(filenames["train"])
        print("prepare validation dataset")
        self.datasets["Xval"], self.datasets["yval"],self.datasets["Yval"] = load_batch(filenames["val"])



    def _weightInitialization(self):
        ## since there are two layers, we store the weights and bias in dict
        ## with its key as number of layers
        self.weight = {}
        self.bias = {}
        self.grad_w = {}
        self.grad_b = {}

        self.weight[0] = np.random.normal(0,np.sqrt(2/self.d),(self.layer_num,self.d )) ## d x layer_num
        self.bias[0] = np.zeros((self.layer_num, 1)) ## layer_num x 1
        
        self.grad_w[0] = np.zeros((self.layer_num,self.d ))
        self.grad_b[0] = np.zeros((self.layer_num,1))

        self.weight[1] = np.random.normal(0,np.sqrt(2/self.layer_num),(self.k,self.layer_num)) ## layer_num x k
        self.bias[1] = np.zeros((self.k,1))
        
        self.grad_w[1] = np.zeros((self.k,self.layer_num))
        self.grad_b[1] = np.zeros((self.k,1))

        
    
    def computeGradient(self,X, Y,h, P):
        G = -(Y-P) ## k x n
        self.grad_w[1] = (1/X.shape[1]) * np.matmul(G,h.T) +  2 * self.lmda * self.weight[1]
        self.grad_b[1] = (1/X.shape[1]) * np.sum(G, axis = 1)

        self.grad_w[0] = np.matmul(self.weight[1].T,G)
        ind_h = h > 0
        g_batch = np.matmul(self.weight[1].T, G)
        

        self.grad_w[0] = (1/X.shape[1]) * np.matmul(g_batch * ind_h, X.T) 
        self.grad_w[0] += 2 * self.lmda * self.weight[0]
        self.grad_b[0] = (1/X.shape[1]) * np.sum(g_batch,axis = 1)
        
        self.grad_b[0] = self.grad_b[0][:,np.newaxis]
        self.grad_b[1] = self.grad_b[1][:,np.newaxis]


    def lr_scheduler(self):

        if self.update_step % (2 * self.n_s) == 0:
            self.l = self.l+1   

        if self.update_step >= 2 * self.l * self.n_s \
            and self.update_step <= (2*self.l+1) * self.n_s:
            
            self.lr = self.min_lr + (self.update_step - 2 * self.l * self.n_s) / self.n_s * (self.max_lr - self.min_lr)
       
        elif self.update_step >= (2*self.l+1) * self.n_s\
            and self.update_step <= 2*(self.l+1) * self.n_s:
            self.lr = self.max_lr - (self.update_step - (2 * self.l + 1) * self.n_s )/self.n_s * (self.max_lr - self.min_lr)
        
        else:
            self.lr = self.lr

        self.lr_list.append(copy.deepcopy(self.lr))

    
    def evaluateClassifier(self,X, train = False):
        s1 = np.matmul(self.weight[0],X) + self.bias[0] ### k x n
        h = np.maximum(0,s1)
        if train and self.dropout:
            p = np.random.binomial(1, (1 - 0.5), size=h.shape[1])
            h *= p
        s = np.matmul(self.weight[1],h) + self.bias[1]
        p = np.exp(s+eps) / np.sum(np.exp(s+eps),axis=0)
        
        return p, h
        
       
    def computeCost(self,X,Y):
        p,_ = self.evaluateClassifier(X,False) ## k x n 
        J = (1/X.shape[1]) * -np.sum(Y * np.log(p+eps)) 
        return J

    def computeLoss(self,X, Y, train = False):

        p,_ = self.evaluateClassifier(X,train) ## k x n
        
        J = (1/X.shape[1]) * -np.sum(Y * np.log(p+eps))
        J += self.lmda * (np.sum(self.weight[0]**2))
        J += self.lmda * (np.sum(self.weight[1]**2))

        return J

    def updateWeight(self):
        self.weight[0] -= self.lr * self.grad_w[0]
        self.weight[1] -= self.lr * self.grad_w[1]
        self.bias[0] -= self.lr * self.grad_b[0]
        self.bias[1] -= self.lr * self.grad_b[1]

    def train(self):
        self.update_step = 0
        for epoch_i in range(self.epoch):
            print(f"=== at {epoch_i}th epoch ===")
            
            idx = list(range(self.datasets["Xtrain"].shape[1]))
            np.random.shuffle(idx)
            self.datasets["Xtrain"] = self.datasets["Xtrain"][:,idx]
            self.datasets["Ytrain"] = self.datasets["Ytrain"][:,idx]
            self.datasets["ytrain"] = np.array(self.datasets["ytrain"])[idx]
            
            for i in range(0,self.n, self.batch_size):
                self.update_step += 1
                self.lr_scheduler()
                P,h = self.evaluateClassifier(self.datasets["Xtrain"][:,i:i+self.batch_size], train = True) ## k x n
                self.computeGradient(self.datasets["Xtrain"][:,i:i+self.batch_size], self.datasets["Ytrain"][:,i:i+self.batch_size],h, P)
                
                
                ### calculate the cost and accuracy for both training and validtion data
                train_cost_temp = self.computeCost(self.datasets["Xtrain"], self.datasets["Ytrain"])
                val_cost_temp = self.computeCost(self.datasets["Xval"], self.datasets["Yval"])

                train_loss_temp = self.computeLoss(self.datasets["Xtrain"], self.datasets["Ytrain"])
                val_loss_temp = self.computeLoss(self.datasets["Xval"], self.datasets["Yval"])
                
                P_train,_ = self.evaluateClassifier(self.datasets["Xtrain"])
                train_acc_temp = self.computeAccuracy(self.datasets["ytrain"], P_train)

                P_val,_ = self.evaluateClassifier(self.datasets["Xval"])
                val_acc_temp = self.computeAccuracy(self.datasets["yval"], P_val)
                
                                
                self.train_cost.append(train_cost_temp)
                self.val_cost.append(val_cost_temp)
                self.train_loss.append(train_loss_temp)
                self.val_loss.append(val_loss_temp)
                self.train_acc.append(train_acc_temp)
                self.val_acc.append(val_acc_temp)

            print(f"training acc : {train_acc_temp} %, validation acc: {val_acc_temp}%")
            print(f"training cost : {train_cost_temp}, validation cost: {val_cost_temp}")
            print(f"training loss : {train_loss_temp}, validation loss: {val_loss_temp}")
            print("---------------------------------------------------------------------")


    

    def plotResult(self):### plot the result
        print("plotting the result......")
        filename = os.path.join(self.out_dir,"accuracy.jpg")
        plt.figure()
        plt.plot(range(len(self.train_acc)), self.train_acc, label="train")
        plt.plot(range(len(self.val_acc)), self.val_acc, label = "val")
        plt.title("The accuracy of each iteration")
        plt.xlabel("iteration")
        plt.ylabel("acc")
        plt.legend()
        plt.savefig(filename)

        filename = os.path.join(self.out_dir,"loss.jpg")
        plt.figure()
        plt.plot(range(len(self.train_loss)), self.train_loss, label="train")
        plt.plot(range(len(self.val_loss)), self.val_loss, label = "val")
        plt.title("The loss of each iteration")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()
        plt.legend()
        plt.savefig(filename)

        filename = os.path.join(self.out_dir,"cost.jpg")
        plt.figure()
        plt.plot(range(len(self.train_cost)), self.train_cost, label="train")
        plt.plot(range(len(self.val_cost)), self.val_cost, label = "val")
        plt.title("The cost of each iteration")
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.legend()
        plt.savefig(filename)

        filename = os.path.join(self.out_dir,"lr.jpg")
        plt.figure()
        plt.plot(range(len(self.lr_list)), self.lr_list)
        plt.title("The learning rate of each iteration")
        plt.xlabel("iteration")
        plt.ylabel("lr")
        plt.legend()
        plt.savefig(filename)


    def computeAccuracy(self, GroundTruth, predict):

        """
        Input :
            GroundTruth = n x 1
            predict = k x n
        Output:
            Accuracy: correct / # of data
        
        """
        if predict.shape[0] != 1 :
            prediction = np.argmax(predict, axis = 0)

        correct = prediction == GroundTruth
        return round(np.sum(correct) / len(correct) * 100,4)

    def Inference(self, testX,testy,testY):

        P,_ = self.evaluateClassifier(testX,False)
        acc = self.computeAccuracy(testy, P)
        return acc








def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 2, help= "batch size")
    parser.add_argument('--epoch', type = int, default = 10,help="number of epoch")
    parser.add_argument('--min_lr', default = 0.0001, type=float, help = "the minimum learning rate")
    parser.add_argument('--max_lr', default = 0.01, type=float, help = "the maximum learning rate")
    parser.add_argument('--n_s', default = 2, type=int, help = "Hyper-Parameter of cycle during training.")
    parser.add_argument('--lmda', default = 0.01, type=float, help = "parameter of L2 regularization")
    parser.add_argument('--outdir', default='./result', type= str, help = " the directory to save the output plot")
    parser.add_argument('--layer_num', default= 50, type= int, help = " the number of nodes on the 2-layer.")
    parser.add_argument('--He', default= 0, type= bool, help = " Use Xavier Initialization or not. (bonus)")
    parser.add_argument('--Dropout', default= 1, type= bool, help = "Apply dropout or not. (bonus)")
    
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    train_filename = ["../Dataset/cifar/data_batch_1","../Dataset/cifar/data_batch_3","../Dataset/cifar/data_batch_4","../Dataset/cifar/data_batch_5"]
    val_filename = "../Dataset/cifar/data_batch_2"
    test_filename = "../Dataset/cifar/test_batch"

    filenames = {}
    filenames["train"] = train_filename
    filenames["val"] = val_filename

    trainer = NNmodel(filenames,
                      batch_size = args.batch_size,
                      epoch = args.epoch,
                      min_lr = args.min_lr,
                      max_lr = args.max_lr,
                      n_s = args.n_s,
                      lmda = args.lmda,
                      dropout =True,
                      layer_num = args.layer_num,
                      He = args.He,
                      outdir = args.outdir)

    trainer.train()
    trainer.plotResult()

    ### do inference
     
    testX, testy,testY = load_batch(test_filename)
    TEST_ACC = trainer.Inference(testX,testy,testY)
    print("test accuracy: ",TEST_ACC,"%")
   

    
if __name__ == "__main__":
    main()