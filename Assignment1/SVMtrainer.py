
import pickle
import numpy as np
import argparse 
import os
import matplotlib.pyplot as plt


def load_batch(filename):
    
    """
    X: np.narray(D,N)
    y: label
    Y: one-hot encoding matrix
    """

    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b"data"] / 255
        X = X.T
        y = dict[b"labels"]
        Y = np.eye(10)[y].T

    return X, y, Y

class SVMmodel():
    def __init__(self, filename,
                       batch_size = 64,
                       epoch = 10,
                       lr = 0.01,
                       lmda = 0,
                       initial = True,
                       shuffle = False,
                       lr_decay = True,
                       outdir = 'result'):

        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.lmda = lmda
        
        ### dataset 
        self.k = 10
        self.datasets = {}
        self.shuffle = shuffle
        self.lr_decay = lr_decay
        self._prepareDataset(filename)

        ### some parameters
        self.d = self.datasets["Xval"].shape[0]
        self.n = self.datasets["Xtrain"].shape[1]
        ### model
        self._weightInitialization(initial)

        ### save 
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        
        self.out_dir = outdir



    def _prepareDataset(self,filenames):
        
        print("prepare training dataset")
        self.datasets["Xtrain"], self.datasets["ytrain"],self.datasets["Ytrain"] = load_batch(filenames["train"])
        print("prepare validation dataset")
        self.datasets["Xval"], self.datasets["yval"],self.datasets["Yval"] = load_batch(filenames["val"])

    

    def _weightInitialization(self,initial=None):

        '''
        d : dimensionality of the data
        k : number of classes
        '''
        if initial == "Xavier":
            print("using Xavier Initialization...")
            self.weight = np.random.normal(0,1/np.sqrt(self.d),(self.k,self.d))
            
        else:
            self.weight = np.random.normal(0,0.01,(self.k,self.d))
        self.bias = np.random.normal(0,0.01,(self.k,1))
    
    def computeGradient(self,X, Y, P):

        '''
        Input: X, Y, P
        X : d x n
        Y : one-hot ground truth label. (k x n)
        P : output from evaluateClassifier. k x n
        
        Update: grad_w, grad_b
        grad_w : gradient of W, k x d
        grad_b : gradient of b, k x 1
        '''

        d, n = X.shape
        k, n = P.shape
        assert n == P.shape[1], "wrong P batch"
        
        self.grad_w = self.lmda * self.weight
        self.grad_b = np.zeros((k,1)) 

    

        ### calculate y(W.Tx + b)  = y * P < 1 -> 
        ### TODO: Write in Matrix form ?
        for i in range(n):
            x = X[:,i]
            y = np.where(Y[:, [i]].T[0] == 1)[0][0]
            p = P[:,i]
            if P[y,i] < 1:
                for j in range(self.k):
                    if j != y:
                        if max(0,P[j,i] - P[j,y] + 1) != 0:
                            self.grad_w[j] += x
                            self.grad_w[y] -= x
                            self.grad_b[j, 0] += 1
                            self.grad_b[y, 0] += -1
        
        self.grad_w /= n
        self.grad_b /= n
        
       
    def lrScheduler(self,method = "decay"):
        if method == "decay" :
            self.lr *= 0.999

    def computeCost(self,X, Y):

        """
        Input: X, Y, W, b, lmda
        - X : d x n
        - Y : one-hot ground truth label : k x n 
        - W : weight : k x d
        - b : bias : k x 1
        - lmda : the L2 regularization
        """
        
        
        p = self.evaluateClassifier(X) ## k x n
        sj = np.max(p * Y, axis = 0) ## 1 x n
        ## broadcasting
        mg = p - sj + 1 
        
        ## j != the true label for loss
        mg[Y==1] = 0
        mg[mg < 0] = 0
        J = np.sum(mg)/X.shape[1] + self.lmda * (np.sum(self.weight**2))
        
        return J
    def updateWeight(self):
        self.weight -= self.lr * self.grad_w
        self.bias -= self.lr * self.grad_b
        
    def train(self):

        for epoch_i in range(self.epoch):
            print(f"=== at {epoch_i}th epoch ===")
            if self.shuffle:
                idx = list(range(self.datasets["Xtrain"].shape[1]))
                np.random.shuffle(idx)
                self.datasets["Xtrain"] = self.datasets["Xtrain"][:,idx]
                self.datasets["Ytrain"] = self.datasets["Ytrain"][:,idx]
                self.datasets["ytrain"] = np.array(self.datasets["ytrain"])[idx]

            if self.lr_decay:
                self.lrScheduler()
            
            for i in range(0,self.n, self.batch_size):
                P = self.evaluateClassifier(self.datasets["Xtrain"][:,i:i+self.batch_size]) ## k x n
                self.computeGradient(self.datasets["Xtrain"][:,i:i+self.batch_size], self.datasets["Ytrain"][:,i:i+self.batch_size], P)
                self.updateWeight()
            ### calculate the cost and accuracy for both training and validtion data
            train_cost_temp = self.computeCost(self.datasets["Xtrain"], self.datasets["Ytrain"])
            val_cost_temp = self.computeCost(self.datasets["Xval"], self.datasets["Yval"])

            P_train = self.evaluateClassifier(self.datasets["Xtrain"])
            train_acc_temp = self.computeAccuracy(self.datasets["ytrain"], P_train)

            P_val = self.evaluateClassifier(self.datasets["Xval"])
            val_acc_temp = self.computeAccuracy(self.datasets["yval"], P_val)
            print(f"training acc : {train_acc_temp} %, validation acc: {val_acc_temp}%")
            print(f"training cost : {train_cost_temp}, validation cost: {val_cost_temp}")
            print("---------------------------------------------------------------------")
            self.train_loss.append(train_cost_temp)
            self.val_loss.append(val_cost_temp)
            self.train_acc.append(train_acc_temp)
            self.val_acc.append(val_acc_temp)
        

    def evaluateClassifier(self,X):

        """
        Input: X, W, b
        - each column of X corresponds to an image and it has size d × n.
        - W and b are the parameters of the network. (W = k x d ; b = k x 1)
        Output: P
        - each column of P contains the probability for each label for the image in the corresponding column of X. P has size K × n.
        """
        
        
        
        s = np.matmul(self.weight,X) + self.bias ### k x n
        
        return s

    def plotResult(self):### plot the result
        print("plotting the result......")
        filename = os.path.join(self.out_dir,"accuracy.jpg")
        plt.figure()
        plt.plot(range(self.epoch), self.train_acc, label="train")
        plt.plot(range(self.epoch), self.val_acc, label = "val")
        plt.title("The accuracy of each epoch")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend()
        plt.savefig(filename)

        filename = os.path.join(self.out_dir,"loss.jpg")
        plt.figure()
        plt.plot(range(self.epoch), self.train_loss, label="train")
        plt.plot(range(self.epoch), self.val_loss, label = "val")
        plt.title("The cost of each epoch")
        plt.xlabel("epoch")
        plt.ylabel("loss")
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
        return round(np.sum(correct) / len(correct) * 100,6)

    def Inference(self, testX,testy,testY):

        P = self.evaluateClassifier(testX)
        acc = self.computeAccuracy(testy, P)
        return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 2, help= "batch size")
    parser.add_argument('--epoch', type = int, default = 10,help="number of epoch")
    parser.add_argument('--lr', default = 0.01, type=float, help = "learning rate")
    parser.add_argument('--lmda', default = 0.01, type=float, help = "parameter of L2 regularization")
    parser.add_argument('--outdir', default='./result', type= str, help = " the directory to save the output plot")
    parser.add_argument('--Xavier', default= 0, type= bool, help = " Use Xavier Initialization or not. (bonus)")
    parser.add_argument('--shuffle', default= 0, type= bool, help = "Shuffle the input image or not. (bonus)")
    parser.add_argument('--lr_scheduler', default= 0, type= bool, help = "Do learning rate scheduler or not. (bonus)")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    train_filename = "../Dataset/cifar/data_batch_1"
    val_filename = "../Dataset/cifar/data_batch_2"
    test_filename = "../Dataset/cifar/test_batch"

    filenames = {}
    filenames["train"] = train_filename
    filenames["val"] = val_filename
    init_method = "Xavier" if args.Xavier else None
    trainer = SVMmodel(filenames,batch_size = args.batch_size, epoch = args.epoch,
                       lr = args.lr,
                       lmda = args.lmda,
                       initial = init_method,
                       shuffle = args.shuffle,
                       outdir = args.outdir)

    trainer.train()
    trainer.plotResult()

    ### do inference
     
    testX, testy,testY = load_batch(test_filename)
    TEST_ACC = trainer.Inference(testX,testy,testY)
    print("test accuracy: ",TEST_ACC,"%")
   

    
if __name__ == "__main__":
    main()