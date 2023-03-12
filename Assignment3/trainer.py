
import pickle
import numpy as np
import argparse 
import os
import matplotlib.pyplot as plt
import copy

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
    def __init__(self, filenames,
                       batch_size = 64,
                       epoch = 20,
                       lmda = 0,
                       initial = "Random",
                       num_layer = 2,
                       num_nodes = [50],
                       min_lr = 1e-5,
                       max_lr = 1e-1,
                       n_s = 2,
                       dropout = True,
                       Adam = False,
                       outdir ="./Result"):

         ### dataset 
        self.k = 10
        self.datasets = {}
        self._prepareDataset(filenames)

        
        self.batch_size = batch_size
        self.epoch = epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.l = 0
        self.lr = 0
        self.lr_list = []

        self.lmda = lmda
        self.num_layer = num_layer
        self.num_nodes = num_nodes
        self.dropout = dropout

        self.d = self.datasets["Xval"].shape[0]
        self.n = self.datasets["Xtrain"].shape[1] 
        self.n_s = n_s * int(self.n / self.batch_size) / 2
        
        self.Adam = Adam
        
       
        ### some parameters
        self.d = self.datasets["Xval"].shape[0]
        self.n = self.datasets["Xtrain"].shape[1]
        
        self.update_step = 0

        ### save 
        self.train_loss = []
        self.train_acc = []
        self.train_cost = []
        self.val_loss = []
        self.val_acc = []
        self.val_cost = []
        
        self.out_dir = outdir
        self.lr_list = []
        ### weight and bias
        self.weight = {}
        self.bias = {}

        self.grad_w = {}
        self.grad_b = {}

        self.mu = {}
        self.vu = {}
        
        self.gamma = {}
        self.beta ={}

        self.grad_gamma = {}
        self.grad_beta = {}

        ### for backward pass
        self.x_batch = {}
        self.s_batch = {}
        self.shat_batch = {}

        self.mu_ave = {}
        self.vu_ave = {}

        if self.Adam:
            self.m_weight = {}
            self.v_weight = {}
            self.m_bias = {}
            self.v_bias = {}
            self.m_gamma = {}
            self.v_gamma = {}
            self.m_beta = {}
            self.v_beta = {}

        ### model
        self._weightInitialization(initial)

    def _prepareDataset(self,filenames):
        
        print("prepare training dataset")
        self.datasets["Xtrain"], self.datasets["ytrain"],self.datasets["Ytrain"] = load_batch(filenames["train"])
        print("prepare validation dataset")
        self.datasets["Xval"], self.datasets["yval"],self.datasets["Yval"] = load_batch(filenames["val"])

    

    def _weightInitialization(self,initial = "Random"):
        
        if initial == "Xavier":
            self.weight[1] = self.grad_w[1] = np.random.normal(0,np.sqrt(1/self.d),(self.num_nodes[0],self.d )) ## layer[0] x d
        elif initial == "He":
            self.weight[1] = self.grad_w[1] = np.random.normal(0,np.sqrt(2/self.d),(self.num_nodes[0],self.d )) ## layer[0] x d
        else:
            self.weight[1]= self.grad_w[1] = np.random.normal(0,0.01,(self.num_nodes[0],self.d )) ## layer[0] x d
        
        self.bias[1]=self.grad_b[1]=self.gamma[1]=self.beta[1]=self.grad_gamma[1]=self.grad_beta[1]=self.mu_ave[1]=self.vu_ave[1] = np.ones((self.num_nodes[0], 1))

        if self.Adam:
            self.m_weight[1]=self.v_weight[1] = np.zeros((self.num_nodes[0],self.d))

            self.m_bias[1]=self.v_bias[1]=self.m_beta[1]=self.v_beta[1]=self.m_gamma[1]=self.v_gamma[1] = np.zeros((self.num_nodes[0], 1))
        
        self.mu_ave[1] =  np.ones((self.num_nodes[0],1))
        self.vu_ave[1] =  np.ones((self.num_nodes[0],1))
        
    
        for i in range(2,self.num_layer):
            if initial == "Xavier":
                self.weight[i]=self.grad_w[i] = np.random.normal(0,np.sqrt(1/self.num_nodes[i-2]),(self.num_nodes[i-1],self.num_nodes[i-2]))
            elif initial == "He":
                self.weight[i]=self.grad_w[i] = np.random.normal(0,np.sqrt(2/self.num_nodes[i-2]),(self.num_nodes[i-1],self.num_nodes[i-2]))   
            else:
                self.weight[i]=self.grad_w[i] = np.random.normal(0,0.01,(self.num_nodes[i-1],self.num_nodes[i-2]))
            self.bias[i]=self.grad_b[i] = np.zeros((self.num_nodes[i-1],1) )  
            
            
            if i < self.num_layer :
                self.gamma[i] = self.beta[i]=self.grad_gamma[i]=self.grad_beta[i]=self.mu_ave[i]=self.vu_ave[i] = np.ones((self.num_nodes[i-1],1))
                

            if self.Adam:
                self.m_weight[i]=self.v_weight[i] = np.zeros((self.num_nodes[i-1],self.num_nodes[i-2]))
    
                self.m_bias[i]=self.v_bias[i] = np.zeros((self.num_nodes[i-1], 1))

                if i < self.num_layer:
                    self.m_beta[i] = self.v_beta[i]=self.m_gamma[i]=self.v_gamma[i] = np.zeros((self.num_nodes[i-1],1))

        ## the last layer
        if initial == "Xavier":
            self.weight[self.num_layer]=self.grad_w[self.num_layer] = np.random.normal(0,np.sqrt(1/self.num_nodes[self.num_layer-2]),(self.k,self.num_nodes[self.num_layer-2]))
        elif initial == "He":
            self.weight[self.num_layer]=self.grad_w[self.num_layer] = np.random.normal(0,np.sqrt(2/self.num_nodes[self.num_layer-2]),(self.k,self.num_nodes[self.num_layer-2]))
        else:
            self.weight[self.num_layer]=self.grad_w[self.num_layer] = np.random.normal(0,0.01,(self.k,1))
        self.bias[self.num_layer]=self.grad_b[self.num_layer] = np.zeros((self.k,1))

        if self.Adam:
                self.m_weight[self.num_layer]=self.v_weight[self.num_layer] = np.ones((self.k,self.num_nodes[self.num_layer-2]))
    
                self.m_bias[self.num_layer]=self.v_bias[self.num_layer] = np.zeros((self.k, 1))
            
    def batchNormBackPass(self,gbatch,sbatch,mu,vu):
        n = gbatch.shape[1]
        sigma_1 = np.power(vu + eps,-0.5)
        sigma_2 = np.power(vu + eps,-1.5)
        G_1 = np.multiply(gbatch , sigma_1@np.ones((1,n)))
        G_2 = np.multiply(gbatch , sigma_2@np.ones((1,n)))
        D = sbatch - mu@np.ones((1,n))
        c = np.multiply(G_2 ,D)@np.ones((n,1))
        gbatch = G_1 - (1/n)*G_1@np.ones((n,1)) - (1/n) * np.multiply(D, c@np.ones((1,n)))

        return gbatch

    def computeGradient(self,X, Y, P):
        
        n = X.shape[1]
        G_batch = - (Y - P)
        self.grad_w[self.num_layer] =(1/n) * G_batch@self.x_batch[self.num_layer].T+ 2 * self.lmda * self.weight[self.num_layer]
        ## k x node_n = (k x n) x ( node_n x n).T + K x node_n
        self.grad_b[self.num_layer] = (1/n) * G_batch@np.ones((n,1)) ## this simply means take the average of it,
        G_batch = np.matmul(self.weight[self.num_layer].T, G_batch)
        G_batch = G_batch * (self.x_batch[self.num_layer]>0)
        
        for i in range(self.num_layer - 1, 0, -1):  
            self.grad_gamma[i] = np.multiply(G_batch , self.shat_batch[i])@np.ones((n,1)) / n
            self.grad_beta[i] = G_batch@np.ones((n,1)) / n
            G_batch = np.multiply(G_batch , self.gamma[i]@np.ones((1,n)))
            
            G_batch = self.batchNormBackPass(G_batch,self.s_batch[i],self.mu[i],self.vu[i])
            self.grad_w[i] = (1/n) * np.matmul(G_batch , self.x_batch[i].T)+ 2 * self.lmda * self.weight[i]
            self.grad_b[i] = (1/n) * G_batch@np.ones((n,1))

            if i > 1 :

                G_batch = np.matmul(self.weight[i].T,G_batch)
                self.x_batch[i][self.x_batch[i]<0] = 0
                self.x_batch[i][self.x_batch[i]>0] = 1
                G_batch = np.multiply(G_batch,self.x_batch[i])
        
    def lrScheduler(self):
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
        

    def computeCost(self,X, Y):
        
        p = self.evaluateClassifier(X,False) ## k x n 
        J = (1/X.shape[1]) * -np.sum(Y * np.log(p + eps)) 
        
        return J

    def AdamUpdate(self,weight,m,v,grad):
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * np.power(grad,2)
        if self.update_step > 1 :
            mhat = m / (1-np.power(0.9,self.update_step-1))
            vhat = v / (1-np.power(0.999,self.update_step-1))
        else:
            mhat = m
            vhat = v
        weight -= self.lr * mhat / (np.sqrt(vhat) + eps)
        
        return weight, mhat, vhat

    def updateWeight(self):

        for i in range(1,self.num_layer+1):
            if self.Adam:
                self.weight[i],self.m_weight[i],self.v_weight[i] = self.AdamUpdate(self.weight[i],self.m_weight[i],self.v_weight[i],self.grad_w[i])
                self.bias[i],self.m_bias[i],self.v_bias[i] = self.AdamUpdate(self.bias[i],self.m_bias[i],self.v_bias[i],self.grad_b[i])
                
                if i < self.num_layer:
                    self.gamma[i],self.m_gamma[i],self.v_gamma[i] = self.AdamUpdate(self.gamma[i],self.m_gamma[i],self.v_gamma[i],self.grad_gamma[i])
                    self.beta[i],self.m_beta[i],self.v_beta[i] = self.AdamUpdate(self.beta[i],self.m_beta[i],self.v_beta[i],self.grad_beta[i])
            else:
                self.weight[i] -= self.lr * self.grad_w[i]
                self.bias[i] -= self.lr * self.grad_b[i]
                if i < self.num_layer:
                    self.gamma[i] -= self.lr * self.grad_gamma[i]
                    self.beta[i] -= self.lr * self.grad_beta[i]

        


    def computeLoss(self,X,Y):

        J = self.computeCost(X,Y)
        for w in self.weight:
            J += self.lmda * np.sum(w**2)
        return J

    def train(self):

        for epoch_i in range(self.epoch):

            print(f"=== at {epoch_i}th epoch ===")
            
            
            idx = list(range(self.datasets["Xtrain"].shape[1]))
            np.random.shuffle(idx)
            self.datasets["Xtrain"] = self.datasets["Xtrain"][:,idx]
            self.datasets["Ytrain"] = self.datasets["Ytrain"][:,idx]
            self.datasets["ytrain"] = np.array(self.datasets["ytrain"])[idx]
            
            for i in range(0,self.n, self.batch_size):
                self.update_step += 1
                self.lrScheduler()
                P = self.evaluateClassifier(self.datasets["Xtrain"][:,i:i + self.batch_size],True) ## k x n
                self.computeGradient(self.datasets["Xtrain"][:,i:i+self.batch_size], self.datasets["Ytrain"][:,i:i+self.batch_size], P)
                self.updateWeight()
                

                train_cost_temp = self.computeCost(self.datasets["Xtrain"], self.datasets["Ytrain"])
                val_cost_temp = self.computeCost(self.datasets["Xval"], self.datasets["Yval"])

                train_loss_temp = self.computeLoss(self.datasets["Xtrain"], self.datasets["Ytrain"])
                val_loss_temp = self.computeLoss(self.datasets["Xval"], self.datasets["Yval"])
                
                P_train = self.evaluateClassifier(self.datasets["Xtrain"],False)
                train_acc_temp = self.computeAccuracy(self.datasets["ytrain"], P_train)

                P_val  = self.evaluateClassifier(self.datasets["Xval"],False)
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
        
    def _batchNormalization(self,s1, i, train = True):
        ## s1 = nodes x n
        n = s1.shape[1]
        if train :
            alpha = 0.9
            # print("s1.shape:",s1.shape)
            cur_mu = np.mean(s1, axis=1, keepdims=True) ## n x 1
            cur_vu = np.var(s1, axis=1, keepdims=True) * (n-1) / n ## n x 1
 
            self.mu_ave[i] = alpha * self.mu_ave[i] + (1 - alpha) * cur_mu
            self.vu_ave[i] = alpha * self.vu_ave[i] + (1 - alpha) * cur_vu
            
            s1 = (s1 - cur_mu) / np.sqrt(cur_vu + eps)
            return s1,cur_mu,cur_vu
        else:
            s1 = (s1 - self.mu_ave[i]) / np.sqrt(self.vu_ave[i] + eps)
            return s1 
        
        

    def evaluateClassifier(self,X, train = False):
        ## first layer
        s1 =np.copy(X)
        self.x_batch[1]=s1
        
        
        for i in range(1,self.num_layer):
            s1 = np.matmul(self.weight[i], s1)  + self.bias[i]
            self.s_batch[i] = s1.copy()
            if train:
                if self.dropout:
                    p = np.random.binomial(1, (1 - 0.5), size=s1.shape[1])
                    s1 *= p
                s1,self.mu[i],self.vu[i] = self._batchNormalization(s1,i,train)
                
            else:
                s1 = self._batchNormalization(s1,i,train)
            
            self.shat_batch[i] = np.copy(s1)
            s1 = np.multiply(self.gamma[i], s1) + self.beta[i]
            s1 = np.maximum(0,s1)
            self.x_batch[i+1] = np.copy(s1)
        
        s1 = np.matmul(self.weight[self.num_layer],s1) + self.bias[self.num_layer]
        self.s_batch[self.num_layer] = np.copy(s1)
        P = np.exp(s1+eps) / np.sum(np.exp(s1+eps),axis=0)
   
        
        return P

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

        P = self.evaluateClassifier(testX)
        acc = self.computeAccuracy(testy, P)
        
        return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 2, help= "batch size")
    parser.add_argument('--epoch', type = int, default = 10,help="number of epoch")
    parser.add_argument('--lmda', default = 0.01, type=float, help = "parameter of L2 regularization")
    parser.add_argument('--outdir', default='./result', type= str, help = " the directory to save the output plot")
    parser.add_argument('--Initialize', default= "He", type= str, help = " Choose the weight initialization: He, Xavier Initialization or random. (default: He)")
    parser.add_argument('--num_layer', default= 3, type= int, help = " Number of Layer (default: 3)")
    parser.add_argument('--num_nodes', default= 50, type= int, nargs='+', help = " Number of node of each layer.")
    parser.add_argument('--min_lr', default = 0.0001, type=float, help = "the minimum learning rate")
    parser.add_argument('--max_lr', default = 0.01, type=float, help = "the maximum learning rate")
    parser.add_argument('--n_s', default = 2, type=int, help = " Hyper-Parameter of cycle during training.")
    parser.add_argument('--Dropout', default= True, type=lambda x: (str(x).lower() == 'true'), help = "Apply dropout or not. (Default : True)")
    parser.add_argument('--Adam', default= False,  type=lambda x: (str(x).lower() == 'true'), help = "Apply Adam optimizer or not. (Default : False)")

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    train_filename = ["../Dataset/cifar/data_batch_1","../Dataset/cifar/data_batch_3","../Dataset/cifar/data_batch_4","../Dataset/cifar/data_batch_5"]
    # train_filename = "../Dataset/cifar/data_batch_1"
    val_filename = "../Dataset/cifar/data_batch_2"
    test_filename = "../Dataset/cifar/test_batch"

    filenames = {}
    filenames["train"] = train_filename
    filenames["val"] = val_filename
    print("args.Adam:",args.Adam)
    if len(args.num_nodes) != args.num_layer - 1:
        raise ValueError("number of nodes and number of layer does not match.")

    if args.Initialize not in ["Xavier","He","Random"]:
        print(f"{args.Initialize}: no such Initialization method, use Random instead.")
        args.Initialize = "Random"

    trainer = NNmodel(filenames,
                       batch_size = args.batch_size,
                       epoch = args.epoch,
                       lmda = args.lmda,
                       initial = args.Initialize,
                       num_layer = args.num_layer,
                       num_nodes = args.num_nodes,
                       min_lr = args.min_lr,
                       max_lr = args.max_lr,
                       n_s = args.n_s,
                       dropout = args.Dropout,
                       Adam = args.Adam,
                       outdir = args.outdir)


    trainer.train()
    trainer.plotResult()

    ### do inference
     
    testX, testy,testY = load_batch(test_filename)
    TEST_ACC = trainer.Inference(testX,testy,testY)
    print("test accuracy: ",TEST_ACC,"%")
   

    
if __name__ == "__main__":
    main()