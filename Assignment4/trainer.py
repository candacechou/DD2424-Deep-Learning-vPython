import numpy as np
import io 
import argparse
import matplotlib.pyplot as plt
eps = 1e-8

def loadDataset(filenames):

    with io.open(filenames, 'r', encoding='utf8') as f:
        book_data = f.read()
        ## unique
        book_chars = list(set(book_data))
        ## ind2char and char2ind
        ind2char = {}
        char2ind = {}
        
        for idx, chars in enumerate(book_chars):
            ind2char[idx] = chars
            char2ind[chars] = idx

        ### do one-Hot encoding
    book_data = np.array(list(book_data))[:,np.newaxis]
    
    

    return book_data, book_chars




class VanillaRNN():

    def __init__(self,
                book_data,
                book_char,
                epoch = 1,
                sig = 0.1,
                eta = 0.1,
                m = 100,
                seq_len = 25,
                optimizer = "adagrad"):

        ### training data
        self.book_data = book_data
        self.book_char = book_char

        
        self.k = len(self.book_char)

        self.sig  = sig
        self.eta = eta
        self.m = m
        self.seq_len = seq_len

        ### initialization
        self.U = np.random.normal(0,sig,(self.m,self.k))
        self.W = np.identity(self.m) * self.sig
        self.V = np.random.normal(0,sig,(self.k,self.m))

        self.b = np.zeros((self.m,1))
        self.c = np.zeros((self.k,1))

        self.grad_U = self.grad_W = self.grad_V = self.grad_b = self.grad_c = None
        self.m_U = np.zeros_like(self.U)
        self.m_W = np.zeros_like(self.W)
        self.m_V = np.zeros_like(self.V)
        self.m_b = np.zeros_like(self.b)
        self.m_c = np.zeros_like(self.c)
        self.__resetGrad()
        ### optimizer
        self.optimizer = optimizer
        self.epoch = epoch
        
        ### save best
        self.losses = []
        self.min_loss = np.inf
        self.min_loss_text = None



    def __onehotEncoding(self,data):
    
        nt = data.shape[0]
        kt = len(self.book_char)
        Y = np.zeros((kt,nt))
        for i in range(kt):
            idx = (data == self.book_char[i])
            Y[i,idx[:,0]] = 1
        return Y
    
    def forwardPass(self,h,x):
        
        a_t = self.W@h + self.U@x +self.b
        h_t = np.tanh(a_t)
        o_t = self.V@h_t + self.c
        p_t = np.exp(o_t) / np.sum(np.exp(o_t + eps),axis=0)
        
        return a_t,h_t,o_t,p_t
   
    def evaluateClassifier(self,h,x,y):
        """
        Input:
        h : hidden state sequance
        x: sequence of input vectors
        """
        O = np.zeros((self.k,x.shape[1]))
        P = np.zeros((self.k,x.shape[1]))
        H = np.zeros((self.m,x.shape[1]+1))
        A = np.zeros((self.m,x.shape[1]))
        H[:,0] = h[:,0]
        for i in range(x.shape[1]):
            
            a_t,h,o_t,p_t = self.forwardPass(h,x[:,i])
            A[:,i] = a_t[:,0]
            H[:,i+1] = h[:,0]
            O[:,i] = o_t[:,0]
            P[:,i] = p_t[:,0]
        
        return A, P, O, H

    def __resetGrad(self):

        self.grad_U = np.zeros((self.m,self.k))
        self.grad_W = np.zeros((self.m,self.m))
        self.grad_V = np.zeros((self.k,self.m))
        self.grad_b = np.zeros((self.m,1))
        self.grad_c = np.zeros((self.k,1))

    def computeGradient(self,X,Y,H,P,A):
        n = X.shape[1]
        self.__resetGrad()
        grad_a = np.zeros((A.shape[0],1))
        G = -(Y - P).T
        self.grad_c = np.sum(G,axis=0,keepdims=True).T
        self.grad_V = np.matmul(G.T, H[:,1:].T)
        for i in reversed(range(n-1)):
            x_t = X[:, i].reshape(X.shape[0],1)
            y_t = Y[:, i].reshape(Y.shape[0],1)
            p_t = P[:, i].reshape(P.shape[0],1)
            h_t = H[:, i+1].reshape(H.shape[0],1)
            ph_t= H[:, i].reshape(H.shape[0],1)
            g = -(y_t - p_t).T
            
             
            
            
            grad_h = np.dot((self.V).T, g.T) + np.dot(self.W.T, grad_a)
            grad_a = grad_h * (1 - h_t**2)

            g = grad_a
            self.grad_b  += g
            
            self.grad_W  += np.matmul(g,ph_t.T)
            self.grad_U  += np.matmul(g, x_t.T)
        
            


        ### Clip
        self.grad_b[self.grad_b < -5] = -5
        self.grad_b[self.grad_b > 5] = 5

        self.grad_c[self.grad_c < -5] = -5
        self.grad_c[self.grad_c > 5] = 5

        self.grad_W[self.grad_W < -5] = -5
        self.grad_W[self.grad_W > 5] = 5

        self.grad_U[self.grad_U < -5] = -5
        self.grad_U[self.grad_U > 5] = 5

        self.grad_V[self.grad_V < -5] = -5
        self.grad_V[self.grad_V > 5] = 5

        
    
    def __adagrad(self,grads,ms,param):
        
        ms += grads ** 2
        param -= ((self.eta / np.sqrt(ms + eps))) * grads
        
        return ms, param

    def __rmsprop(self,grads,ms,param):
        
        gamma = 0.9
        ms = gamma * ms + (1 - gamma) * (grads ** 2)
        param += - ((self.eta / np.sqrt(ms+eps))) * grads

        return ms,param

    def updateWeight(self):

        if self.optimizer == "adagrad":
            self.m_W, self.W = self.__adagrad(self.grad_W,self.m_W,self.W)
            self.m_U, self.U = self.__adagrad(self.grad_U,self.m_U,self.U)
            self.m_V, self.V = self.__adagrad(self.grad_V,self.m_V,self.V)
            self.m_c, self.c = self.__adagrad(self.grad_c,self.m_c,self.c)
            self.m_b, self.b = self.__adagrad(self.grad_b,self.m_b,self.b)
            
        elif self.optimizer == "RMSProp":
            self.m_W, self.W = self.__rmsprop(self.grad_W,self.m_W,self.W)
            self.m_U, self.U = self.__rmsprop(self.grad_U,self.m_U,self.U)
            self.m_V, self.V = self.__rmsprop(self.grad_V,self.m_V,self.V)
            self.m_c, self.c = self.__rmsprop(self.grad_c,self.m_c,self.c)
            self.m_b, self.b = self.__rmsprop(self.grad_b,self.m_b,self.b)

    def computeLoss(self,Y,P):
        
        loss = 0
        
        for i in range(Y.shape[1]):
            loss += -np.log(np.dot(Y[:, i].T, P[:,i]))

        return loss

    def train(self):

        h = None
        smooth_loss = 0
        iters = 0
        for i_epoch in range(self.epoch):
            self.e = 0
            while(self.e <= len(self.book_data) - self.seq_len - 1):
                e1 = self.e + 1
                X_batch = self.book_data[self.e : self.e + self.seq_len - 1]
                Y_batch = self.book_data[self.e + 1: self.e + self.seq_len]
                x_one_hot = self.__onehotEncoding(X_batch)
                y_one_hot = self.__onehotEncoding(Y_batch)

                if  self.e == 0:
                    hprev = np.zeros((self.m, 1))
                    print(hprev.shape)
                else:
                    hprev = H[:,1]
                    hprev = hprev[:,np.newaxis]
                

                A, P, O, H = self.evaluateClassifier(hprev,x_one_hot,y_one_hot)
                self.computeGradient(x_one_hot, y_one_hot, H, P, A)
                loss = self.computeLoss(y_one_hot,P)
                if smooth_loss == 0:
                    smooth_loss = loss

                smooth_loss = 0.999 * smooth_loss + 0.001 * loss
                self.losses.append(smooth_loss)

                if iters == 0 or iters % 1000 == 0:
                    print("Iteration " + str(iters) + " " + "(epoch " +
                          str(i_epoch) + ")" + ": " + str(smooth_loss))

                    print("----------------------------------")
                    synth = self.synthesizeText(
                        hprev, x_one_hot[:, 0], y_one_hot, 200)
                    decoded_text = self.decodeOneHot(synth)
                    print(decoded_text)
                    print("----------------------------------")

                    if self.min_loss < smooth_loss :
                        self.min_loss = smooth_loss
                        self.min_loss_text = decoded_text
                        

                self.updateWeight()
                iters += 1
                self.e += self.seq_len
                

        print(f"The best losses:{self.min_loss}, and its text is : \n")
        print(self.min_loss_text)

    def plotLoss(self):
        filename = "result/loss.jpg"
        plt.figure()
        plt.plot(range(len(self.losses)),self.losses)    
        plt.title("The loss of each iterations")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(filename)  

    def decodeOneHot(self, text):
        chars = ""
        for i in range(len(text)):
            pos = np.where(text[i] != 0)
            chars += self.book_char[pos[0][0]]

        return chars

    def synthesizeText(self,h,x,y,n):
        """
        """
        synthesized_text = []

        for i in range(n):

            _, _, _, p_i = self.forwardPass(h, x)
            cp = np.cumsum(p_i, axis=0)
            a = np.random.rand()
            ixs = np.nonzero(cp - a > 0)
            ii = ixs[0][0]
            x = np.zeros((self.k, 1))
            x[ii][0] = 1
            synthesized_text.append(x)

        return synthesized_text



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type = int, default = 10,help="number of epoch")
    parser.add_argument('--lr', default = 0.1, type=float, help = "learning rate")
    parser.add_argument('--sig', default = 0.1, type=float, help = "random sig ")
    parser.add_argument('--m', default=100, type= int, help = "dimensionality of its hidden state")
    parser.add_argument('--seq_len', default= 25, type= int, help = "length of the input sequences")
    parser.add_argument('--optimizer', default= "adagrad", type= str, help = "choose optimizer :[adagrad, RMSprop], (default adagrad)")

    args = parser.parse_args()
    filename = "../Dataset/JK-Rowling/goblet_book.txt"
    book_data, book_chars = loadDataset(filename)

    trainer = VanillaRNN(book_data,
                         book_chars,
                         epoch = args.epoch,
                         sig = args.sig,
                         eta = args.lr,
                           m = args.m,
                     seq_len = args.seq_len,
                    optimizer = args.optimizer)

    trainer.train()
    trainer.plotLoss()

         
    return 0

if __name__ == "__main__":
    main()



