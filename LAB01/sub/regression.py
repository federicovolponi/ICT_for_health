
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin

def euclidean_distance(p, q):
  dist = np.sqrt(np.sum(np.square(p - q)))
  return dist

class regression:
    def __init__(self, X, y, Ntr, sy, my):
        self.Np = y.shape[0] 
        self.Ntr = Ntr
        self.Nte = self.Np - self.Ntr
        self.Nf = X.shape[1]
        self.X = X
        self.y = y
        self.X_tr=X[0:Ntr]
        self.X_te=X[Ntr:]
        self.y_tr=y[0:Ntr].values
        self.y_te=y[Ntr:].values
        self.w_hat_LLS = np.zeros((self.Nf, ), dtype=float)
        self.w_hat_SD = np.zeros((self.Nf, ), dtype=float)
        self.sy = sy
        self.my = my
        return

    def LLS(self):
        m = mymin.SolveLLS(self.y_tr, self.X_tr)
        m.run()
        self.w_hat_LLS = m.sol
        self.y_hat_te_LLS = self.X_te.values @ self.w_hat_LLS
        self.y_hat_tr_LLS = self.X_tr.values @ self.w_hat_LLS
        return

    def plot_LLS_vs_SD(self):
        regressors = list(self.X_tr.columns)
        nn = np.arange(self.Nf)
        plt.figure(figsize=(6,4))
        plt.plot(nn,self.w_hat_LLS,'-o', label ="LLS")
        plt.plot(nn,self.w_hat_SD,'-o', label = "Steepest Descent")
        plt.legend()
        ticks=nn
        plt.xticks(ticks, regressors, rotation=90)
        plt.ylabel(r'$\^w(n)$')
        plt.title('LLS-Optimized weights vs SD Optimized weights')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\LLSvsSD_w_hat_weights')
        #plt.show()

    def steepestDescent(self):
        y_tr = self.y_tr
        X_tr = self.X_tr.values
        m = mymin.steepestDescentAlgorithm(y_tr, X_tr)
        m.run(Nit = 20, eps=1e-6)
        self.w_hat_SD = m.sol
        self.y_hat_te_SD = self.X_te.values @ self.w_hat_SD
        self.y_hat_tr_SD = self.X_tr.values @ self.w_hat_SD
        return
    
    def plotHistrogram(self, title = "LLS-Error", algorithm = "LLS"):
        y_tr = self.y_tr
        y_te = self.y_te
        if algorithm == "LLS":
            y_hat_tr = self.y_hat_tr_LLS
            y_hat_te = self.y_hat_te_LLS
        elif algorithm == "SD":
            y_hat_tr = self.y_hat_tr_SD
            y_hat_te = self.y_hat_te_SD
        elif algorithm == "LR":
            y_hat_tr = self.y_hat_tr_LR
            y_hat_te = self.y_hat_te_LR
        E_tr= y_tr.reshape(len(y_tr),1) - y_hat_tr.reshape(len(y_tr),1)# training
        E_te= y_te.reshape(len(y_tr),1) - y_hat_te.reshape(len(y_tr),1) # test
        e=[E_tr.reshape(len(E_tr), ),E_te.reshape(len(E_tr), )]
        plt.figure(figsize=(6,4))
        plt.hist(e,bins=50,density=True, histtype='bar',label=['training','test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title(f'{title} histograms using all the training dataset')
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\{title}.png')
        plt.show()
        return

    def plotRegressionLine(self, title = "yhat_vs_y.png", algorithm = "LLS"):
        if algorithm == "LLS":
            y_hat_te = self.y_hat_te_LLS
        elif algorithm == "SD":
            y_hat_te = self.y_hat_te_SD
        elif algorithm == "LR":
            y_hat_te = self.y_hat_te_LR
        plt.figure(figsize=(6,4))
        plt.plot(self.y_te, y_hat_te,'.', label = algorithm)
        v=plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
        plt.legend()
        plt.xlabel(r'$y$')
        plt.axis('square')
        plt.ylabel(r'$\^y$')
        plt.grid()
        plt.title(f'Regressione line for {title}')
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\{title}')
        #plt.show()

    def denormalize(self, sy, my, algorithm = "Default"):
        if algorithm == "LLS":
            self.y_hat_te_LLS = self.y_hat_te_LLS * sy + my
            self.y_hat_tr_LLS = self.y_hat_tr_LLS * sy + my
        elif algorithm == "SD":
            self.y_hat_te_SD = self.y_hat_te_SD * sy + my
            self.y_hat_tr_SD = self.y_hat_tr_SD * sy + my
        elif algorithm == "LR":
            self.y_hat_te_LR = self.y_hat_te_LR * sy + my
            self.y_hat_tr_LR = self.y_hat_tr_LR * sy + my
        else:
            self.y_te = self.y_te * sy + my
            self.y_tr = self.y_tr * sy + my

    def errorsAndCoefficients(self, algorithm = "LLS", toPrint = False):
        y_tr = self.y_tr
        y_te = self.y_te
        if algorithm == "LLS":
            y_hat_tr = self.y_hat_tr_LLS
            y_hat_te = self.y_hat_te_LLS
            if toPrint:
                print("LLS:\n")
        elif algorithm == "SD":
            y_hat_tr = self.y_hat_tr_SD
            y_hat_te = self.y_hat_te_SD
            if toPrint:
                print("Steepest descent: \n")
        elif algorithm == "LR":
            y_hat_tr = self.y_hat_tr_LR
            y_hat_te = self.y_hat_te_LR
            if toPrint:
                print("Local regression: \n")
        
        self.E_tr= y_tr.reshape(len(y_tr),1) - y_hat_tr.reshape(len(y_tr),1)# training
        self.E_te= y_te.reshape(len(y_tr),1) - y_hat_te.reshape(len(y_tr),1) # test
        self.E_tr_max=self.E_tr.max()
        self.E_tr_min=self.E_tr.min()
        self.E_tr_mu=self.E_tr.mean()
        self.E_tr_sig=self.E_tr.std()
        self.E_tr_MSE=np.mean(self.E_tr**2)
        self.R2_tr=1-self.E_tr_MSE/(np.std(y_tr)**2)
        self.c_tr=np.mean((y_tr.reshape(len(y_tr),1)-y_tr.reshape(len(y_tr),1).mean())*(y_hat_tr.reshape(len(y_tr),1)-y_hat_tr.reshape(len(y_tr),1).mean()))/(y_tr.reshape(len(y_tr),1).std()*y_hat_tr.reshape(len(y_tr),1).std())
        self.E_te_max=self.E_te.max()
        self.E_te_min=self.E_te.min()
        self.E_te_mu=self.E_te.mean()
        self.E_te_sig=self.E_te.std()
        self.E_te_MSE=np.mean(self.E_te**2)
        self.R2_te=1-self.E_te_MSE/(np.std(y_te)**2)
        self.c_te=np.mean((y_te.reshape(len(y_tr),1)- y_te.reshape(len(y_tr),1).mean())*(y_hat_te.reshape(len(y_tr),1)-y_hat_te.reshape(len(y_tr),1).mean()))/(y_te.reshape(len(y_tr),1).std()*y_hat_te.reshape(len(y_tr),1).std())

        if toPrint:
            cols=['min','max','mean','std','MSE','R^2','corr_coeff']
            rows=['Training','test']
            p=np.array([
                [self.E_tr_min,self.E_tr_max,self.E_tr_mu,self.E_tr_sig,self.E_tr_MSE,self.R2_tr,self.c_tr],
                [self.E_te_min,self.E_te_max,self.E_te_mu,self.E_te_sig,self.E_te_MSE,self.R2_te,self.c_te],
                        ])

            results=pd.DataFrame(p,columns=cols,index=rows)
            print(results, "\n\n")

    def localRegression(self, N):
        y_hat_te = np.zeros([self.Nte, 1])
        y_hat_tr = np.zeros([self.Ntr, 1])
        X_te = self.X_te.values
        X_tr = self.X_tr.values
        for iter in range(self.Nte):
            dist_te = []
            dist_tr = []
            for i in range(self.Ntr):
                dist_te.append(euclidean_distance(X_te[iter, :], X_tr[i, :]))
                dist_tr.append(euclidean_distance(X_tr[iter, :], X_tr[i, :]))
            
            neighbors_index_te = np.argsort(dist_te)
            neighbors_index_tr = np.argsort(dist_tr)
            neighbors_Xtr_te = np.zeros([N,self.Nf]) 
            neighbors_ytr_te = np.zeros([N, 1])
            neighbors_Xtr_tr = np.zeros([N,self.Nf])
            neighbors_ytr_tr = np.zeros([N, 1])
            for i in range(N):
                
                neighbors_Xtr_te[i] = X_tr[neighbors_index_te[i]]
                neighbors_ytr_te[i] = self.y_tr[neighbors_index_te[i]]
                 
                neighbors_Xtr_tr[i] = X_tr[neighbors_index_tr[i]]
                neighbors_ytr_tr[i] = self.y_tr[neighbors_index_tr[i]]


            m1 = mymin.steepestDescentAlgorithm(neighbors_ytr_te, neighbors_Xtr_te)
            m1.run()
            w_hat_te = m1.sol
            y_hat = X_te[iter, :] @ w_hat_te
            y_hat_te[iter] = y_hat
              
            m2 = mymin.steepestDescentAlgorithm(neighbors_ytr_tr, neighbors_Xtr_tr)
            m2.run()
            w_hat_tr = m2.sol
            y_hat = X_tr[iter, :] @ w_hat_tr
            y_hat_tr[iter] = y_hat

        self.y_hat_te_LR = y_hat_te
        self.y_hat_tr_LR = y_hat_tr    