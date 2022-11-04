
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin

def euclidean_distance(p, q):
  dist = np.sqrt(np.sum(np.square(p - q)))
  return dist

class regression:
    def __init__(self, X, y, Ntr):
        self.Np = y.shape[0] 
        self.Ntr = Ntr
        self.Nte = self.Np - self.Ntr
        self.Nf = X.shape[1]
        self.X = X
        self.y = y
        self.X_tr=X[0:Ntr]
        self.X_te=X[Ntr:]
        self.y_tr=y[0:Ntr]
        self.y_te=y[Ntr:]
        self.w_hat = np.zeros((self.Nf, ), dtype=float)
        return

    def LLS(self, title = "LLS-what.png"):
        m = mymin.SolveLLS(self.y_tr, self.X_tr)
        m.run()
        self.w_hat = m.sol
        self.y_hat_te = self.X_te @ self.w_hat
        self.y_hat_tr = self.X_tr @ self.w_hat
        regressors = list(self.X_tr.columns)
        nn = np.arange(self.Nf)
        plt.figure(figsize=(6,4))
        plt.plot(nn,self.w_hat,'-o')
        ticks=nn
        plt.xticks(ticks, regressors, rotation=90)
        plt.ylabel(r'$\^w(n)$')
        plt.title('LLS-Optimized weights')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\{title}')
        #plt.show()
        return
    
    def steepestDescent(self, title ="SteepestDescent.png"):
        y_tr = self.y_tr.values
        X_tr = self.X_tr.values
        m = mymin.steepestDescentAlgorithm(y_tr, X_tr)
        m.run()
        self.w_hat = m.sol
        self.y_hat_te = self.X_te @ self.w_hat
        self.y_hat_tr = self.X_tr @ self.w_hat
        regressors = list(self.X_tr.columns)
        nn = np.arange(self.Nf)
        plt.figure(figsize=(6,4))
        plt.plot(nn,self.w_hat,'-o')
        ticks=nn
        plt.xticks(ticks, regressors, rotation=90)
        plt.ylabel(r'$\^w(n)$')
        plt.title('SteepestDescent-Optimized weights')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\{title}')
        #plt.show()
        return
    
    def plotHistrogram(self, title = "LLS-hist.png"):
        y_tr = self.y_tr.values
        y_hat_tr = self.y_hat_tr.values
        y_te = self.y_te.values
        y_hat_te = self.y_hat_te.values
        E_tr= y_tr.reshape(len(y_tr),1) - y_hat_tr.reshape(len(y_tr),1)# training
        E_te= y_te.reshape(len(y_tr),1) - y_hat_te.reshape(len(y_tr),1) # test
        e=[E_tr.reshape(len(E_tr), ),E_te.reshape(len(E_tr), )]
        plt.figure(figsize=(6,4))
        plt.hist(e,bins=50,density=True, histtype='bar',label=['training','test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title('LLS-Error histograms using all the training dataset')
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\{title}')
        #plt.show()
        return

    def plotRegressionLine(self, title = "LLS-yhat_vs_y.png"):
        plt.figure(figsize=(6,4))
        plt.plot(self.y_te,self.y_hat_te,'.', label = "all")
        plt.legend()
        v=plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
        plt.xlabel(r'$y$')
        plt.axis('square')
        plt.ylabel(r'$\^y$')
        plt.grid()
        plt.title('LLS-test')
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\{title}')
        plt.show()

    def errorsAndCoefficients(self):
        y_tr = self.y_tr.values
        y_hat_tr = self.y_hat_tr.values
        y_te = self.y_te.values
        y_hat_te = self.y_hat_te.values
        E_tr= y_tr.reshape(len(y_tr),1) - y_hat_tr.reshape(len(y_tr),1)# training
        E_te= y_te.reshape(len(y_tr),1) - y_hat_te.reshape(len(y_tr),1) # test
        E_tr_max=E_tr.max()
        E_tr_min=E_tr.min()
        E_tr_mu=E_tr.mean()
        E_tr_sig=E_tr.std()
        E_tr_MSE=np.mean(E_tr**2)
        R2_tr=1-E_tr_MSE/(np.std(y_tr)**2)
        c_tr=np.mean((y_tr.reshape(len(y_tr),1)-y_tr.reshape(len(y_tr),1).mean())*(y_hat_tr.reshape(len(y_tr),1)-y_hat_tr.reshape(len(y_tr),1).mean()))/(y_tr.reshape(len(y_tr),1).std()*y_hat_tr.reshape(len(y_tr),1).std())
        E_te_max=E_te.max()
        E_te_min=E_te.min()
        E_te_mu=E_te.mean()
        E_te_sig=E_te.std()
        E_te_MSE=np.mean(E_te**2)
        R2_te=1-E_te_MSE/(np.std(y_te)**2)
        c_te=np.mean((y_te.reshape(len(y_tr),1)- y_te.reshape(len(y_tr),1).mean())*(y_hat_te.reshape(len(y_tr),1)-y_hat_te.reshape(len(y_tr),1).mean()))/(y_te.reshape(len(y_tr),1).std()*y_hat_te.reshape(len(y_tr),1).std())

        cols=['min','max','mean','std','MSE','R^2','corr_coeff']
        rows=['Training','test']
        p=np.array([
            [E_tr_min,E_tr_max,E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr,c_tr],
            [E_te_min,E_te_max,E_te_mu,E_te_sig,E_te_MSE,R2_te,c_te],
                    ])

        results=pd.DataFrame(p,columns=cols,index=rows)
        print(results, "\n\n")

    def localRegression(self, N):
        dist = []
        neighbors_index = []
        neighbors_Xtr = []
        neighbors_Xte = []
        neighbors_ytr = []
        neighbors_yte = []
        
        for iter in range(self.Nte):
            for i in range(self.Ntr):
                dist.append(euclidean_distance(self.X_te.iloc[iter], self.X_tr.iloc[i]))

            neighbors_index = np.argsort(dist)
            for i in range(N):
                newrow_Xtr = self.X_tr.values[neighbors_index[i]]
                newrow_Xte = self.X_te.values[neighbors_index[i]]
                newrow_ytr = self.y_tr.values[neighbors_index[i]]
                newrow_yte = self.y_te.values[neighbors_index[i]]
                neighbors_ytr.append(newrow_ytr)
                neighbors_yte.append(newrow_yte)
                neighbors_Xtr.append(newrow_Xtr)
                neighbors_Xte.append(newrow_Xte)

            self.X_tr = pd.DataFrame(neighbors_Xtr)
            #self.X_te = pd.DataFrame(neighbors_Xte)
            self.y_tr = pd.DataFrame(neighbors_ytr)
            #self.y_te = pd.DataFrame(neighbors_yte)
            
            y_tr = self.y_tr.values
            X_tr = self.X_tr.values
            m = mymin.steepestDescentAlgorithm(y_tr, X_tr)
            m.run()
            self.w_hat = m.sol
            self.y_hat_te = self.X_te.iloc[iter] @ self.w_hat
            #self.y_hat_tr = self.X_tr @ self.w_hat