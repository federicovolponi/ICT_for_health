import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin

# Function to evaluate the euclidean distance between two vectors
def euclidean_distance(p, q):
  dist = np.sqrt(np.sum(np.square(p - q)))
  return dist

class LinearRegression:
    """
    # Linear regression
    Class to perform linear regression:

    ========================================================================================
    
    ### Methods:
    
    - LLS(): finds y_hat for the test and training set using the LLS method

    - SteepestDescent(): finds y_hat for the test and training set using the Steepest descent algorithm
    
    - localRegression(N): finds y_hat for the test and training set implementing local regression on the N values nearer to a test sample
    
    - plot_LLS_vs_SD(): plots the two weight vectors found by the LLS method and the SD algorithm
    
    - plotHistrogram(algorithm): plot the histogram of the errors between y_hat and y, for both training and test set, for a given algorithm
    
    - plotRegressionLine(algorithm): plot the regression line for a given algorithm
    
    - denormalize(): simple method to denormalize if necessary
    
    - errorsAndCoefficients(): analysis of the errors and coefficients for both the training and test set
    """
    def __init__(self, X, y, Ntr, sy, my):
        self.Np = y.shape[0] # number of rows of the dataframe (number of patients)
        self.Ntr = Ntr  # number of rows for the training set
        self.Nte = self.Np - self.Ntr   # number of rows for the test set
        self.Nf = X.shape[1]    # number of features
        self.X = X  # dataframe
        self.y = y  # regressand
        self.X_tr=X[0:Ntr]  # training set
        self.X_te=X[Ntr:]   #test set
        self.y_tr=y[0:Ntr].values   # regressand training set, from Pandas Dataframe to numpy array
        self.y_te=y[Ntr:].values    # regressand test set, from Pandas Dataframe to numpy array
        self.w_hat_LLS = np.zeros((self.Nf, ), dtype=float)
        self.w_hat_SD = np.zeros((self.Nf, ), dtype=float)
        self.sy = sy    # standard deviation of the regressand
        self.my = my    # mean of the regressand
        return
    
    def LLS(self):
        m = mymin.SolveLLS(self.y_tr, self.X_tr)    # call to minimization class to evaluate w_hat
        m.run()
        self.w_hat_LLS = m.sol
        # Estimation of the regressand for both the training and test set
        self.y_hat_te_LLS = self.X_te.values @ self.w_hat_LLS
        self.y_hat_tr_LLS = self.X_tr.values @ self.w_hat_LLS
        return

    def steepestDescent(self, Nit = 100, eps = 1e-6):   # set number of iteration and eps for stopping condition
        y_tr = self.y_tr
        X_tr = self.X_tr.values
        m = mymin.steepestDescentAlgorithm(y_tr, X_tr) # call to minimization class to evaluate w_hat
        m.run(Nit, eps)
        self.w_hat_SD = m.sol
        # Estimation of the regressand for both the training and test set
        self.y_hat_te_SD = self.X_te.values @ self.w_hat_SD
        self.y_hat_tr_SD = self.X_tr.values @ self.w_hat_SD
        return

    def localRegression(self, N):   # local regression on the N neighbors
        y_hat_te = np.zeros([self.Nte, 1])
        y_hat_tr = np.zeros([self.Ntr, 1])
        X_te = self.X_te.values # test set from Pandas Dataframe to numpy array
        X_tr = self.X_tr.values # training set from Pandas Dataframe to numpy array
        
        # ASSUMPTION: Nte = Ntr
        for iter in range(self.Nte):    # loop on the Nte/Ntr patients 
            dist_te = []
            dist_tr = []
            for i in range(self.Ntr):   # loop to evaluate the euclidean distance between the iter-sample and the points in the training set
                dist_te.append(euclidean_distance(X_te[iter, :], X_tr[i, :]))
                dist_tr.append(euclidean_distance(X_tr[iter, :], X_tr[i, :]))
            # sort the indexes in ascending order
            neighbors_index_te = np.argsort(dist_te)
            neighbors_index_tr = np.argsort(dist_tr)

            neighbors_Xtr_te = np.zeros([N,self.Nf]) 
            neighbors_ytr_te = np.zeros([N, 1])
            neighbors_Xtr_tr = np.zeros([N,self.Nf])
            neighbors_ytr_tr = np.zeros([N, 1])
            for i in range(N):
                # take the N nearer neighbors to the sample
                neighbors_Xtr_te[i] = X_tr[neighbors_index_te[i]]
                neighbors_ytr_te[i] = self.y_tr[neighbors_index_te[i]]
                neighbors_Xtr_tr[i] = X_tr[neighbors_index_tr[i]]
                neighbors_ytr_tr[i] = self.y_tr[neighbors_index_tr[i]]

            # evaluate the y_hat using the steepest descent method 
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

        # save the final y_hat's
        self.y_hat_te_LR = y_hat_te
        self.y_hat_tr_LR = y_hat_tr    

    def plot_LLS_vs_SD(self):
        regressors = list(self.X_tr.columns)    # list of regressors
        nn = np.arange(self.Nf)
        plt.figure(figsize=(6,4))
        plt.plot(nn,self.w_hat_LLS,'-o', label ="LLS")  # plot of the weight vector estimate using LLS
        plt.plot(nn,self.w_hat_SD,'-o', label = "Steepest Descent") # plot of the weight vector estimate using SD
        plt.legend()
        ticks=nn
        plt.xticks(ticks, regressors, rotation=90)
        plt.ylabel(r'$\^w(n)$')
        plt.title('LLS-Optimized weights vs SD Optimized weights')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\LLSvsSD_w_hat_weights')
        plt.show()

    def plotHistrogram(self, title = "LLS-Error", algorithm = "LLS"):
        y_tr = self.y_tr
        y_te = self.y_te
        
        # select the algorithm for which plot the histogram
        if algorithm == "LLS":
            y_hat_tr = self.y_hat_tr_LLS
            y_hat_te = self.y_hat_te_LLS
        elif algorithm == "SD":
            y_hat_tr = self.y_hat_tr_SD
            y_hat_te = self.y_hat_te_SD
        elif algorithm == "LR":
            y_hat_tr = self.y_hat_tr_LR
            y_hat_te = self.y_hat_te_LR

        E_tr= y_tr.reshape(len(y_tr),1) - y_hat_tr.reshape(len(y_tr),1) # errors of the training set
        E_te= y_te.reshape(len(y_tr),1) - y_hat_te.reshape(len(y_tr),1) # errors of the test set
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
        # select the algorithm for which plot the histogram
        if algorithm == "LLS":
            y_hat_te = self.y_hat_te_LLS
        elif algorithm == "SD":
            y_hat_te = self.y_hat_te_SD
        elif algorithm == "LR":
            y_hat_te = self.y_hat_te_LR

        plt.figure(figsize=(6,4))
        plt.plot(self.y_te, y_hat_te,'.', label = algorithm)    # plot the points of the real regressand and of the estimation
        v=plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
        plt.legend()
        plt.xlabel(r'$y$')
        plt.axis('square')
        plt.ylabel(r'$\^y$')
        plt.grid()
        plt.title(f'Regressione line for {algorithm}')
        plt.tight_layout()
        plt.savefig(f'C:\Coding\ICT_for_health\LAB01\charts\{title}')
        plt.show()
    
    def denormalize(self):
        # denormalize for LLS
        self.y_hat_te_LLS = self.y_hat_te_LLS * self.sy + self.my
        self.y_hat_tr_LLS = self.y_hat_tr_LLS * self.sy + self.my
        # denormalize for SD
        self.y_hat_te_SD = self.y_hat_te_SD * self.sy + self.my
        self.y_hat_tr_SD = self.y_hat_tr_SD * self.sy + self.my
        # denormalize for LR
        self.y_hat_te_LR = self.y_hat_te_LR * self.sy + self.my
        self.y_hat_tr_LR = self.y_hat_tr_LR * self.sy + self.my
        # denormalize true regressands
        self.y_te = self.y_te * self.sy + self.my
        self.y_tr = self.y_tr * self.sy + self.my

    def errorsAndCoefficients(self, algorithm = "LLS", toPrint = False):
        y_tr = self.y_tr
        y_te = self.y_te
        
        # select the algorithm for which plot the histogram
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
        
        self.E_tr= y_tr.reshape(len(y_tr),1) - y_hat_tr.reshape(len(y_tr),1)    # errors of the training set
        self.E_te= y_te.reshape(len(y_tr),1) - y_hat_te.reshape(len(y_tr),1 )   # errors of the test set
        # Errors and coefficients for training set
        self.E_tr_max=self.E_tr.max()   # maximum error 
        self.E_tr_min=self.E_tr.min()   # minimum error
        self.E_tr_mu=self.E_tr.mean()   # mean error
        self.E_tr_sig=self.E_tr.std()   # standard deviation of the error
        self.E_tr_MSE=np.mean(self.E_tr**2) # mean square error
        self.R2_tr=1-self.E_tr_MSE/(np.std(y_tr)**2)    # coefficient of determination
        self.c_tr=np.mean((y_tr.reshape(len(y_tr),1)-y_tr.reshape(len(y_tr),1).mean())*(y_hat_tr.reshape(len(y_tr),1)-y_hat_tr.reshape(len(y_tr),1).mean()))\
                            /(y_tr.reshape(len(y_tr),1).std()*y_hat_tr.reshape(len(y_tr),1).std())  # correlation coefficient
        # Errors and coefficients for test set
        self.E_te_max=self.E_te.max()   # maximum error
        self.E_te_min=self.E_te.min()   # minimum error
        self.E_te_mu=self.E_te.mean()   # mean error
        self.E_te_sig=self.E_te.std()   # standard deviation of the error
        self.E_te_MSE=np.mean(self.E_te**2) # mean square error
        self.R2_te=1-self.E_te_MSE/(np.std(y_te)**2)    # coefficient of determination
        self.c_te=np.mean((y_te.reshape(len(y_tr),1)- y_te.reshape(len(y_tr),1).mean())*(y_hat_te.reshape(len(y_tr),1)-y_hat_te.reshape(len(y_tr),1).mean()))\
                        /(y_te.reshape(len(y_tr),1).std()*y_hat_te.reshape(len(y_tr),1).std())   # correlation coefficient
        
        cols=['min','max','mean','std','MSE','R^2','corr_coeff']
        rows=['Training','test']
        p=np.array([    # definition of matrix with all the errors
            [self.E_tr_min,self.E_tr_max,self.E_tr_mu,self.E_tr_sig,self.E_tr_MSE,self.R2_tr,self.c_tr],
            [self.E_te_min,self.E_te_max,self.E_te_mu,self.E_te_sig,self.E_te_MSE,self.R2_te,self.c_te],
                    ])

        results=pd.DataFrame(p,columns=cols,index=rows) # matrix to dataframe 
        if toPrint:
            print(results, "\n\n")
        return results

    