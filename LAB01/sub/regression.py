import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sub.minimization as mymin


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
        plt.show()
        return
