import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import sub.minimization as mymin


class regression:
    def __init__(self, X, y ):
        self.X = X
        self.y = y
        return
""" 
    def LLS(self):
        m = mymin.SolveLLS(self.y, self.X)
        m.run()
        w_hat = m.sol
        regressors = list(self.X.columns)
        Nf = len(w_hat)
        nn = np.arange(Nf)
        plt.figure(figsize=(6,4))
        plt.plot(nn,w_hat,'-o')
        ticks=nn
        plt.xticks(ticks, regressors, rotation=90)
        plt.ylabel(r'$\^w(n)$')
        plt.title('LLS-Optimized weights')
        plt.grid()
        plt.tight_layout()
        plt.savefig('C:\Coding\ICT_for_health\LAB01\charts\LLS-what.png')
        plt.show()
 """
