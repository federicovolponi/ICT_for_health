import sub.tools as tools
import numpy as np

class GPRegression():
    def __init__(self, X, y,Np, Ntr, Nval):
        self.X_tr = X[0:Ntr].values
        self.X_te = X[Ntr:Np-Nval].values
        self.X_val = X[Np-Nval:].values
        self.y_tr =  y[0:Ntr].values
        self.y_te = y[Ntr:Np-Nval].values
        self.y_val = y[Np-Nval:].values
        self.Nval = Nval
        self.Ntr = Ntr
        self.Nte = self.X_te.shape[0]
        self.y_hat_GP = np.zeros(Nval)
        return
    
    def run(self, r_2 = 3, sigma_v_2 = 0.001, teta = 1, N = 10):
        for k in range(self.Nval):
            x = self.X_val[k, :]
            y = self.y_val[k]
            dist_tr = []
            for i in range(self.Ntr):
                dist_tr.append(tools.euclidean_distance(x, self.X_tr[i, :]))
            
            # sort the indexes in ascending order
            neighbors_index_tr = np.argsort(dist_tr)

            neighbors_Xtr_tr = np.zeros([N, self.X_tr.shape[1]])
            neighbors_y_tr = np.zeros([N - 1, 1])
            for i in range(N - 1):
                # take the N nearer neighbors to the sample
                neighbors_Xtr_tr[i] = self.X_tr[neighbors_index_tr[i]]
                neighbors_y_tr[i] = self.y_tr[neighbors_index_tr[i]]
            
            neighbors_Xtr_tr[N - 1] = x
            # Create covarinace matrix
            R = np.zeros([N, N])
            for i in range(N):
                for j in range(N):
                    R[i, j] = teta*np.exp(-np.linalg.norm(neighbors_Xtr_tr[i] - neighbors_Xtr_tr[j])/2*r_2) + sigma_v_2

            k_GP = R[:-1, -1]
            R_N_1 = R[:-1, :-1]
            d = R[-1, -1]
            mu = k_GP.T @ np.linalg.inv(R_N_1) @ neighbors_y_tr
            var = d - k_GP.T @ np.linalg.inv(R_N_1) @ neighbors_y_tr
            std = np.sqrt(var)

            self.y_hat_GP[k] = mu
        return self.y_hat_GP