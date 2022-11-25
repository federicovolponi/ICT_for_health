# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 19:27:13 2022

@author: d001834
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sub.minimization as mymin

def euclidean_distance(p, q):
  dist = np.sqrt(np.sum(np.square(p - q)))
  return dist

plt.close('all')
Np=200 # number of points in the Gaussian random process
Nm=21 # FIR filter memory 
Nprev=2*Nm
T=10
th=np.arange(-Nprev,Nprev)
h=np.exp(-(th/T)**2) # Gaussian impulse response
h=h/np.linalg.norm(h)
plt.figure() # show the impulse response
plt.plot(th,h)
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('h(t)')
plt.title('Impulse response')
#plt.show()
x=np.random.randn(Np,) # input white Gaussian process
y=np.convolve(x,h,mode='same') # output filtered Gaussian process
t=np.arange(len(y))
plt.figure() # show the realization
plt.grid()
plt.plot(t,y)
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.title('Realization of the Gaussian random process')
#plt.show()
#%%
autocorr=np.exp(-(th/T)**2/2)
plt.figure()
plt.plot(th,autocorr)
plt.xlabel(r'$\tau (s)$')
plt.ylabel(r'$R_Y(\tau)$')
plt.title('Autocorrelation function')
plt.grid()
#plt.show()
#%%
M_sampled=10 # number of points used in GP regression
t_sampled=np.random.choice(t,(M_sampled,),replace=False)
y_sampled=y[t_sampled]
# randomly select the new point t_star for which we perform GP regression
t_rem=list(set(t)-set(t_sampled))
t_star=np.random.choice(t_rem,(1,),replace=False)[0]
y_true=y[t_star]
#%%
t_sampled1=t_sampled[:, np.newaxis]
t_new=np.vstack((t_sampled1,t_star))
delta_t_matr=t_new-t_new.T
R=np.exp(-(delta_t_matr/T)**2/2)
plt.matshow(R)
plt.colorbar()
plt.title('Theoretical covariance matrix')
#plt.show()
#%% GP regression
k = R[:-1, -1]
R_N_1 = R[:-1, :-1]
d = R[-1, -1]
mean = k.T @ np.linalg.inv(R_N_1) @ y_sampled
var = d - k.T @ np.linalg.inv(R_N_1) @ y_sampled
std = np.sqrt(var)


A = np.array([t_sampled, np.ones(10)])
print(A)
m = mymin.SolveLLS(y_sampled, A.T)
m.run()
w_hat = m.sol
y_hat = t_star * w_hat[0] + w_hat[1]
plt.figure()
plt.plot(t, y)
plt.plot(t_sampled, y_sampled, 'o')
plt.plot(t_star, mean, 'x')
plt.plot([t_star, t_star], [mean - std, mean + std])
plt.plot(t_star, y_hat, 'o')
plt.plot(t_star, y_true, 'X')
plt.plot(t, t * w_hat[0] + w_hat[1])
#plt.show()

################### Data preparation #########################################
x = pd.read_csv("C:\Coding\ICT_for_health\LAB01\parkinsons_updrs.csv")  # read the dataset
subj = pd.unique(x['subject#']) #unique values of patient ID

X = pd.DataFrame()
for k in subj:
    xk = x[x['subject#'] == k]  #data of user k
    xk1 = xk.copy()
    xk1.test_time = xk1.test_time.astype(int)   #remove decimal values to consider just the day and not the hour
    xk1['g'] = xk1['test_time'] # add a new feature g with test_time values
    v = xk1.groupby('g').mean() # group by the g features(test_time) to have averaged values per day 
    X = pd.concat([X,v], axis=0, ignore_index=True) #Concatenate the k-patients by row ignoring index

features = list(x.columns)
Np, Nc = X.shape
Ntr = int(Np*0.5)   # Dimension training set
Nval = Nte = int(Np*0.25)    # Dimensions test and validation sets

Xsh = X.sample(frac=1, replace=False, random_state=309709, axis=0, ignore_index=True)   # Shuffle of the dataframe
# Generate training and test matrices
X_tr = Xsh[0:Ntr]   #dataframe of the training data
mm = X_tr.mean()
ss = X_tr.std()
my = mm['total_UPDRS']  #mean of total_UPDRS
sy = ss['total_UPDRS']  #st. dev. of total UPDRS

# Generate the normalized training, test and validation datasets, remove unwanted regressors
Xsh_norm=(Xsh-mm)/ss  #normalized data
ysh_norm=Xsh_norm['total_UPDRS']  #regressand only
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#', 'Jitter:DDP', 'Shimmer:DDA', 'sex', 'test_time', 'Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','RPDE','DFA'],axis=1) #regressors only, removing unwanted features

X_tr = Xsh_norm[0:Ntr].values
X_te = Xsh_norm[Ntr:Np-Nval].values
X_val = Xsh_norm[Np-Nval:].values
y_tr =  ysh_norm[0:Ntr].values
y_te = ysh_norm[Ntr:Np-Nval].values
y_val = ysh_norm[Np-Nval:].values

################# GP Regression ###################################
N = 10
teta = 1
sigma_v_2_list = np.linspace(1, 10, 10)
r_2_list = np.linspace(0.00001, 0.1, 10)
y_hat_GP = np.zeros(Nval)
MSE_min = 1
# Find the neighbors of sample x from training set
for sigma_v_2 in sigma_v_2_list:
    for r_2 in r_2_list:

        for k in range(Nval):
            x = X_val[k, :]
            y = y_val[k]
            dist_tr = []
            for i in range(Ntr):
                dist_tr.append(euclidean_distance(x, X_tr[i, :]))
            
            # sort the indexes in ascending order
            neighbors_index_tr = np.argsort(dist_tr)

            neighbors_Xtr_tr = np.zeros([N, X_tr.shape[1]])
            neighbors_y_tr = np.zeros([N - 1, 1])
            for i in range(N - 1):
                # take the N nearer neighbors to the sample
                neighbors_Xtr_tr[i] = X_tr[neighbors_index_tr[i]]
                neighbors_y_tr[i] = y_tr[neighbors_index_tr[i]]
            
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

            y_hat_GP[k] = mu

        Err = y_hat_GP - y_val
        MSE = np.mean(Err**2)
        if MSE < MSE_min:
            MSE_min = MSE
            r_2_min = r_2
            sigma_v_2_min = sigma_v_2
print(f"\nMSE for r_2 = {r_2_min} and sigma_v_2 = {sigma_v_2_min}:  {MSE_min}")

# Plot regression line
plt.figure(figsize=(6,4))
plt.plot(y_hat_GP, y_val,'.')    # plot the points of the real regressand and of the estimation
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
plt.legend()
plt.xlabel(r'$y$')
plt.axis('square')
plt.ylabel(r'$\^y$')
plt.grid()
plt.tight_layout()
#plt.savefig(f'charts\{title}')
plt.show()

#%% linear regression

#%% Final plot
