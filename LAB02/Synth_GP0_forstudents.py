# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 19:27:13 2022

@author: d001834
"""

import numpy as np
import matplotlib.pyplot as plt
import sub.minimization as mymin
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
plt.show()
x=np.random.randn(Np,) # input white Gaussian process
y=np.convolve(x,h,mode='same') # output filtered Gaussian process
t=np.arange(len(y))
plt.figure() # show the realization
plt.grid()
plt.plot(t,y)
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.title('Realization of the Gaussian random process')
plt.show()
#%%
autocorr=np.exp(-(th/T)**2/2)
plt.figure()
plt.plot(th,autocorr)
plt.xlabel(r'$\tau (s)$')
plt.ylabel(r'$R_Y(\tau)$')
plt.title('Autocorrelation function')
plt.grid()
plt.show()
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
plt.show()
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
plt.show()
#%% linear regression

#%% Final plot
