import sub.regression as myreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = pd.read_csv("c:\Coding\ICT_for_health\LAB01\parkinsons_updrs.csv")
#Analysis of dataframe
'''
x.describe().T
x.info()
features = list(x.columns)
print(features)
'''
subj = pd.unique(x['subject#']) #unique values of patient ID
print("The number of distinct patients in the dataset is ", len(subj))

X = pd.DataFrame()
for k in subj:
    xk = x[x['subject#'] == k]  #data of user k
    xk1 = xk.copy()
    xk1.test_time = xk1.test_time.astype(int)   #remove decimal values to consider just the day and not the hour
    xk1['g'] = xk1['test_time'] # add a new feature g with test_time values
    v = xk1.groupby('g').mean() # group by the g features(test_time) to have averaged values per day 
    X = pd.concat([X,v], axis=0, ignore_index=True) #Concatenate the k-patients by row ignoring index

features = list(x.columns)
print("The dataset shape after the mean is: ", X.shape)
print("The features of the dataset are ", len(features))
print(features)
print("\n\n\n\n")
Np, Nc = X.shape

# Measure and show the covariance matrix
Xnorm = (X - X.mean())/X.std() #normalize dataset
c = Xnorm.cov() #measure the covariance
plt.figure()
plt.matshow(np.abs(c.values), fignum = 0)
plt.xticks(np.arange(len(features)), features, rotation = 90)
plt.yticks(np.arange(len(features)), features, rotation = 0)
plt.colorbar()
plt.title("Correlation coefficients of the features")
plt.tight_layout()
plt.savefig("./corr_coeff.png")
plt.show()
plt.figure()

c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation = 90)
plt.title("corr. coeff among total UPDRS and the other features")
plt.tight_layout()
plt.show()

#Shuffle the data
Xsh = X.sample(frac=1, replace=False, random_state=309709, axis=0, ignore_index=True)

# Generate training and test matrices
Ntr = int(Np*0.5)
Nte = Np - Ntr
X_tr = Xsh[0:Ntr]   #dataframe of the training data
mm = X_tr.mean()
ss = X_tr.std()
my = mm['total_UPDRS']  #mean of total_UPDRS
sy = ss['total_UPDRS']  #st. dev. of total UPDRS

# Generate the normalized training and test datasets, remove unwanted regressors
Xsh_norm=(Xsh-mm)/ss# normalized data
ysh_norm=Xsh_norm['total_UPDRS']# regressand only
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#'],axis=1)# regressors only
#Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#', 'Jitter:DDP', 'Shimmer:DDA'],axis=1)
X_tr_norm=Xsh_norm[0:Ntr]
X_te_norm=Xsh_norm[Ntr:]
y_tr_norm=ysh_norm[0:Ntr]
y_te_norm=ysh_norm[Ntr:]
#LLS regression
#r = myreg(X_tr_norm, y_tr_norm)
#re.LLS()
w_hat=np.linalg.inv(X_tr_norm.T@X_tr_norm)@(X_tr_norm.T@y_tr_norm)
regressors = list(X_tr_norm.columns)
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
plt.savefig('./myLLS-what.png')
plt.show()
#Evaluate y_hat for test and training set
y_hat_te_norm = X_te_norm @ w_hat
y_hat_tr_norm = X_tr_norm @ w_hat

#De-normalize y_hat
y_hat_tr=y_hat_tr_norm*sy+my
y_tr=y_tr_norm*sy+my
y_hat_te=y_hat_te_norm*sy+my
y_te=y_te_norm*sy+my

#Histogram of the error Y - Y_hat
E_tr=(y_tr-y_hat_tr)# training
E_te=(y_te-y_hat_te)# test
e=[E_tr,E_te]
plt.figure(figsize=(6,4))
plt.hist(e,bins=50,density=True, histtype='bar',label=['training','test'])
plt.xlabel(r'$e=y-\^y$')
plt.ylabel(r'$P(e$ in bin$)$')
plt.legend()
plt.grid()
plt.title('LLS-Error histograms using all the training dataset')
plt.tight_layout()
plt.savefig('./LLS-hist.png')
plt.show()

#Plot regression line
plt.figure(figsize=(6,4))
plt.plot(y_te,y_hat_te,'.')
plt.legend()
v=plt.axis()
plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
plt.xlabel(r'$y$')
plt.axis('square')
plt.ylabel(r'$\^y$')
plt.grid()
plt.title('LLS-test')
plt.tight_layout()
plt.savefig('./LLS-yhat_vs_y.png')
plt.show()

#Errors and coefficients
E_tr_max=E_tr.max()
E_tr_min=E_tr.min()
E_tr_mu=E_tr.mean()
E_tr_sig=E_tr.std()
E_tr_MSE=np.mean(E_tr**2)
R2_tr=1-E_tr_MSE/(np.std(y_tr)**2)
c_tr=np.mean((y_tr-y_tr.mean())*(y_hat_tr-y_hat_tr.mean()))/(y_tr.std()*y_hat_tr.std())
E_te_max=E_te.max()
E_te_min=E_te.min()
E_te_mu=E_te.mean()
E_te_sig=E_te.std()
E_te_MSE=np.mean(E_te**2)
R2_te=1-E_te_MSE/(np.std(y_te)**2)
c_te=np.mean((y_te-y_te.mean())*(y_hat_te-y_hat_te.mean()))/(y_te.std()*y_hat_te.std())

cols=['min','max','mean','std','MSE','R^2','corr_coeff']
rows=['Training','test']
p=np.array([
    [E_tr_min,E_tr_max,E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr,c_tr],
    [E_te_min,E_te_max,E_te_mu,E_te_sig,E_te_MSE,R2_te,c_te],
            ])

results=pd.DataFrame(p,columns=cols,index=rows)
print(results)
# %%
