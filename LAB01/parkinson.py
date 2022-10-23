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
w_hat = np.linalg.inv(X_tr_norm.T @ X_tr_norm) @ (X_tr_norm.T @ y_tr_norm)

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
plt.savefig('./LLS-what.png')
plt.show()