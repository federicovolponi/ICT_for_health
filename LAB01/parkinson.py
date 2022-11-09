
import sub.regression as myreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def average(l):
    return sum(l)/len(l)

n_different_seed = 20
x = pd.read_csv("C:\Coding\ICT_for_health\LAB01\parkinsons_updrs.csv")

#Analysis of dataframe
x.describe().T
x.info()
subj = pd.unique(x['subject#']) #unique values of patient ID
print("\nThe number of distinct patients in the dataset is ", len(subj))

X = pd.DataFrame()
for k in subj:
    xk = x[x['subject#'] == k]  #data of user k
    xk1 = xk.copy()
    xk1.test_time = xk1.test_time.astype(int)   #remove decimal values to consider just the day and not the hour
    xk1['g'] = xk1['test_time'] # add a new feature g with test_time values
    v = xk1.groupby('g').mean() # group by the g features(test_time) to have averaged values per day 
    X = pd.concat([X,v], axis=0, ignore_index=True) #Concatenate the k-patients by row ignoring index

features = list(x.columns)
print("\nThe dataset shape after the mean is: ", X.shape)
print("\nThe features of the dataset are ", len(features))
print("\n", features)
print("\n\n")
Np, Nc = X.shape
Ntr = int(Np*0.5)
Nte = Np - Ntr
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
plt.savefig("C:\Coding\ICT_for_health\LAB01\charts\corr_coeff_features.png")
#plt.show()

plt.figure()
c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation = 90)
plt.title("Correlation coefficient among total UPDRS and the other features")
plt.tight_layout()
plt.savefig("C:\Coding\ICT_for_health\LAB01\charts\corr_coeffTotal.png")
#plt.show()

#Shuffle the data
E_tr_max = []
E_tr_min = []
E_tr_mu = []
E_tr_sig = []
E_tr_MSE = []
R2_tr = []
c_tr = []
E_te_max = []
E_te_min = []
E_te_mu = []
E_te_sig = []
E_te_MSE = []
R2_te = []
c_te = []

""" for i in range(n_different_seed):
    seed = np.random.seed()
    Xsh = X.sample(frac=1, replace=False, random_state=seed, axis=0, ignore_index=True)

    # Generate training and test matrices
    X_tr = Xsh[0:Ntr]   #dataframe of the training data
    mm = X_tr.mean()
    ss = X_tr.std()
    my = mm['total_UPDRS']  #mean of total_UPDRS
    sy = ss['total_UPDRS']  #st. dev. of total UPDRS

    # Generate the normalized training and test datasets, remove unwanted regressors
    Xsh_norm=(Xsh-mm)/ss  #normalized data
    ysh_norm=Xsh_norm['total_UPDRS']  #regressand only
    Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#'],axis=1) #regressors only

    #LLS regression
    #All the features
    #Excluding Jitter:DDP and Shimmer:DDA
    Xsh_norm=Xsh_norm.drop(['Jitter:DDP', 'Shimmer:DDA'],axis=1)
    r2 = myreg.regression(Xsh_norm, ysh_norm, Ntr, sy, my)
    #r2.localRegression(100)
    r2.LLS()
    r2.steepestDescent()
    #r2.localRegression(100)
    
    #r2.errorsAndCoefficients(algorithm="LLS")
    r2.errorsAndCoefficients(algorithm="SD")
    #r2.errorsAndCoefficients(algorithm="LR")
    r2.denormalize(sy, my, algorithm="SD")
    E_tr_max.append(r2.E_tr_max)
    E_tr_min.append(r2.E_tr_min)
    E_tr_mu.append(r2.E_tr_mu)
    E_tr_sig.append(r2.E_tr_sig)
    E_tr_MSE.append(r2.E_tr_MSE)
    R2_tr.append(r2.R2_tr)
    c_tr.append(r2.c_tr)
    E_te_max.append(r2.E_te_max)
    E_te_min.append(r2.E_te_min)
    E_te_mu.append(r2.E_te_mu)
    E_te_sig.append(r2.E_te_sig)
    E_te_MSE.append(r2.E_te_MSE)
    R2_te.append(r2.R2_te)
    c_te.append(r2.c_te)

cols=['min','max','mean','std','MSE','R^2','corr_coeff']
rows=['Training','test']
p=np.array([
    [average(E_tr_min),average(E_tr_max),average(E_tr_mu),average(E_tr_sig),average(E_tr_MSE),average(R2_tr),average(c_tr)],
    [average(E_te_min),average(E_te_max),average(E_te_mu),average(E_te_sig),average(E_te_MSE),average(R2_te),average(c_te)],
            ])

results=pd.DataFrame(p,columns=cols,index=rows)
print(results, "\n\n")

 """


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
Xsh_norm=(Xsh-mm)/ss  #normalized data
ysh_norm=Xsh_norm['total_UPDRS']  #regressand only
Xsh_norm=Xsh_norm.drop(['total_UPDRS','subject#'],axis=1) #regressors only

#LLS regression
#All the features
#Excluding Jitter:DDP and Shimmer:DDA
Xsh_norm=Xsh_norm.drop(['Jitter:DDP', 'Shimmer:DDA'],axis=1)
r2 = myreg.regression(Xsh_norm, ysh_norm, Ntr, sy, my)
r2.localRegression(100)
r2.LLS()
r2.steepestDescent()
r2.plot_LLS_vs_SD()
r2.denormalize(sy, my)
r2.denormalize(sy, my, algorithm="LLS")
r2.denormalize(sy, my, algorithm="SD")
r2.denormalize(sy, my, algorithm="LR")
r2.plotRegressionLine(title="regressionline_LLS", algorithm="LLS")
r2.plotRegressionLine(title="regressionline_SD", algorithm="SD")
r2.plotRegressionLine(title="regressionline_LR", algorithm="LR")
r2.plotHistrogram(title="LLS-Error", algorithm="LLS")
r2.plotHistrogram(title="SD-Error", algorithm="SD")
r2.plotHistrogram(title="LR-Error", algorithm="LR")
r2.errorsAndCoefficients(algorithm="LLS", toPrint=True)
r2.errorsAndCoefficients(algorithm="SD", toPrint=True)
r2.errorsAndCoefficients(algorithm="LR", toPrint=True)

