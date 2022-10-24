import sub.regression as myreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = pd.read_csv("/home/federico/Coding/ICT_for_health/LAB01/parkinsons_updrs.csv")
#Analysis of dataframe
'''
x.describe().T
x.info()
features = list(x.columns)
print(features)
'''
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
plt.savefig("ICT_for_health\LAB01\charts\corr_coeff.png")
#plt.show()
plt.figure()

c.total_UPDRS.plot()
plt.grid()
plt.xticks(np.arange(len(features)), features, rotation = 90)
plt.title("corr. coeff among total UPDRS and the other features")
plt.tight_layout()
#plt.show()

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

#LLS regression
#All the features
r1 = myreg.regression(Xsh_norm, ysh_norm, Ntr)
r1.LLS("LLS-what-all.png")
#r1.steepestDescent()
#Excluding Jitter:DDP and Shimmer:DDA
Xsh_norm=Xsh_norm.drop(['Jitter:DDP', 'Shimmer:DDA'],axis=1)
r2 = myreg.regression(Xsh_norm, ysh_norm, Ntr)
r2.LLS()
r2.steepestDescent()
#De-normalize y_hat
r1.y_hat_tr=r1.y_hat_tr*sy+my
r1.y_tr=r1.y_tr*sy+my
r1.y_hat_te=r1.y_hat_te*sy+my
r1.y_te=r1.y_te*sy+my

r1.plotHistrogram("LLS-hist_all.png")

r2.y_hat_tr=r2.y_hat_tr*sy+my
r2.y_tr=r2.y_tr*sy+my
r2.y_hat_te=r2.y_hat_te*sy+my
r2.y_te=r2.y_te*sy+my

r2.plotHistrogram()

#Plot regression line
r1.plotRegressionLine("y_hat_vs_y-all.png")
r2.plotRegressionLine()

r1.errorsAndCoefficients()
r2.errorsAndCoefficients()
