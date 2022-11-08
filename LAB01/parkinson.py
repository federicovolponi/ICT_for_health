
import sub.regression as myreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
Nf = Nc - 4
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
y_te = np.zeros([Ntr,n_different_seed])
y_tr = np.zeros([Ntr,n_different_seed])
y_hat_tr_LLS = np.zeros([Ntr,n_different_seed])
y_hat_te_LLS = np.zeros([Nte,n_different_seed])
w_hat_LLS = np.zeros([Nf, n_different_seed])
y_hat_tr_SD = np.zeros([Ntr,n_different_seed])
y_hat_te_SD = np.zeros([Nte,n_different_seed])
w_hat_SD = np.zeros([Nf, n_different_seed])
y_hat_tr_LR = np.zeros([Ntr,n_different_seed])
y_hat_te_LR = np.zeros([Nte,n_different_seed])

for i in range(n_different_seed):
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
    y_te[:, i] = r2.y_te
    y_tr[:, i] = r2.y_tr
    #r2.localRegression(100)
    r2.LLS()
    y_hat_te_LLS[:, i] = r2.y_hat_te_LLS
    y_hat_tr_LLS[:, i] = r2.y_hat_tr_LLS
    w_hat_LLS[:, i] =  r2.w_hat_LLS
    
    r2.steepestDescent()
    y_hat_te_SD[:, i] = r2.y_hat_te_SD[:, 0]
    y_hat_tr_SD[:, i] = r2.y_hat_tr_SD[:, 0]
    w_hat_SD[:, i] =  r2.w_hat_SD[:, 0]

    r2.localRegression(100)
    y_hat_te_LR[:, i] = r2.y_hat_te_LR[:, 0]
    y_hat_tr_LR[:, i] = r2.y_hat_tr_LR[:, 0]



for i in range(Ntr):
    sum_y_te = 0
    sum_y_tr = 0
    sum_y_hat_te_LLS = 0
    sum_y_hat_tr_LLS = 0
    sum_y_hat_te_SD = 0
    sum_y_hat_tr_SD = 0
    sum_y_hat_te_LR = 0
    sum_y_hat_tr_LR = 0
    sum_w_hat_LLS = 0
    sum_w_hat_SD = 0
    for j in range(n_different_seed):
        sum_y_te += y_te[i][j]
        sum_y_tr += y_tr[i][j]
        sum_y_hat_te_LLS += y_hat_te_LLS[i][j]
        sum_y_hat_tr_LLS += y_hat_tr_LLS[i][j]
        sum_y_hat_te_SD += y_hat_te_SD[i][j]
        sum_y_hat_tr_SD += y_hat_tr_SD[i][j]
        sum_y_hat_te_LR += y_hat_te_LR[i][j]
        sum_y_hat_tr_LR += y_hat_tr_LR[i][j]
        if i <= Nf - 1:
            sum_w_hat_LLS += w_hat_LLS[i][j]
            sum_w_hat_SD += w_hat_SD[i][j]

    r2.y_te[i] = sum_y_te / n_different_seed
    r2.y_tr[i] = sum_y_tr / n_different_seed
    r2.y_hat_te_LLS[i]= sum_y_hat_te_LLS / n_different_seed
    r2.y_hat_tr_LLS[i]= sum_y_hat_tr_LLS / n_different_seed
    r2.y_hat_te_SD[i]= sum_y_hat_te_SD / n_different_seed
    r2.y_hat_tr_SD[i]= sum_y_hat_tr_SD / n_different_seed
    r2.y_hat_te_LR[i]= sum_y_hat_te_LR / n_different_seed
    r2.y_hat_tr_LR[i]= sum_y_hat_tr_LR / n_different_seed
    
    if i <= Nf - 1:
        r2.w_hat_SD[i] = sum_w_hat_SD / n_different_seed
        r2.w_hat_SD[i] = sum_w_hat_SD / n_different_seed

r2.plot_LLS_vs_SD()
r2.denormalize(sy, my)

r2.plotRegressionLine(title="regressionline_LLS", algorithm="LLS")
r2.plotRegressionLine(title="regressionline_SD", algorithm="SD")
r2.plotRegressionLine(title="regressionline_LR", algorithm="LR")
r2.plotHistrogram(title="LLS-Error", algorithm="LLS")
r2.plotHistrogram(title="SD-Error", algorithm="SD")
r2.plotHistrogram(title="LR-Error", algorithm="LR")
r2.errorsAndCoefficients(algorithm="LLS")
r2.errorsAndCoefficients(algorithm="SD")
r2.errorsAndCoefficients(algorithm="LR")


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

r2.plotRegressionLine(title="regressionline_LLS", algorithm="LLS")
r2.plotRegressionLine(title="regressionline_SD", algorithm="SD")
r2.plotRegressionLine(title="regressionline_LR", algorithm="LR")
r2.plotHistrogram(title="LLS-Error", algorithm="LLS")
r2.plotHistrogram(title="SD-Error", algorithm="SD")
r2.plotHistrogram(title="LR-Error", algorithm="LR")
r2.errorsAndCoefficients(algorithm="LLS")
r2.errorsAndCoefficients(algorithm="SD")
r2.errorsAndCoefficients(algorithm="LR")

 """