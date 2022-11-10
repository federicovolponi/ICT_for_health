'''
Federico Volponi s309709
LABORATORIO 1 - Linear Regression on Parkinson's disease data
17/11/2022
'''
import sub.regression as myreg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simple function to perform the average
def average(l):
    return sum(l)/len(l)

##########################################################################################

if __name__ == "__main__":
    
    ####################### Analysis of dataframe #########################################
    x = pd.read_csv("C:\Coding\ICT_for_health\LAB01\parkinsons_updrs.csv")  # read the dataset
    x.describe().T
    print("\n")
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

    n_different_seed = 20
    results_LLS = []
    results_SD = []
    results_LR = []
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
        R1 = myreg.regression(Xsh_norm, ysh_norm, Ntr, sy, my)

        R1.LLS()
        R1.steepestDescent()
        R1.localRegression(100)
        R1.denormalize()
        results_LLS.append(R1.errorsAndCoefficients(algorithm="LLS"))
        results_LR.append(R1.errorsAndCoefficients(algorithm="LR"))
        results_SD.append(R1.errorsAndCoefficients(algorithm="SD"))
        

    sum_LR = np.zeros((results_LR[0].shape)) #shape of one dataframe
    sum_LLS = np.zeros((results_LLS[0].shape))
    sum_SD = np.zeros((results_SD[0].shape))
    for i in range(n_different_seed):
        sum_LR += results_LR[i].values
        sum_LLS += results_LLS[i].values
        sum_SD += results_SD[i].values
    cols=['min','max','mean','std','MSE','R^2','corr_coeff']
    rows=['Training','test']
    results_LR=pd.DataFrame(sum_LR / n_different_seed,columns=cols,index=rows)
    results_LLS=pd.DataFrame(sum_LLS / n_different_seed,columns=cols,index=rows)
    results_SD=pd.DataFrame(sum_SD / n_different_seed,columns=cols,index=rows)
    print("\nLLS error table - averaged results on 20 seeds")
    print(results_LLS)
    print("\nSD error table - averaged results on 20 seeds")
    print(results_SD)
    print("\nLLR error table - averaged results on 20 seeds")
    print(results_LR)



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
    R1 = myreg.regression(Xsh_norm, ysh_norm, Ntr, sy, my)
    R1.localRegression(100)
    R1.LLS()
    R1.steepestDescent()
    R1.plot_LLS_vs_SD()
    R1.denormalize()

    R1.plotRegressionLine(title="regressionline_LLS", algorithm="LLS")
    R1.plotRegressionLine(title="regressionline_SD", algorithm="SD")
    R1.plotRegressionLine(title="regressionline_LR", algorithm="LR")
    R1.plotHistrogram(title="LLS-Error", algorithm="LLS")
    R1.plotHistrogram(title="SD-Error", algorithm="SD")
    R1.plotHistrogram(title="LR-Error", algorithm="LR")
    R1.errorsAndCoefficients(algorithm="LLS", toPrint=True)
    R1.errorsAndCoefficients(algorithm="SD", toPrint=True)
    R1.errorsAndCoefficients(algorithm="LR", toPrint=True)