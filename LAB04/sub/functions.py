import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt# Function to generate dataframes
def generateDF(filedir,colnames,sensors,patients,activities,slices):
    # get the data from files for the selected patients
    # and selected activities
    # concatenate all the slices
    # generate a pandas dataframe with an added column: activity
    x=pd.DataFrame()
    for pat in patients:
        for a in activities:
            subdir='a'+f"{a:02d}"+'/p'+str(pat)+'/'
            for s in slices:
                filename=filedir+subdir+'s'+f"{s:02d}"+'.txt'
                #print(filename)
                x1=pd.read_csv(filename,usecols=sensors,names=colnames)
                x1['activity']=a*np.ones((x1.shape[0],),dtype=int)
                x=pd.concat([x,x1], axis=0, join='outer', ignore_index=True, 
                            keys=None, levels=None, names=None, verify_integrity=False, 
                            sort=False, copy=True)
    return x

# Function for sampling at different frequency (not 25 Hz)
def sampling(x, n_samples=50):
    
    lenDf = len(x)
    t = np.linspace(0, lenDf - 1, n_samples, dtype=int)
    sampled_x = pd.DataFrame(0, index = np.arange(len(t)), columns=x.columns)
    for i in range(len(t)):
        sampled_x.iloc[i] = x.iloc[t[i]]
    return sampled_x

# Function for average each sensor on the three different axis

def AverageSensors(x):
    x = x.values
    n_sensors = x.shape[1]
    N = x.shape[0]
    averaged_x = np.zeros([N, int(n_sensors/3)])
    for i in range(N):
        k = 0
        for j in range(int(n_sensors/3)):
            averaged_x[i][j] = np.linalg.norm(x[i, (j+k):(j+k+3)])
            k += 2
    averaged_x = pd.DataFrame(averaged_x, columns=['T_acc', 'T_gyro', 'T_mag',
                                                    'RA_acc', 'RA_gyro', 'RA_mag',
                                                    'LA_acc', 'LA_gyro,', 'LA_mag',
                                                    'RL_acc', 'RL_gyro', 'RL_mag',
                                                    'LL_acc', 'LL_gyro', 'LL_mag'])
    return averaged_x

def standardize_features(x):
    features = x.columns
    for feature in features:
        x[feature] = np.log(x[feature]+1e-9)
    return x

def evaluateCorr(x): #correlation for sensors
    x = x.values
    corr_matr = np.zeros([x.shape[0], x.shape[0]])
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            corr_matr[i][j] = np.corrcoef(x[i], x[j])[0][1]
    return corr_matr

def averageSampling(x, n_samp = 25):
    features = x.columns
    x = x.values
    len_x = x.shape[0]
    n_feat = x.shape[1]
    new_x = np.zeros([int(len_x/n_samp), n_feat])
    buff = 0
    for i in range(0, new_x.shape[0]):
        y = x[i:i+n_samp]
        new_x[i] = x[i:i+n_samp].mean(axis=0)
    new_x = pd.DataFrame(new_x, columns=features)
    return new_x

def butter_lowpass_filter(df, cutoff, fs, order):
    filtDF = np.zeros([len(df), len(df.columns)])
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    for i in range(len(df.columns)):
        y = filtfilt(b, a, df.iloc[:, i])
        filtDF[:, i] = y
    filtDF = pd.DataFrame(filtDF, columns=df.columns)
    return filtDF

def myPCA(x):
    x_norm = (x - x.mean())/x.std()
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(x_norm)
    principalDf = pd.DataFrame(data=principal_components)
    return principalDf