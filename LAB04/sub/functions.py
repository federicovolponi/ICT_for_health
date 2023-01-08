import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

############################ Function to generate Dataframe #################################
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

######################## Evaluate a sampling, averaging the data in between #########################
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
    
####################### Map sensor names to corresponding sensor's number ###############################
def mapSensors(feat_importances, sensDic):
    feat = feat_importances
    sensors = []
    for i in range(len(feat_importances)):
        sensors.append(sensDic.get(feat[i]))
    sensors.sort()
    return sensors

################ Evaluate the importance of features by picking the n with smallest variance #################
def featureImportanceVar(X, colNames, n_smallest):
    X = pd.DataFrame(X, columns=colNames)
    var_X = X.var()
    var_XSmall = var_X.nsmallest(n_smallest)
    features = var_XSmall.index.tolist()
    return features

################################ Evaluate a linear interpolation of the data ############################
def interpolation(df, nPoints):
    interpDF =  np.zeros([int(nPoints*len(df)), len(df.columns)])
    x = np.arange(len(df))
    x_new = np.linspace(0, len(df)-1, num=int(len(df)*nPoints))
    for i in range(len(df.columns)):
        f = interp1d(x, df.iloc[:, i])
        y = f(x_new)
        interpDF[:, i] = y
    interpDF = pd.DataFrame(interpDF, columns=df.columns)
    return interpDF
