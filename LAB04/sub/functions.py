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

################################## Evaluate cumulative moving average CMA #############################
def cumulativeMovingAverage(df):
    avgDf = np.zeros([len(df), len(df.columns)])
    
    for i in range(len(df.columns)):
        x = df.iloc[:, i]
        avgDf[:, i] = x.expanding().mean()
    avgDf = pd.DataFrame(avgDf, columns=df.columns)
    return avgDf