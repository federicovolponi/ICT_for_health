import numpy as np
import pandas as pd

# Function to generate dataframes
def generateDF(filedir,colnames,patients,activities,slices):
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
              x1=pd.read_csv(filename,names=colnames)
              x1['activity']=a*np.ones((x1.shape[0],),dtype=int)
              x=pd.concat([x,x1], axis=0, join='outer', ignore_index=True, 
                          keys=None, levels=None, names=None, verify_integrity=False, 
                          sort=False, copy=True)
  return x

def sampling(x):
    
    lenDf = len(x)
    t = np.linspace(0, lenDf - 1, 60, dtype=int)
    sampled_x = pd.DataFrame(0, index = np.arange(len(t)), columns=x.columns)
    for i in range(len(t)):
        sampled_x.iloc[i] = x.iloc[t[i]]
    return sampled_x
