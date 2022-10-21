import numpy
import matplotlib.pyplot as plt
import pandas as pd

x = pd.read_csv("parkinsons_updrs.csv")
#x.describe().T
x.info()
features = list(x.columns)
print(features)
subj = pd.unique(x['subject#'])
print("The number of distinct patients in the dataset is ", len(subj))

X = pd.DataFrame()
for k in subj:
    xk = x[x['subject#'] == k]
    xk1 = xk.copy()
    xk1.test_time = xk1.test_time.astype(int)
    xk1['g'] = xk1['test_time']
    v = xk1.groupby('g').mean()
    X = pd.concat([X,v], axis=0, ignore_index=True)
print("\n\n\n", v)
