import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk
#plt.rcParams["font.family"] = "Times New Roman"

def calcSensSpec(thresh, x0, x1):
    
    Np=np.sum(swab==1)
    Nn=np.sum(swab==0)
    n1=np.sum(x1>thresh)
    sens=n1/Np
    n0=np.sum(x0 < thresh)
    spec = n0/Nn
# %%
plt.close('all')
xx = pd.read_csv("LAB03/covid_serological_results.csv")
# results from swab: 0= no illness, 1 = unclear, 2=illness
swab = xx.COVID_swab_res.values
swab[swab >= 1] = 1  # non reliable values are considered positive
print("Description of the dataset:\n")
print(xx.describe())

data = xx.values
data_norm = (data-data.mean())/data.std()  # normalization
clustering = sk.DBSCAN(eps=2.6).fit(data_norm)  # fitting
ii = np.argwhere(clustering.labels_ == -1)[:, 0]  # outliers
print(xx.iloc[ii])
xx = xx.drop(ii)
swab = xx.COVID_swab_res.values
Test1 = xx.IgG_Test1_titre.values
Test2 = xx.IgG_Test2_titre.values

x = Test2
y = swab
x0 = x[swab==0]
x1= x[swab==1]
Np=np.sum(swab==1)
Nn=np.sum(swab==0)
thresh= np.sort(Test2)
n1=np.sum(x1>thresh)
sens=n1/Np
n0=np.sum(x0 < thresh)
spec = n0/Nn
x = [x0, x1]
plt.hist(x, density=True)   #plot conditional pdf's
plt.show()

data_descr = False
# %% data analysis

if data_descr:
    xx_pos = xx[xx.COVID_swab_res == 1]
    xx_neg = xx[xx.COVID_swab_res == 0]
    xx_pos = xx_pos.drop(columns=['COVID_swab_res'])
    xx_neg = xx_neg.drop(columns=['COVID_swab_res'])
    xx_pos.hist(bins=50)
    xx_neg.hist(bins=50)
    pd.plotting.scatter_matrix(xx_pos)
    pd.plotting.scatter_matrix(xx_neg)
    xx_norm = (xx-xx.mean())/xx.std()
    c = xx_norm.corr()
    plt.figure()
    plt.matshow(np.abs(c.values), fignum=0)  # absolute value of corr.coeffs
    plt.xticks(np.arange(3), xx.columns, rotation=90)
    plt.yticks(np.arange(3), xx.columns, rotation=0)
    plt.colorbar()
    plt.title('Correlation coefficients of the original dataset')
    plt.tight_layout()
    plt.show()
