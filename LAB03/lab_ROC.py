import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk
#plt.rcParams["font.family"] = "Times New Roman"

# Function to evaluate sensitivity and specificity
def calcSensSpec(thresh, x0, x1):
    Np=np.sum(swab==1)
    n1=np.sum(x1>thresh)
    sens= n1/Np
    Nn=np.sum(swab==0)
    n0=np.sum(x0 < thresh)
    spec = n0/Nn
    return sens, spec

# Function to evalute Area Under a Curve (AUC) using trapezoidal rule
def calcAUC(n, sens):
    h = 1/n
    len_sens = len(sens)
    x = np.linspace(0, len_sens, num=n, endpoint=False, dtype=int)
    sum_buff = sens[x[0]]
    for i in range(1, n-1):
        sum_buff += 2*sens[x[i]]
    sum_buff += sens[x[n - 1]]
    AUC = h/2*sum_buff
    return AUC
# %%
plt.close('all')
xx = pd.read_csv("LAB03/covid_serological_results.csv")
# results from swab: 0= no illness, 1 = unclear, 2=illness
swab = xx.COVID_swab_res.values
swab[swab >= 1] = 1  # non reliable values are considered positive
print("\nDescription of the dataset:\n")
print(xx.describe())
print("\n===================================================================\n")
data = xx.values
data_norm = (data-data.mean())/data.std()  # normalization
clustering = sk.DBSCAN(eps=2.6).fit(data_norm)  # fitting
ii = np.argwhere(clustering.labels_ == -1)[:, 0]  # outliers
print("Outliers excluded using DBSCAN:\n")
print(xx.iloc[ii])
print("\n===================================================================\n")
xx = xx.drop(ii)
swab = xx.COVID_swab_res.values
Test1 = xx.IgG_Test1_titre.values
Test2 = xx.IgG_Test2_titre.values

######################### TEST 2 ####################################
x = Test2
y = swab
x0 = x[swab==0]
x1= x[swab==1]
Np=np.sum(swab==1)
Nn=np.sum(swab==0)
thresh= np.sort(Test2)
sens = np.zeros(len(thresh))    # sensitivity
spec = np.zeros(len(thresh))    # specificity
FA = np.zeros(len(thresh))  # False alarm
# Evaluate sens, spec given threshold having values of Test 2 sorted
for i in range(len(thresh)):
    sens[i], spec[i] = calcSensSpec(thresh[i], x0, x1)  
    FA[i] = 1 - spec[i]
# Plot specifity and sensitivity versus the threshold for Test 2
plt.figure()
plt.plot(thresh, spec, label="specifity - P(Tn|H)")
plt.plot(thresh, sens, label="sensitivity - P(Tp|D)")
plt.xlabel("threshold")
plt.title("Test 2")
plt.legend()
#plt.show()
# Plot ROC curve (Sensitivity vs Falsa alarm) for Test 2
plt.figure()
plt.plot(FA, sens)
plt.xlabel("False alarm - P(Tp|H)")
plt.ylabel("Sensitivity - P(Tn|H)")
plt.title("ROC curve")
#plt.show()
# Calculate AUC
AUC_test2 = calcAUC(n=100, sens=sens)

######################### TEST 1 #####################################
x = Test1
y = swab
x0 = x[swab==0]
x1= x[swab==1]
Np=np.sum(swab==1)
Nn=np.sum(swab==0)
thresh= np.sort(Test1)
sens = np.zeros(len(thresh))    # sensitivity
spec = np.zeros(len(thresh))    # specificity
FA = np.zeros(len(thresh))  # False alarm
# Evaluate sens, spec given threshold having values of Test 1 sorted
for i in range(len(thresh)):
    sens[i], spec[i] = calcSensSpec(thresh[i], x0, x1)  
    FA[i] = 1 - spec[i]
# Plot specifity and sensitivity versus the threshold for Test 1
plt.figure()
plt.plot(thresh, spec, label="specifity - P(Tn|H)")
plt.plot(thresh, sens, label="sensitivity - P(Tp|D)")
plt.xlabel("threshold")
plt.title("Test 1")
plt.legend()
#plt.show()
# Plot ROC curve (Sensitivity vs Falsa alarm) for Test 2
plt.figure()
plt.plot(FA, sens)
plt.xlabel("False alarm - P(Tp|H)")
plt.ylabel("Sensitivity - P(Tn|H)")
plt.title("ROC curve")
#plt.show()
AUC_test1 = calcAUC(n=100, sens=sens)

# Print AUC for test 1 and test 2
print("AUC for test 1:", round(AUC_test1, 4))
print("AUC for test 2:", round(AUC_test2,4))
print("\n===================================================================\n")
""" 
n1=np.sum(x1>thresh)
sens=n1/Np
n0=np.sum(x0 < thresh)
spec = n0/Nn
x = [x0, x1]
plt.hist(x, density=True)   #plot conditional pdf's
plt.show()
 """
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
