import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as sk
import scipy
from sklearn import metrics
from sklearn.metrics import roc_auc_score,average_precision_score
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

# %%
plt.close('all')
xx = pd.read_csv("ICT_for_health\LAB03\covid_serological_results.csv")
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
# Find specificty and sensitivity for a given threshold and plot pdf's
thresh = 5
Np=np.sum(swab==1)
Nn=np.sum(swab==0)
n1=np.sum(x1>thresh)
sens=n1/Np
n0=np.sum(x0 < thresh)
spec = n0/Nn
x = [x0, x1]
plt.figure()
plt.hist(x, density=True)   #plot conditional pdf's
plt.title("pdf for Test 2")
#plt.show()

# Find specificty and sensitivity for a threshold with the sorted values of Test 2
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
plt.grid()
plt.title("Test 2")
plt.legend()
#plt.show()

# Plot ROC curve (Sensitivity vs Falsa alarm) for Test 2
plt.figure()
plt.plot(FA, sens)
plt.xlabel("False alarm - P(Tp|H)")
plt.ylabel("Sensitivity - P(Tn|H)")
plt.grid()
plt.title("ROC curve - Test 2")
#plt.show()

# ROC curve using sklearn for test 2
""" fpr,tpr,thresh = metrics.roc_curve(swab,Test2,pos_label=1)
plt.figure
plt.plot(fpr, tpr)
plt.show() """

# Calculate AUC for test 2
AUC_test2 = np.abs(scipy.integrate.trapezoid(sens, FA))

# Find the best threshold (False negative small, Sensitivity high)
prev = 0.01 #prevalence
healthy = 1 - prev  # healthy
FNR = 1 - sens
# probability that a person with a negative test is truly healthy
P_H_Tn = (spec * healthy)/(spec * healthy +  FNR * prev)
# probability that a person with a negative test is ill
P_D_Tn = 1 - P_H_Tn
# probability that a person with a positive test is truly ill
P_D_Tp = (sens * prev)/(sens * prev + FA * healthy)
# probabilty that a person with a positive test is healthy
P_H_Tp = 1 - P_D_Tp

# P(D|Tp), P(D|Tn) versus threshold
plt.figure()
plt.plot(thresh, P_D_Tp, label= "P(D|Tp)")
plt.plot(thresh,P_D_Tn, label="P(D|Tn)")
plt.grid()
plt.xlabel("Threshold")
plt.title("P(D|Tp), P(D|Tn) versus threshold - Test 2")
plt.legend()
plt.show()
# P(H|Tp), P(H|Tn) versus threshold
plt.figure()
plt.plot(thresh, P_H_Tn, label= "P(H|Tn)")
plt.plot(thresh,P_H_Tp, label="P(H|Tp)")
plt.grid()
plt.xlabel("Threshold")
plt.title("P(H|Tp), P(H|Tn) versus threshold - Test 2")
plt.legend()
plt.show()
# P(D|Tp) versus P(H|Tn)
plt.figure()
plt.plot(P_D_Tp, P_H_Tn)
plt.grid()
plt.xlabel("P(D|Tp)")
plt.ylabel("P(H|Tn)")
plt.title("P(D|Tp) versus P(H|Tn) - Test 2")
if prev == 0.01:
    plt.axis([0, 1, 0.990, 1])
plt.tight_layout()
plt.show()


######################### TEST 1 #####################################
x = Test1
y = swab
x0 = x[swab==0]
x1= x[swab==1]
# Find specificty and sensitivity for a given threshold and plot pdf's
thresh = 5
Np=np.sum(swab==1)
Nn=np.sum(swab==0)
n1=np.sum(x1>thresh)
sens=n1/Np
n0=np.sum(x0 < thresh)
spec = n0/Nn
x = [x0, x1]
plt.figure()
plt.title("pdf for Test 1")
plt.hist(x, density=True)   #plot conditional pdf's
#plt.show()

# Find specificty and sensitivity for a threshold with the sorted values of Test 1
thresh= np.sort(Test1)
thresh[0] = 0
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
plt.grid()
plt.legend()
#plt.show()
# Plot ROC curve (Sensitivity vs Falsa alarm) for Test 1
plt.figure()
plt.plot(FA, sens)
plt.xlabel("False alarm - P(Tp|H)")
plt.ylabel("Sensitivity - P(Tn|H)")
plt.grid()
plt.title("ROC curve - Test 1")
plt.show()

# ROC curve using sklearn for test 1
""" fpr,tpr,thresh = metrics.roc_curve(swab,Test2,pos_label=1)
plt.figure
plt.plot(fpr, tpr)
plt.show()
 """
# Calculate AUC for test 1
AUC_test1 = np.abs(scipy.integrate.trapezoid(sens, FA))

# Find the best threshold (False negative small, Sensitivity high)
# P(D|Tp), P(D|Tn) versus threshold
plt.figure()
plt.plot(thresh, P_D_Tp, label= "P(D|Tp)")
plt.plot(thresh,P_D_Tn, label="P(D|Tn)")
plt.grid()
plt.xlabel("Threshold")
plt.title("P(D|Tp), P(D|Tn) versus threshold - Test 1")
plt.legend()
plt.show()
# P(H|Tp), P(H|Tn) versus threshold
plt.figure()
plt.plot(thresh, P_H_Tn, label= "P(H|Tn)")
plt.plot(thresh,P_H_Tp, label="P(H|Tp)")
plt.grid()
plt.xlabel("Threshold")
plt.title("P(H|Tp), P(H|Tn) versus threshold - Test 1")
plt.legend()
plt.show()
# P(D|Tp) versus P(H|Tn)
plt.figure()
plt.plot(P_D_Tp, P_H_Tn)
plt.grid()
plt.xlabel("P(D|Tp)")
plt.ylabel("P(H|Tn)")
plt.title("P(D|Tp) versus P(H|Tn) - Test 1")
if prev == 0.01:
    plt.axis([0, 1, 0.990, 1])
plt.tight_layout()
plt.show()

# Print AUC for test 1 and test 2
print("AUC for test 1:", round(AUC_test1, 3))
print("AUC for test 2:", round(AUC_test2,3))
print("\n===================================================================\n")

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

# Find the best threshold (False negative small, Sensitivity high)
prev = 0.01 #prevalence
healthy = 1 - prev  # healthy
FNR = 1 - sens
# probability that a person with a negative test is truly healthy
P_H_Tn = (spec * healthy)/(spec * healthy +  FNR * prev)
# probability that a person with a negative test is ill
P_D_Tn = 1 - P_H_Tn
# probability that a person with a positive test is truly ill
P_D_Tp = (sens * prev)/(sens * prev + FA * healthy)
# probabilty that a person with a positive test is healthy
P_H_Tp = 1 - P_D_Tp
