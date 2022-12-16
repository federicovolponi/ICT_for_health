import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import sklearn.ensemble as ske

pd.set_option('display.precision', 3,'display.max_columns',50,'display.max_rows',50)
np.set_printoptions(precision=3)
#np.set_printoptions(linewidth=100)
#%%
# define the feature names:
feat_names=['age','bp','sg','al','su','rbc','pc',
'pcc','ba','bgr','bu','sc','sod','pot','hemo',
'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
'ane','classk']
ff=np.array(feat_names)
feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
         'num','num','num','num','num','num','num','num','num',
         'cat','cat','cat','cat','cat','cat','cat'])
# import the dataframe:
#xx=pd.read_csv("./data/chronic_kidney_disease.arff",sep=',',
#               skiprows=29,names=feat_names, 
#               header=None,na_values=['?','\t?'],
#               warn_bad_lines=True)
xx=pd.read_csv("./data/chronic_kidney_disease_v2.arff",sep=',',
    skiprows=29,names=feat_names, 
    header=None,na_values=['?','\t?'],)
Np,Nf=xx.shape
#%% change categorical data into numbers:
mapping={
    'normal':0,
    'abnormal':1,
    'present':1,
    'notpresent':0,
    'yes':1,
    ' yes':1,
    'no':0,
    '\tno':0,
    '\tyes':1,
    'ckd':1,
    'notckd':0,
    'poor':1,
    'good':0,
    'ckd\t':1}
xx=xx.replace(mapping.keys(),mapping.values())

# key_list=["normal","abnormal","present","notpresent","yes",
# "no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
# key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
# xx=xx.replace(key_list,key_val)
print(xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values
print(xx.info())
#%% manage the missing data through regression
x=xx.copy()
# drop rows with less than 19=Nf-6 recorded features:
x=x.dropna(thresh=19)
x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
n=x.isnull().sum(axis=1)# check the number of missing values in each row
print('max number of missing values in the reduced dataset: ',n.max())
print('number of points in the reduced dataset: ',len(n))
# take the rows with exctly Nf=25 useful features; this is going to be the training dataset
# for regression
Xtrain=x.dropna(thresh=25)
Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
# get the possible values (i.e. alphabet) for the categorical features
alphabets=[]
for k in range(len(feat_cat)):
    if feat_cat[k]=='cat':
        val=Xtrain.iloc[:,k]
        val=val.unique()
        alphabets.append(val)
    else:
        alphabets.append('num')
# run regression tree on all the missing data
#normalize the training dataset
mm=Xtrain.mean(axis=0)
ss=Xtrain.std(axis=0)
Xtrain_norm=(Xtrain-mm)/ss
#%% perform regression
# normalize the test dataset using the coeffs found for the 
# training dataset
Xtest=x.drop(x[x.isnull().sum(axis=1)==0].index)
Xtest.reset_index(drop=True, inplace=True)# reset the index of the dataframe
Xtest_norm=(Xtest-mm)/ss
Np,Nf=Xtest_norm.shape
regr=tree.DecisionTreeRegressor()# instantiate the regressor
for kk in range(Np):
    xrow=Xtest_norm.iloc[kk]#k-th row
    mask=xrow.isna()# columns with nan in row kk
    Data_tr_norm=Xtrain_norm.loc[:,~mask]# remove the columns from the training dataset
    y_tr_norm=Xtrain_norm.loc[:,mask]# columns to be regressed
    regr=regr.fit(Data_tr_norm,y_tr_norm)# find the regression tree
    Data_te_norm=Xtest_norm.loc[kk,~mask].values.reshape(1,-1)
    ytest_norm=regr.predict(Data_te_norm)
    a=xrow.values.astype(float)
    a[mask]=ytest_norm.flatten()
    Xtest_norm.iloc[kk]=a # substitute nan with regressed values
Xtest_new=Xtest_norm*ss+mm # denormalize
#%% substitute regressed numerical values with the closest element in the alphabet
index=np.argwhere(feat_cat=='cat').flatten()
for k in index:
    val=alphabets[k] # possible values for the feature
    c=Xtest_new.iloc[:,k].values # values in the column
    c=c.reshape(-1,1)# column vector
    val=val.reshape(1,-1) # row vector
    d=(val-c)**2 # matrix with all the distances w.r.t. the alphabet values
    ii=d.argmin(axis=1) # find the index of the closest alphabet value
    Xtest_new.iloc[:,k]=val.flatten()[ii]
#%% get the new dataset with no missing values
X_new= pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)
X_new.to_csv('kidney_no_nan.csv')
#%% check the distributions
L=X_new.shape[0]
plotCDF=True
if plotCDF:
    for k in range(Nf):
        plt.figure()
        a=xx.iloc[:,k].dropna()
        M=a.shape[0]
        plt.plot(np.sort(a),np.arange(M)/M,label='original dataset')
        plt.plot(np.sort(X_new.iloc[:,k]),np.arange(L)/L,label='regressed dataset')
        plt.title('CDF of '+xx.columns[k])
        plt.xlabel('x')
        plt.ylabel('P(X<=x)')
        plt.grid()
        plt.legend(loc='upper left')
############################ Decision tree #####################################
# add here the missing lines

r=np.random.randint(100)
print('------------------')
print('Random seed: ',r)
Xsh = X_new.sample(frac = 1,replace=False, random_state=r,axis=0, ignore_index=True)
Ntrain = L//50
XshTrain = Xsh[Xsh.index<Ntrain]
XshTest = Xsh[Xsh.index>=Ntrain]

target = XshTrain['classk']
inform = XshTrain.drop('classk',axis=1)
clfX = tree.DecisionTreeClassifier(criterion = 'entropy')
clfX = clfX.fit(inform,target)
test_pred = clfX.predict(XshTest.drop('classk',axis=1))

from sklearn.metrics import accuracy_score, confusion_matrix
print("############### DECISION TREE #########################")
print('accuracy =', accuracy_score(XshTest['classk'],test_pred))
print('Confusion matrix')
print(confusion_matrix(XshTest['classk'],test_pred))

target_names = ['notckd','ckd']
fig, axes = plt.subplots()
tree.plot_tree(clfX, feature_names=feat_names[:24],
                    class_names = target_names,
                    rounded=True,
                    proportion=False,
                    filled = True)
plt.title(f"shuffled data (seed {r})")
plt.savefig(f"threeshuffle{r}.png")

############################## Random Forest #########################################
clfRandF = ske.RandomForestClassifier(n_estimators=100, criterion="entropy")
clfRandF = clfRandF.fit(inform, target)
test_pred = clfRandF.predict(XshTest.drop('classk',axis=1))

accuracy_tr = clfRandF.score(inform, target)
accuracy_te = clfRandF.score(XshTest.drop('classk',axis=1), XshTest['classk'])
print("############### RANDOM FOREST #########################")
print("Accuracy for training:", accuracy_tr)
print("Accuracy for test:", accuracy_te)
print('Confusion matrix')
print(confusion_matrix(XshTest['classk'],test_pred))