#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:04:25 2022

@author: mvisintin

Sensor units are calibrated to acquire data at 25 Hz sampling 
frequency. 
The 5-min signals are divided into 5-sec segments so that 
480(=60x8) signal segments are obtained for each activity.

"""
#%% libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sub.functions as myFn
from sklearn.preprocessing import  StandardScaler
from sklearn.decomposition import PCA
import sklearn.cluster as sk
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time
start_time = time.time()
cm = plt.get_cmap('gist_rainbow')
line_styles=['solid','dashed','dotted']
#pd.set_option('display.precision', 3)
#%%

#%% initialization
plt.close('all')
filedir='LAB04/data/'
pathCharts = 'LAB04/charts/'
sensNames=[
        'T_xacc', 'T_yacc', 'T_zacc', 
        'T_xgyro','T_ygyro','T_zgyro',
        'T_xmag', 'T_ymag', 'T_zmag',
        'RA_xacc', 'RA_yacc', 'RA_zacc', 
        'RA_xgyro','RA_ygyro','RA_zgyro',
        'RA_xmag', 'RA_ymag', 'RA_zmag',
        'LA_xacc', 'LA_yacc', 'LA_zacc', 
        'LA_xgyro','LA_ygyro','LA_zgyro',
        'LA_xmag', 'LA_ymag', 'LA_zmag',
        'RL_xacc', 'RL_yacc', 'RL_zacc', 
        'RL_xgyro','RL_ygyro','RL_zgyro',
        'RL_xmag', 'RL_ymag', 'RL_zmag',
        'LL_xacc', 'LL_yacc', 'LL_zacc', 
        'LL_xgyro','LL_ygyro','LL_zgyro',
        'LL_xmag', 'LL_ymag', 'LL_zmag']

sensDic = {'T_xacc': 0, 'T_yacc':1, 'T_zacc':2, 
        'T_xgyro':3,'T_ygyro':4,'T_zgyro':5,
        'T_xmag':6, 'T_ymag':7, 'T_zmag':8,
        'RA_xacc':9, 'RA_yacc':10, 'RA_zacc':11, 
        'RA_xgyro':12,'RA_ygyro':13,'RA_zgyro':14,
        'RA_xmag':15, 'RA_ymag':16, 'RA_zmag':17,
        'LA_xacc':18, 'LA_yacc':19, 'LA_zacc':20, 
        'LA_xgyro':21,'LA_ygyro':22,'LA_zgyro':23,
        'LA_xmag':24, 'LA_ymag':25, 'LA_zmag':26,
        'RL_xacc':27, 'RL_yacc':28, 'RL_zacc':29, 
        'RL_xgyro':30,'RL_ygyro':31,'RL_zgyro':32,
        'RL_xmag':33, 'RL_ymag':34, 'RL_zmag':35,
        'LL_xacc':36, 'LL_yacc':37, 'LL_zacc':38, 
        'LL_xgyro':39,'LL_ygyro':40,'LL_zgyro':41,
        'LL_xmag':42, 'LL_ymag':43, 'LL_zmag':44}

actNames=[
    'sitting',  # 1
    'standing', # 2
    'lying on back',# 3
    'lying on right side', # 4
    'ascending stairs' , # 5
    'descending stairs', # 6
    'standing in an elevator still', # 7
    'moving around in an elevator', # 8
    'walking in a parking lot', # 9
    'walking on a treadmill with a speed of 4 km/h in flat', # 10
    'walking on a treadmill with a speed of 4 km/h in 15 deg inclined position', # 11
    'running on a treadmill with a speed of 8 km/h', # 12
    'exercising on a stepper', # 13
    'exercising on a cross trainer', # 14
    'cycling on an exercise bike in horizontal positions', # 15
    'cycling on an exercise bike in vertical positions', # 16
    'rowing', # 17
    'jumping', # 18
    'playing basketball' # 19
    ]
actNamesShort=[
    'sitting',  # 1
    'standing', # 2
    'lying.ba', # 3
    'lying.ri', # 4
    'asc.sta' , # 5
    'desc.sta', # 6
    'stand.elev', # 7
    'mov.elev', # 8
    'walk.park', # 9
    'walk.4.fl', # 10
    'walk.4.15', # 11
    'run.8', # 12
    'exer.step', # 13
    'exer.train', # 14
    'cycl.hor', # 15
    'cycl.ver', # 16
    'rowing', # 17
    'jumping', # 18
    'play.bb' # 19
    ]
ID=309709
s=ID%8+1
patients=[s]  # list of selected patients
activities=list(range(1,19)) #list of indexes of activities to plot
Num_activities=len(activities)
NAc=19 # total number of activities
actNamesSub=[actNamesShort[i-1] for i in activities] # short names of the selected activities
sensors=list(range(45)) # list of sensors
#sensors = [3,4,5,6,7,8,12,13,14,15,16,17,21,22,23,24,25,26,30,31,32,33,34,35,39,40,41,42,43,44] # no acc sensors

sensNamesSub=[sensNames[i] for i in sensors] # names of selected sensors
Nslices=12 # number of slices to plot
NtotSlices=60 #total number of slices
slices=list(range(1,Nslices+1))# first Nslices to plot
fs=25 # Hz, sampling frequency
samplesPerSlice=fs*5 # samples in each slice

cutoff = 1.5
fs = 25
order = 1
max_accuracy_te = max_accuracy_tr = 0
cutoff_range = np.arange(0.2, 1.5, 0.1)
sampling = 75

for nTrainSlices in range(13, 14):
    for n_smallest in range(19, 20):
        sensors=list(range(45))
###################### Evaluate feature importance #######################################
        slicesTrain = list(range(1,nTrainSlices+1)) 
        slicesTest = list(range(nTrainSlices+1, NtotSlices+1))
        N_tr= nTrainSlices * NAc * 125
        X_train = np.zeros([N_tr, len(sensors)])
        y_tr =np.zeros(N_tr)
        N_te = (NtotSlices - nTrainSlices) * NAc * 125
        X_test = np.zeros([N_te, len(sensors)])
        y_te =np.zeros(N_te)
        iter_tr = 0
        iter_te = 0
        for i in range(1, NAc+1):
            activities = [i]
            # Training Set
            x_tr=myFn.generateDF(filedir,sensNamesSub,sensors, patients,activities,slicesTrain)
            y_tr[iter_tr:len(x_tr)+iter_tr] = i - 1
            x_tr=x_tr.drop(columns=['activity'])
            x_tr = x_tr.values
            X_train[iter_tr:len(x_tr)+iter_tr, :] = x_tr
            iter_tr += len(x_tr)

        feat = myFn.featureImportanceVar(X_train, sensNames, n_smallest)
        sensors = myFn.mapSensors(feat, sensDic)
        ###################### Generate training and test set #####################################
        N_tr= nTrainSlices * NAc * 5
        X_train = np.zeros([N_tr, len(sensors)])
        y_tr =np.zeros(N_tr)
        N_te = (NtotSlices - nTrainSlices) * NAc * 5
        X_test = np.zeros([N_te, len(sensors)])
        y_te =np.zeros(N_te)
        iter_tr = 0
        iter_te = 0
        for i in range(1, NAc+1):
            activities = [i]
            # Training Set
            x_tr=myFn.generateDF(filedir,sensNamesSub,sensors, patients,activities,slicesTrain)
            x_tr = myFn.interpolation(x_tr)
            x_tr = myFn.butter_lowpass_filter(x_tr, cutoff, fs, order)
            x_tr = myFn.averageSampling(x_tr, sampling)
            y_tr[iter_tr:len(x_tr)+iter_tr] = i - 1
            x_tr=x_tr.drop(columns=['activity'])
            x_tr = x_tr.values
            X_train[iter_tr:len(x_tr)+iter_tr, :] = x_tr
            iter_tr += len(x_tr)
            # Test set
            x_te=myFn.generateDF(filedir,sensNamesSub,sensors, patients,activities,slicesTest)
            x_te=x_te.drop(columns=['activity'])
            #x_te = myFn.interpolation(x_te)
            x_te = myFn.butter_lowpass_filter(x_te, cutoff, fs, order)
            x_te = myFn.averageSampling(x_te, sampling)
            y_te[iter_te:len(x_te)+iter_te] = i - 1
            x_te = x_te.values
            X_test[iter_te:len(x_te)+iter_te, :] = x_te
            iter_te += len(x_te)

        """ X_train = pd.DataFrame(X_train)
        X_train['y'] = y_tr
        corr = X_train.corr()
        corr_y = abs(corr['y'])
        print(corr_y.sort_values(ascending=True)) """
        ########################Centroids evaluation #############################
        n_sensors = len(sensors)
        centroids=np.zeros((NAc,n_sensors))# centroids for all the activities
        stdpoints=np.zeros((NAc,n_sensors))# variance in cluster for each sensor
        # Evaluate centroids and std of sensor values
        for i in range(1,NAc+1):
            activities=[i]
            x=myFn.generateDF(filedir,sensNamesSub,sensors, patients,activities,slicesTrain)
            x=x.drop(columns=['activity'])
            x = myFn.interpolation(x)
            x = myFn.butter_lowpass_filter(x, cutoff, fs, order)   #0.8, 25, 2
            x = myFn.averageSampling(x, sampling)
            centroids[i-1,:]=x.mean().values
            stdpoints[i-1]=np.sqrt(x.var().values)

        ####################### K-Means ###################################
        kmeans = KMeans(n_clusters=NAc, init=centroids, n_init=1, max_iter=20)
        kmeans.fit(X_train)
        y_hat_tr = kmeans.labels_
        y_hat_te = kmeans.predict(X_test)
        # Evaluate accuracy
        accuracy_tr = accuracy_score(y_hat_tr, y_tr)
        accuracy_te = accuracy_score(y_hat_te, y_te)
        print("\nNumber of used sensors: ", n_smallest)
        print("Number of slices used for training", nTrainSlices)
        print(f"Order: {order}  Cutoff freq: {cutoff}")
        print("Accuracy on train: ", accuracy_tr)
        print("Accuracy on test: ", accuracy_te)
        if max_accuracy_te < accuracy_te:
            max_accuracy_te = accuracy_te
            max_accuracy_tr = accuracy_tr
            max_n_smallest = n_smallest
            max_n_trainSlices = nTrainSlices
            max_y_hat_tr = y_hat_tr
            max_y_hat_te = y_hat_te
            max_centroids = centroids
            max_stdpoint = stdpoints
            max_cutoff = cutoff
            max_order = order

print("\n##################### Best result #########################")
print("Number of used sensors: ", max_n_smallest)
print("Number of slices used for training", max_n_trainSlices)
print(f"Order: {max_order}  Cutoff freq: {max_cutoff}")
print("Accuracy on train: ", max_accuracy_tr)
print("Accuracy on test: ", max_accuracy_te)

print("Execution time:  ", time.time()-start_time)

conf_matr_tr = confusion_matrix(y_tr, max_y_hat_tr)
cmd = ConfusionMatrixDisplay(confusion_matrix=conf_matr_tr, display_labels = actNamesShort)
cmd.plot(xticks_rotation=90)
plt.show()
conf_matr_te = confusion_matrix(y_te, max_y_hat_te)
cmd = ConfusionMatrixDisplay(confusion_matrix=conf_matr_te, display_labels = actNamesShort)
cmd.plot(xticks_rotation=90)
plt.show()
#%% plot the measurements of each selected sensor for each of the activities
plotSensAct = False
if plotSensAct:
    for i in activities:
        activities=[i]
        x=myFn.generateDF(filedir,sensNamesSub,sensors, patients,activities,slices)
        x=x.drop(columns=['activity'])
        sensors=list(x.columns)
        data=x.values
        plt.figure(figsize=(6,6))
        time=np.arange(data.shape[0])/fs # set the time axis
        for k in range(len(sensors)):
            lines=plt.plot(time,data[:,k],'.',label=sensors[k],markersize=1)
            lines[0].set_color(cm(k//3*3/len(sensors)))
            lines[0].set_linestyle(line_styles[k%3])
        plt.legend()
        plt.grid()
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.title(actNames[i-1])
    #plt.show()

# Plotting centroids and std of sensor values
plotCentr = False
if plotCentr:
    plt.figure(figsize=(12,6))
    for i in range(1,NAc+1):
        activities=[i]
        x=myFn.generateDF(filedir,sensNamesSub,sensors, patients,activities,slices)
        x=x.drop(columns=['activity'])
        x = myFn.sampling(x, 60)
        centroids[i-1,:]=x.mean().values
        plt.subplot(1,2,1)
        lines = plt.plot(centroids[i-1,:],label=actNamesShort[i-1])
        lines[0].set_color(cm(i//3*3/NAc))
        lines[0].set_linestyle(line_styles[i%3])
        stdpoints[i-1]=np.sqrt(x.var().values)
        plt.subplot(1,2,2)
        lines = plt.plot(stdpoints[i-1,:],label=actNamesShort[i-1])
        lines[0].set_color(cm(i//3*3/NAc))
        lines[0].set_linestyle(line_styles[i%3])
    plt.subplot(1,2,1)
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Centroids using '+ str(len(sensors))+' sensors')
    plt.xticks(np.arange(x.shape[1]),list(x.columns),rotation=90)
    plt.subplot(1,2,2)
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('Standard deviation using '+str(len(sensors))+' sensors')
    plt.xticks(np.arange(x.shape[1]),list(x.columns),rotation=90)
    plt.tight_layout()
    plt.savefig(pathCharts + "CentroidandSTD.png")
    #plt.show()
#%% between centroids distance 
d=np.zeros((NAc,NAc))
for i in range(NAc):
    for j in range(NAc):
        d[i,j]=np.linalg.norm(max_centroids[i]-max_centroids[j])

plt.matshow(d)
plt.colorbar()
plt.xticks(np.arange(NAc),actNamesShort,rotation=90)
plt.yticks(np.arange(NAc),actNamesShort)
plt.savefig(pathCharts + "MatCentroidDistance.png")
plt.title('Between-centroids distance')

#%% compare minimum distance between two centroids and mean distance from a cluster point
# and its centroid
dd=d+np.eye(NAc)*1e6# remove zeros on the diagonal (distance of centroid from itself)
dmin=dd.min(axis=0)# find the minimum distance for each centroid
dpoints=np.sqrt(np.sum(max_stdpoint**2,axis=1))
plt.figure()
plt.plot(dmin,label='minimum centroid distance')
plt.plot(dpoints,label='mean distance from points to centroid')
plt.grid()
plt.xticks(np.arange(NAc),actNamesShort,rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(pathCharts + "centroidDistance.png")
# if the minimum distance is less than the mean distance, then some points of the cluster are closer 
# to another centroid
plt.show()
