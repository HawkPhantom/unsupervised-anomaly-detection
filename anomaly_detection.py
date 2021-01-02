# -*- coding: utf-8 -*-
##Imports
import pandas as pd
import subprocess
import sys
 
try:
    from adtk.data import validate_series
    from adtk.visualization import plot
    from adtk.detector import PersistAD
    from adtk.transformer import DoubleRollingAggregate
    from adtk.pipe import Pipeline
 
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "adtk"])
    from adtk.data import validate_series
    from adtk.visualization import plot
    from adtk.detector import PersistAD
    from adtk.transformer import DoubleRollingAggregate
    from adtk.pipe import Pipeline

from google.colab import files
uploaded = files.upload()

##PersistAD Detects Spikes (Extremely Abnormal Values)
##High Tolerance Model
persist_ad = PersistAD(agg='mean',side='both',c=6) ##Side Parameter Filters Positive and Negative Sided Anomalies
anomalies_1 = persist_ad.fit_detect(s_train,return_list=True)
plot(s_train, anomaly=anomalies_1, anomaly_color="red", anomaly_tag="marker")

for i in anomalies_1: 
    print(i)
print(len(anomalies_1))

##Pipeline Definition to Combine PersistAD Detector with Double Rolling Aggregate Transformer
##Low Tolerance Model
steps = [
    ("DoubleRolling", DoubleRollingAggregate(agg="median",window=(4,1))), ##Calcuates the Difference of Aggregated Metrics Between Two Sliding Windows.
    ("persist", PersistAD(agg="median",side="both",c=4)), #c is Factor Used to Determine the Bound of Normal Range Based on Historical Interquartile Range. Default Value is: 3.0.
]

pipeline = Pipeline(steps)

##Training Phase
anomalies_2 = pipeline.fit_detect(s_train,return_list=True) ##fit_detect Function Trains the Model and Detects Anomalies in Training Set
plot(s_train, anomaly=anomalies_2, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');

for i in anomalies_2: 
    print(i)

print(len(anomalies_2))

##Take difference between high tolerance and low tolerance modelds
def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
  
li1 = anomalies_1
li2 = anomalies_2 

##How li1 differs from li2
a = list(set(li1) - set(li2))

for i in a:
    print(i) 

##Combine the results of low tolerance and high tolerance models
for i in a:
    li2.append(i)


a = list(set(li1) - set(li2))

for i in a:
    print(i) 

print(len(a))

##Combined results of low and high tolerance models
for i in li2:
    print(i)
print(len(li2))

plot(s_train, anomaly=li2, ts_markersize=1, anomaly_markersize=2, anomaly_tag="marker", anomaly_color='red');