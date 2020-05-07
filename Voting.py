import pandas as pd
import numpy as np
import os
import numpy as np

from sklearn.metrics import roc_curve, auc

Ture_Label = np.loadtxt(os.path.join(os.path.abspath('.') + '/data/phenotypes', "results_of_single atlas.csv"), delimiter=",",usecols=(1,),skiprows=0,dtype='int')

cc200_predictlabel = np.loadtxt(os.path.join(os.path.abspath('.') + '/data/phenotypes', "results_of_single atlas.csv"), delimiter=",",usecols=(2,),skiprows=0,dtype='int')

aal_predictlabel = np.loadtxt(os.path.join(os.path.abspath('.') + '/data/phenotypes', "results_of_single atlas.csv"), delimiter=",",usecols=(3,),skiprows=0,dtype='int')

dosenbach160_predictlabel = np.loadtxt(os.path.join(os.path.abspath('.') + '/data/phenotypes', "results_of_single atlas.csv"), delimiter=",",usecols=(4,),skiprows=0,dtype='int')


predictlabel1 = cc200_predictlabel+aal_predictlabel+dosenbach160_predictlabel

y_score_SDA = np.loadtxt(os.path.join(os.path.abspath('.') + '/data/phenotypes', "results_of_single atlas.csv"), delimiter=",",usecols=(6,),skiprows=0,dtype='float')
y_test_SDA = np.loadtxt(os.path.join(os.path.abspath('.') + '/data/phenotypes', "results_of_single atlas.csv"), delimiter=",",usecols=(1,),skiprows=0,dtype='int')
fpr_SDA, tpr_SDA, threshold_SDA = roc_curve(y_test_SDA, y_score_SDA)
roc_auc_SDA = auc(fpr_SDA, tpr_SDA)
# print(predictlabel1[0])
# print(predictlabel1[1])
# print(predictlabel1[2])


for i in range(949):
    if(predictlabel1[i]==1):
        predictlabel1[i]=0

for i in range(949):
    if(predictlabel1[i]==2):
        predictlabel1[i]=1

for i in range(949):
    if(predictlabel1[i]==3):
        predictlabel1[i]=1

count=0
for i in range(949):
    if(Ture_Label[i]==predictlabel1[i]):
        count=count+1

TPcount=0
for i in range(949):
    if(Ture_Label[i]==1 and predictlabel1[i]==1):
        TPcount=TPcount+1

TNcount=0
for i in range(949):
    if(Ture_Label[i]==0 and predictlabel1[i]==0):
        TNcount=TNcount+1
ACC = float(count)/ float(949)
SEN = float(TPcount)/float(530)
SPE = float(TNcount)/ float(419)


print "ACC %.4f" % ACC
print "SEN %.4f" % SEN
print "SPE %.4f" % SPE
print "AUC %.4f" %roc_auc_SDA
