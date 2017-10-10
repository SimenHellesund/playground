###############
# Description #
###############

# Author: Simen Hellesund
# Contact: shellesu@cern.ch / simehe@uio.no
# Usage: pyton trainingSize.py

#About:
"""

"""


################
# Dependencies #
################

import h5py #reading input file

import numpy as np
import matplotlib.pyplot as plt

# Machine Learning Deps:
# Helpers and tools:
from sklearn import preprocessing
from sklearn.utils import resample

# For the actual classifier:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# For testing and diagnosing
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# Write plots to pdf
from matplotlib.backends.backend_pdf import PdfPages

# Declare output pdf
pdf_pages = PdfPages('output_trainingSize.pdf') 
np.set_printoptions(threshold=np.nan)


####################
# Read Input Files #
####################

print "Reading Input Files"

#read successes and failures only input files
inputSucc = h5py.File("QualInput/newSept_successes.h5","r")
inputFail = h5py.File("QualInput/newSept_failures.h5","r")
tree = inputSucc.keys()[0]
dataSucc = np.array(inputSucc[tree])
dataFail = np.array(inputFail[tree])


####################
# Define Variables #
####################

verbose = True

sizes = []
AUC = []

trainingSize = 5000
testingSize = 5000
updateSize = 1000

########################################
# Loop Over Different Size of Training #
########################################

while trainingSize <= len(dataSucc) - testingSize:

    if verbose: print "Now training on %i variables" %trainingSize
    
    #Shape data
    dataSuccTemp = resample(dataSucc,n_samples=(trainingSize+testingSize))
    dataFailTemp = resample(dataFail,n_samples=(trainingSize+testingSize))

    # Split data into traing and testing sample. 
    trainingSuccTemp = dataSuccTemp[:trainingSize]
    testingSuccTemp = dataSuccTemp[-testingSize:]
    trainingFailTemp = dataFailTemp[:trainingSize]
    testingFailTemp = dataFailTemp[-testingSize:]

    print len(trainingSuccTemp)
    print len(testingSuccTemp)

    #add success and failure samples together.
    trainingDataTemp = np.concatenate((trainingSuccTemp,trainingFailTemp),axis=0)
    testingDataTemp = np.concatenate((testingSuccTemp,testingFailTemp),axis=0)

     # separate out "target". If a transfer was successful or not is stored as a value (0 for failure, 1 for success) in the last column of the input data.
    trainingVarsTemp, trainingTargetTemp = np.split(trainingDataTemp,[trainingDataTemp.shape[1]-1],axis=1)
    testingVarsTemp, testingTargetTemp = np.split(testingDataTemp,[testingDataTemp.shape[1]-1],axis=1)

## scale variables. Map all variables to values between 0 and 1. This is to prevent large numbers from dominating in the testing. 
    min_max_scaler = preprocessing.MinMaxScaler()
    trainingVarsTemp = min_max_scaler.fit_transform(trainingVarsTemp)
    testingVarsTemp = min_max_scaler.transform(testingVarsTemp)

    #build and train bdt
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)
    bdt.fit(trainingVarsTemp, np.ravel(trainingTargetTemp))

    #bdt score for testing sample
    output_test = bdt.decision_function(testingVarsTemp)

    #calculate area under curve
    auc = roc_auc_score(testingTargetTemp, output_test)
    if verbose: print "Area under ROC = ", auc
    
    #append AUC and size to lists for plotting after loop
    sizes.append(trainingSize)
    AUC.append(auc)

    #update Size
    trainingSize += updateSize



#plotting
figA, ax = plt.subplots(1, 1)
ax.plot(sizes, AUC)
ax.set_ylabel("AUC")
ax.set_xlabel("Training Sample Size")
pdf_pages.savefig(figA)
plt.show()

#Save pdf
pdf_pages.close()

print "Done!"
