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
inputNameSucc = "sept_all_noDups_successes.h5"
inputNameFail = "sept_all_noDups_failures.h5"
inputSucc = h5py.File("QualInput/" + inputNameSucc,"r")
inputFail = h5py.File("QualInput/" + inputNameFail,"r")
tree = inputSucc.keys()[0]
dataSucc = np.array(inputSucc[tree])
dataFail = np.array(inputFail[tree])

balancedTraining = True
balancedTesting = True

####################
# Define Variables #
####################

verbose = True

sizes = []
AUC = []

trees = 500
depth = 3

########################################
# Loop Over Different Size of Training #
########################################

loopSizes = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000,300000]

testingSize = 5000

#while trainingSize <= len(dataFail) - testingSize:
for trainingSize in loopSizes:

    if verbose: print "Now training on %i transfers" %trainingSize

    #Shape data
    dataSuccTemp = resample(dataSucc,n_samples=int(trainingSize/2)+int(testingSize/2))
    dataFailTemp = resample(dataFail,n_samples=int(trainingSize/2)+int(testingSize/2))

    # Split data into traing and testing sample. 
    trainingSuccTemp = dataSuccTemp[:int(trainingSize/2)]
    testingSuccTemp = dataSuccTemp[-int(testingSize/2):]
    trainingFailTemp = dataFailTemp[:int(trainingSize/2)]
    testingFailTemp = dataFailTemp[-int(testingSize/2):]

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
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),n_estimators=trees)
    bdt.fit(trainingVarsTemp, np.ravel(trainingTargetTemp))

    #bdt score for testing sample
    output_test = bdt.decision_function(testingVarsTemp)

    #calculate area under curve
    auc = roc_auc_score(testingTargetTemp, output_test)
    if verbose: print "Area under ROC = ", auc
    
    #append AUC and size to lists for plotting after loop
    sizes.append(trainingSize)
    AUC.append(auc)

    #update Size (if while loop)
#    trainingSize += updateSize


######################################
# Information Sheet for the Plot PDF #
######################################

textstr = "Size of testing sample: %s \nInput file successes: %s \nInput File failures: %s \nBalanced sam\
ples training: %s \nBalanced samples testing: %s \nBDT depth: %s \nBDT n_estimators: %s" %(testingSize,inputNameSucc,inputNameFail,balancedTraining,balancedTesting,depth,trees)
fig, ax1 = plt.subplots(1,1)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top')
ax1.set_title("Information")
ax1.axis('off')
pdf_pages.savefig(fig)

#plotting
figA, ax = plt.subplots(1, 1)
ax.semilogx(sizes, AUC)
ax.set_ylabel("AUC")
ax.set_ylim([0,1])
ax.set_xlabel("Training Sample Size")
pdf_pages.savefig(figA)

#show plots on screen
#plt.show()

#Save pdf
pdf_pages.close()

print "Done!"
