
###############
# Description #
###############

# Author: Simen Hellesund (based on work done by James Catmore as well as scikit-learn documentaion)
# Contact: shellesu@cern.ch / simehe@uio.no
# Usage: pyton depth.py 

#

################
# Dependencies #
################

import h5py #reading input file

import numpy as np
import matplotlib.pyplot as plt

# Machine Learning Deps:
# Helpers and tools:
from sklearn import preprocessing
from sklearn.utils import shuffle

# For the actual classifier:
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# For testing and diagnosing
from sklearn import metrics
from sklearn.metrics import zero_one_loss

from scipy.stats import mode

# Write plots to pdf
from matplotlib.backends.backend_pdf import PdfPages

# Declare output pdf
pdf_pages = PdfPages('outputDepth.pdf') 
np.set_printoptions(threshold=np.nan)


#########################################
# Read Input Files and Shape Input Data #
#########################################

print "Reading Input Files"
#read successes and failures only input files
inputNameSucc = "big_sept_noDups_successes.h5"
inputNameFail = "big_sept_noDups_failures.h5"
inputSucc = h5py.File("QualInput/" + inputNameSucc,"r") 
inputFail = h5py.File("QualInput/" + inputNameFail,"r")
tree = inputSucc.keys()[0]
dataSucc = np.array(inputSucc[tree])
dataFail = np.array(inputFail[tree])

balancedTraining = True
balancedTesting = True

nEvents = 10000#min([len(dataSucc),len(dataFail)])
dataSucc = shuffle(dataSucc,n_samples=nEvents)
dataFail = shuffle(dataFail,n_samples=nEvents)

# Split data into traing and testing sample of equal size. 
trainingSucc,testingSucc = np.array_split(dataSucc,2,axis=0)
trainingFail,testingFail = np.array_split(dataFail,2,axis=0)

#add success and failure samples together.
trainingData = np.concatenate((trainingSucc,trainingFail),axis=0)
testingData = np.concatenate((testingSucc,testingFail),axis=0)

# separate out "target". If a transfer was successful or not is stored as a value (0 for failure, 1 for success) in the last column of the input data.
trainingVars, trainingTarget = np.split(trainingData,[trainingData.shape[1]-1],axis=1)
testingVars, testingTarget = np.split(testingData,[testingData.shape[1]-1],axis=1)

trees = 2000

textstr = "Size of training sample: %s \nSize of testing sample: %s \nInput file successes: %s \nInput File failures: %s \nBalanced samples training: %s \nBalanced samples testing: %s" %(len(trainingData),len(testingData),inputNameSucc,inputNameFail,balancedTraining,balancedTesting)
fig, ax = plt.subplots(1,1)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')
ax.set_title("Information")
ax.axis('off')
pdf_pages.savefig(fig)


#loop over max depth
for depth in list(range(6)):
    depth += 1 #cant have zero depth

    print "Declaring Classifier with Depth %s" %depth
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),n_estimators=trees)

#print "Training BDT" 
    bdt.fit(trainingVars, np.ravel(trainingTarget))


#pred_train = bdt.predict(trainingVars)
#pred_test = bdt.predict(testingVars)
#output_train = bdt.decision_function(trainingVars)
#output_test = bdt.decision_function(testingVars)

    bdt_err_train = np.zeros((trees,))
    for i, y_pred in enumerate(bdt.staged_predict(trainingVars)):
        bdt_err_train[i] = zero_one_loss(y_pred, trainingTarget)

    bdt_err_test = np.zeros((trees,))
    for i, y_pred in enumerate(bdt.staged_predict(testingVars)):
        bdt_err_test[i] = zero_one_loss(y_pred, testingTarget)

    figF, axF = plt.subplots(1,1)
    axF.plot(np.arange(trees) + 1, bdt_err_train,
             label='Training Error',
             color='black')
    axF.plot(np.arange(trees) + 1, bdt_err_test,
             label='Testing Error',
             color='blue')
    commonVal = mode(bdt_err_test)[0]
#    print commonVal
    axF.plot((0,trees), (commonVal,commonVal), 'r-',label="mode: %s"%commonVal)
    axF.set_ylabel("Error")
    axF.set_xlabel("Number of Trees (n_estimators)")
    axF.legend(loc='upper right')
    axF.set_title("Training and Testing Error, max_depth: %s" %depth)
    pdf_pages.savefig(figF)



pdf_pages.close()

print "Done!"
