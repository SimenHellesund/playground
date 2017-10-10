###############
# Description #
###############

# Author: Simen Hellesund (based on work done by James Catmore as well as scikit-learn documentaion)
# Contact: shellesu@cern.ch / simehe@uio.no
# Usage: pyton BDT.py 

"""
Using historical transfer data collected using the downloading script, train and test a boosted decision tree classifier.

Needs h5py and scikit-learn python packages installed to run.
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
from sklearn.utils import shuffle

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

# List of variables in the order they appear in the training samples
from varList_final import variables

# Declare output pdf
pdf_pages = PdfPages('outputBDT.pdf') 
np.set_printoptions(threshold=np.nan)

#########################################
# Read Input Files and Shape Input Data #
#########################################

print "Reading Input File"
filename = 'QualInput/output.h5'
inputFile = h5py.File(filename, 'r')
# List all groups
tree = inputFile.keys()[0]
# Get the data
data = np.array(inputFile[tree])

#read successes and failures only input files
inputSucc = h5py.File("QualInput/allVars_successes.h5","r")
inputFail = h5py.File("QualInput/allVars_failures.h5","r")
dataSucc = np.array(inputSucc[tree])
dataFail = np.array(inputFail[tree])

#pick only nEvents for training and testing. Could be useful if input very large and you only want to use a subset of the data to save time.

nEvents = min([len(dataSucc),len(dataFail)])#10000
dataSucc = shuffle(dataSucc,n_samples=nEvents)
dataFail = shuffle(dataFail,n_samples=nEvents)

# Split data into traing and testing sample of equal size. 
trainingSucc,testingSucc = np.array_split(dataSucc,2,axis=0)
trainingFail,testingFail =np.array_split(dataFail,2,axis=0)

#add success and failure samples together.
trainingData = np.concatenate((trainingSucc,trainingFail),axis=0)
testingData = np.concatenate((testingSucc,testingFail),axis=0)

# separate out "target". If a transfer was successful or not is stored as a value (0 for failure, 1 for success) in the last column of the input data.
trainingVars, trainingTarget = np.split(trainingData,[trainingData.shape[1]-1],axis=1)
testingVars, testingTarget = np.split(testingData,[testingData.shape[1]-1],axis=1)


## scale variables. Map all variables to values between 0 and 1. This is to prevent large numbers from dominating in the testing. 
print "Scaling Variables"
#min_max_scaler = preprocessing.MinMaxScaler()
#trainingVars = min_max_scaler.fit_transform(trainingVars)
#testingVars = min_max_scaler.transform(testingVars)

##############################################
# Create and fit an AdaBoosted decision tree #
##############################################

# This is where the magic happens!

print "Declaring Classifier"
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)

print "Training BDT" 
bdt.fit(trainingVars, np.ravel(trainingTarget))


#######################################################################################
# Calculate Rate of Correct and Incorrect Classification in Training and Testing Data #
#######################################################################################

pred_train = bdt.predict(trainingVars)
pred_test = bdt.predict(testingVars)
output_train = bdt.decision_function(trainingVars)
output_test = bdt.decision_function(testingVars)

train_SS = 0
train_SB = 0
train_BS = 0
train_BB = 0

for number,entry in enumerate(trainingTarget):
    if entry == 1:
        if pred_train[number] == 1:
            train_SS += 1
        elif pred_train[number] == 0:
            train_SB += 1
    elif entry == 0:
        if pred_train[number] == 1:
            train_BS += 1
        elif pred_train[number] == 0:
            train_BB += 1

test_SS = 0
test_SB = 0
test_BS = 0
test_BB = 0

for number,entry in enumerate(testingTarget):
    if entry == 1:
        if pred_test[number] == 1:
            test_SS += 1
        elif pred_test[number] == 0:
            test_SB += 1
    elif entry == 0:
        if pred_test[number] == 1:
            test_BS += 1
        elif pred_test[number] == 0:
            test_BB += 1

print "Perfomance and Testing"
print "test_SS: ",test_SS
print "test_SB: ",test_SB
print "test_BS: ",test_BS
print "test_BB: ",test_BB
print ""
print "Training sample...."
print "  Success identified as success (%)    : ",100*float(train_SS)/(train_SS + train_SB)
print "  Success identified as failure (%)    : ",100*float(train_SB)/(train_SS + train_SB)
print "  Failure identified as success (%)    : ",100*float(train_BS)/(train_BS + train_BB)
print "  Failure identified as failure (%)    : ",100*float(train_BB)/(train_BS + train_BB)
print ""
print "Testing sample...."
print "  Success identified as success (%)    : ",100*float(test_SS)/(test_SS + test_SB)
print "  Success identified as failure (%)    : ",100*float(test_SB)/(test_SS + test_SB)
print "  Failure identified as success (%)    : ",100*float(test_BS)/(test_BS + test_BB)
print "  Failure identified as failure (%)    : ",100*float(test_BB)/(test_BS + test_BB)

##################################################
# Plot Classifier Score For Training and Testing #
##################################################

print "Plotting Scores"

figA, axsA = plt.subplots(2, 1)
figA.subplots_adjust(hspace=.5)
ax1, ax2 = axsA.ravel()
for ax in ax1, ax2:
    ax.set_ylabel("transfers")
    ax.set_xlabel("Classifier Score")
bins = np.linspace(-.25, .25, 250)
ax1.hist(output_train[(trainingTarget==0.0).reshape(len(trainingTarget),)], bins, facecolor='red', alpha=0.4, histtype='stepfilled',label="Failures")
ax1.hist(output_train[(trainingTarget==1.0).reshape(len(trainingTarget),)], bins, facecolor='green', alpha=0.4, histtype='stepfilled',label="Successes")
ax1.legend(loc='upper left')
ax2.hist(output_test[(testingTarget==0.0).reshape(len(testingTarget),)], bins, facecolor='red', alpha=0.4, histtype='stepfilled',label="Failures")
ax2.hist(output_test[(testingTarget==1.0).reshape(len(testingTarget),)], bins, facecolor='green', alpha=0.4, histtype='stepfilled',label="Successes")
ax2.legend(loc='upper left')
pdf_pages.savefig(figA)

##########################
# Plot Performance Curve #
##########################

print "Plotting Performance Curve"

fpr, tpr, thresholds = roc_curve(testingTarget, output_test, pos_label=1)
auc = roc_auc_score(testingTarget, output_test)
print "Area under ROC = ",auc
figB, axB1 = plt.subplots()
#axB1,axB2 = axsB.ravel()
axB1.plot(fpr, tpr, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Signal Rate')
axB1.set_ylabel('True Signal Rate')
axB1.text(0.4,0.2,"AUC = %.4f" % auc,fontsize=15)
pdf_pages.savefig(figB)

############################ 
# Plot Variable Importance # 
############################

print "Plotting Variable Importance"
# Variable importances
y_pos = np.arange(len(variables))
figC, axC1 = plt.subplots(1,1)
axC1.barh(y_pos, 100.0*bdt.feature_importances_, align='center', alpha=0.4)
axC1.set_ylim([0,len(variables)])
axC1.set_yticks(y_pos)
axC1.set_yticklabels(variables,fontsize=7)
axC1.set_xlabel('Relative importance, %')
axC1.set_title("Estimated variable importance using outputs (BDT)")
pdf_pages.savefig(figC)

################################################
# Plot Variables vs. Scores for Testing Sample #
################################################

#"undo" feature scaling for plotting
#testingVars = min_max_scaler.inverse_transform(testingVars)

# Closer Look at Misidentified Transfers #
corrSucc = testingVars[(testingTarget==1.0).reshape(len(testingTarget),) & (output_test>=0).reshape(len(output_test),)]#correctly identified successes (score > 0)
misSucc = testingVars[(testingTarget==1.0).reshape(len(testingTarget),) & (output_test<=0).reshape(len(output_test),)]#misidentified successes. Successful transfer the classifier mistook for failures (score < 0)
corrFail = testingVars[(testingTarget==0.0).reshape(len(testingTarget),) & (output_test<=0).reshape(len(output_test),)]#correctly identified failures
misFail = testingVars[(testingTarget==0.0).reshape(len(testingTarget),) & (output_test>=0).reshape(len(output_test),)]#misidentified failures

protocol_labels = ["srm","gsiftp","davs","root"]
retried_labels = ["no","yes"]

#scatterplots
for column,variable in enumerate(variables):
    #do successes and failures in different colours
    figD, axD = plt.subplots(1,1)
    axD.scatter(testingVars[(testingTarget==0.0).reshape(len(testingTarget),),column],output_test[(testingTarget==0.0).reshape(len(testingTarget),)], alpha=0.4,color="red",label="Failures")
    axD.scatter(testingVars[(testingTarget==0.0).reshape(len(testingTarget),),column],output_test[(testingTarget==1.0).reshape(len(testingTarget),)], alpha=0.4,color="green",label="Successes")
    axD.legend(loc='upper right')
    axD.set_ylabel("Classifier Score")
    axD.set_xlabel(variable)
    axD.set_title("Classifier Score vs %s (Testing Sample)" %variable)
    pdf_pages.savefig(figD)

#histograms
for column,variable in enumerate(variables):

    bins = np.linspace(0, np.amax(testingVars[:,column],axis=0), 25)
 
    #do successes and failures in different colours
    figD, axD = plt.subplots(1,1)
    axD.hist(testingVars[(testingTarget==0.0).reshape(len(testingTarget),),column],bins, alpha=0.4,facecolor="red",label="Failures",histtype='bar')
    axD.hist(testingVars[(testingTarget==1.0).reshape(len(testingTarget),),column],bins, alpha=0.4,facecolor="green",label="Successes",histtype='bar')
    axD.legend(loc='upper right')
    axD.set_ylabel("Transfers")
    axD.set_xlabel(variable)
    if variable == "protocol":
        axD.set_xticks([0,1,2,3])
        axD.set_xticklabels(protocol_labels)
    if variable == "retried":
        axD.set_xticks([0,1])
        axD.set_xticklabels(retried_labels)    
    axD.set_title("%s Distribution for Successes and Failures" %variable.capitalize())
    pdf_pages.savefig(figD)

    #make histograms comparing mis- and correctly identified successful transfers
    figE, axE = plt.subplots(1,1)
    axE.hist(misFail[:,column],bins,alpha=0.4,facecolor="orange",label='Misidentified',histtype='stepfilled',normed=True)
    axE.hist(corrFail[:,column],bins, alpha=0.4,facecolor="red",label='Correcly Identified',histtype='stepfilled',normed=True)
    axE.legend(loc="upper right")
    axE.set_ylabel("Arbitrary (norm.)")
    axE.set_xlabel(variable)
    if variable == "protocol":
        axD.set_xticks([0,1,2,3])
        axD.set_xticklabels(protocol_labels)
    if variable == "retried":
        axD.set_xticks([0,1])
        axD.set_xticklabels(retried_labels)    
    axE.set_title("%s Distribution for mis- and Correctly Identified Failed Transfers" %variable.capitalize())
    pdf_pages.savefig(figE)

    #make histograms comparing mis- and correctly identified successful transfers
    figE, axE = plt.subplots(1,1)
    axE.hist(misSucc[:,column],bins,alpha=0.4,facecolor="blue",label='Misidentified',histtype='stepfilled',normed=True)
    axE.hist(corrSucc[:,column],bins, alpha=0.4,facecolor="green",label='Correcly Identified',histtype='stepfilled',normed=True)
    axE.legend(loc="upper right")
    axE.set_ylabel("Arbitrary (norm.)")
    axE.set_xlabel(variable)
    if variable == "protocol":
        axD.set_xticks([0,1,2,3])
        axD.set_xticklabels(protocol_labels)
    if variable == "retried":
        axD.set_xticks([0,1])
        axD.set_xticklabels(retried_labels)    
    axE.set_title("%s Distribution for mis- and Correctly Identified Successful Transfers" %variable.capitalize())
    pdf_pages.savefig(figE)





#Draw Plots
#plt.show()

#Save pdf
pdf_pages.close()

print "Done!"
