## Description:






################
# Dependencies #
################

#reading input file
import h5py

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model

#machine learning
from sklearn import preprocessing

from sklearn.utils import resample

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

#list of variables, in the right order
from varList import variables

#writing plots to pdf
from matplotlib.backends.backend_pdf import PdfPages
pdf_pages = PdfPages('outputOLD.pdf') 
np.set_printoptions(threshold=np.nan)

## Read input files
print "reading input file"
filename = 'output.h5'
inputFile = h5py.File(filename, 'r')
# List all groups
a_group_key = inputFile.keys()[0]
# Get the data
data = np.array(inputFile[a_group_key])

#read successes and failures only input file
inputSucc = h5py.File("output_successes.h5","r")
inputFail = h5py.File("output_failures.h5","r")
dataSucc = np.array(inputSucc[a_group_key])
dataFail = np.array(inputFail[a_group_key])

nEvents = 10000
#draw nEvents random events from full input samples                                                                                     
dataSucc = resample(dataSucc,n_samples=nEvents)
dataFail = resample(dataFail,n_samples=nEvents)

trainingSucc = dataSucc[:len(dataSucc) - int(len(dataSucc)/2)] # first half
testingSucc = dataSucc[len(dataSucc) - int(len(dataSucc)/2):] # second half
trainingFail = dataFail[:len(dataFail) - int(len(dataFail)/2)] # first half
testingFail = dataFail[len(dataSucc) - int(len(dataSucc)/2):] # first half

trainingData = np.concatenate((trainingSucc,trainingFail),axis=0)
testingData = np.concatenate((testingSucc,testingFail),axis=0)

# separate out "target"
trainingVars, trainingTarget = np.split(trainingData,[trainingData.shape[1]-1],axis=1)
testingVars, testingTarget = np.split(testingData,[testingData.shape[1]-1],axis=1)

## scale variables
print "Scaling Variables"
min_max_scaler = preprocessing.MinMaxScaler()
trainingVars = min_max_scaler.fit_transform(trainingVars)
testingVars = min_max_scaler.transform(testingVars)

# Create and fit an AdaBoosted decision tree
print "Building and training neural network"
model = Sequential()
from keras.layers import Dense, Activation
model.add(Dense(len(variables), input_dim=len(variables)))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(optimizer="rmsprop",#"sgd",
              loss="mean_squared_error",
              metrics=["mean_squared_error"])
model.fit(trainingVars,trainingTarget,epochs=100,batch_size=100)




#Compare to testing sample
pred_train = model.predict_classes(trainingVars)
pred_test = model.predict_classes(testingVars)
output_train = model.predict_proba(trainingVars)
output_test = model.predict_proba(testingVars)


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

#plotting

#print output_train.shape
#print trainingTarget.shape


figA, axsA = plt.subplots(2, 1)
figA.subplots_adjust(hspace=.5)
ax1, ax2 = axsA.ravel()
for ax in ax1, ax2:
    ax.set_ylabel("transfers")
    ax.set_xlabel("Classifier Score")
bins = np.linspace(-1.5, 1.5, 250)
ax1.hist(output_train[(trainingTarget==0.0).reshape(len(trainingTarget),)], bins, facecolor='blue', alpha=0.4, histtype='stepfilled',label="Failures")
ax1.hist(output_train[(trainingTarget==1.0).reshape(len(trainingTarget),)], bins, facecolor='red', alpha=0.4, histtype='stepfilled',label="Successes")
ax1.legend(loc='upper left')
ax2.hist(output_test[(testingTarget==0.0).reshape(len(testingTarget),)], bins, facecolor='green', alpha=0.4, histtype='stepfilled',label="Failures")
ax2.hist(output_test[(testingTarget==1.0).reshape(len(testingTarget),)], bins, facecolor='red', alpha=0.4, histtype='stepfilled',label="Successes")
ax2.legend(loc='upper left')
pdf_pages.savefig(figA)

# Plotting - performance curves
# ROC
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

# Assess variable importance using weights method
weights = np.array([])
for layer in model.layers:
    if layer.name =="dense_1":
        weights = layer.get_weights()[0]
# Ecol. Modelling 160 (2003) 249-264
sumWeights = np.sum(np.absolute(weights),axis=0)
Q = np.absolute(weights)/sumWeights
R = 100.0 * np.sum(Q,axis=1) / np.sum(np.sum(Q,axis=0))
y_pos = np.arange(len(variables))
figC, axC = plt.subplots()
axC.barh(y_pos, R, align='center', alpha=0.4)
axC.set_ylim([0,len(variables)])
axC.set_yticks(y_pos)
axC.set_yticklabels(variables,fontsize=10)
axC.set_xlabel('Relative importance, %')
axC.set_title('Estimated variable importance using input-hidden weights (ecol.model)')
pdf_pages.savefig(figC)










#Draw Plots
plt.show()



#Save pdf
pdf_pages.close()


print "Done!"
