###############
# Description #
###############

#Author: Simen Hellesund
#Contact: shellesu@cern.ch / simehe@uio.no
#Usage: python split.py path_to_h5_file name_of_output_file

"""
Script to split training data into separate files according to success and failure.  The number of successes is also limited to the same number as failures. This is done in order for the multivariate classifier to test on a balanced sample in success and failure. Before doing this there are a lot more successes than failures in the training data (an order of magnitude or so). 

h5py package needed to run script
"""


################
# Dependencies #
################
import h5py # reading and writing .h5 files
import numpy as np # manipulating numpy arrays
import sys #using input arguments

###########################################
# Read input file and collect numpy array #
###########################################

print "reading input file and collecting numpy array"

if len(sys.argv) != 3:
    raise Exception('Usage: python split.py path_to_h5_file name_of_output_file')

filename = sys.argv[1]
outname = sys.argv[2]
inputFile = h5py.File(filename, 'r')
tree = inputFile.keys()[0]
data = list(inputFile[tree])

######################################################### 
# Split data into two lists based on success or failure #
#########################################################

print "Splitting data into successes and failures"

successes = []
failures = []

#failures
for array in data:
    if array[-1] == 0:
        failures.append(array)
    elif array[-1] == 1:
        successes.append(array)

new_successes = []

#successes. Fill only as many as there are failures. Want balanced samples to train classifier.
counter = 0
while counter<=len(failures):
    new_successes.append(successes[counter])
    counter+=1

#################################################
# Convert back into numpy array and write files #
#################################################

print "Making new arrays"

successes = np.array(new_successes)
failures = np.array(failures)

print "writing output files"

outputSucc = "QualInput/" + outname + "_successes.h5"
outputFail = "QualInput/" + outname + "_failures.h5"

#storing file           
print "Writing successes to file %s" %outputSucc
f = h5py.File(outputSucc, "w")
f.create_dataset("transfer_data", data=successes)
f.close()

#storing file                                   
print "Writing failures to file %s" %outputFail                                                                                       
f = h5py.File(outputFail, "w")
f.create_dataset("transfer_data", data=failures)
f.close()

print "Done!"


