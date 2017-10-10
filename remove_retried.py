###############
# Description #
###############

# Author: Simen Hellesund
# Contact: shellesu@cern.ch / simehe@uio.no
# Usage: python remove_retried.py PATH_TO_INTPUT_FILE
# Descripion:
"""
Script for removing the "retried" column from input data. Although this variable seems to have show good separation between successes and failures, it will not be a known quantity at the time of submitting a transfer.

h5py package needed to run script.
"""

################
# Dependencies #
################
import h5py # reading and writing .h5 files
import numpy as np # manipulating numpy arrays
import sys #using input arguments
from varList_final import variables

print "reading input file and collecting numpy array"

if len(sys.argv) != 2:
    raise Exception('Usage: python removeDups.py PATH_TO_INPUT_FILE')

fileName = sys.argv[1]
outputPath = fileName.split(".")[0] + "_noRetried.h5"
inputFile = h5py.File(fileName, 'r')
tree = inputFile.keys()[0]
data = np.array(inputFile[tree])

#find the position of the "retired" column in the input array
index = [col for  col, val in enumerate(variables) if val == "retried"][0]

#remove column "index" from input array
new_data = np.delete(data, index, axis=1)

#storing file           
print "Writing new array to file %s" %outputPath
f = h5py.File(outputPath, "w")
f.create_dataset("transfer_data", data=new_data)
f.close()

print "Done!"
