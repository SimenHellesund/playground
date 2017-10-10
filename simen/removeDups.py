###############
# Description #
###############

# Author: Simen Hellesund
# Contact: shellesu@cern.ch / simehe@uio.no
# Usage: python removeDups.py PATH_TO_INTPUT_FILE
# Descripion:
"""
Script for removing duplicate rows from input data.

h5py package needed to run script.
"""

################
# Dependencies #
################
import h5py # reading and writing .h5 files
import numpy as np # manipulating numpy arrays
import sys #using input arguments

########
# Body #
########
print "reading input file and collecting numpy array"

if len(sys.argv) != 2:
    raise Exception('Usage: python removeDups.py PATH_TO_INPUT_FILE')

fileName = sys.argv[1]
outputPath = fileName.split(".")[0] + "_noDups.h5"
inputFile = h5py.File(fileName, 'r')
tree = inputFile.keys()[0]
data = np.array(inputFile[tree])
print "Rows in original array: "+ str(len(data))


#Removing Duplicate rows
print "Removing duplicate rows"
uniques = np.unique(data,axis=0)
print "Rows after removing duplicate rows: " + str(len(uniques))


#storing file           
print "Writing new array to file %s" %outputPath
f = h5py.File(outputPath, "w")
f.create_dataset("transfer_data", data=uniques)
f.close()

print "Done!"
