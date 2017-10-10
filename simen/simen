##### script for 


###############
# Description #
###############

#Author: Simen Hellesund
#Contact: shellesu@cern.ch / simehe@uio.no
#Usage: python merge.py path_to_h5_file1 path_to_h5_file2 name_of_output_file

"""
Script to merge two numpy array stored in two separate .h5 file into one array, stored in a new. h5 file.

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

print "reading input files and collecting numpy arrays"

if len(sys.argv) != 4:
    raise Exception('Usage: python merge.py path_to_h5_file1 path_to_h5_file2 name_of_output_file')

fileName1 = sys.argv[1]
fileName2 = sys.argv[2]
outName = sys.argv[3]
outputPath = "QualInput/" + outName + ".h5"

inputFile1 = h5py.File(fileName1, 'r')
inputFile2 = h5py.File(fileName2, 'r')
tree = inputFile1.keys()[0]
data1 = np.array(inputFile1[tree])
data2 = np.array(inputFile2[tree])

#concatenating the two input arrays
print "Merging Arrays"
outputData = np.concatenate((data1,data2),axis=0)


#storing file           
print "Writing new array to file %s" %outputPath
f = h5py.File(outputPath, "w")
f.create_dataset("transfer_data", data=outputData)
f.close()

print "Done!"
