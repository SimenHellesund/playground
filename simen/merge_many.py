###############
# Description #
###############

# Author: Simen Hellesund
# Contact: shellesu@cern.ch / simehe@uio.no
# Usage: python merge_many.py name_of_output_file path_to_input_1 path_to_intput_2 ... path to input_N
# Description:
"""
Script to merge any number numpy arrays stored in individual .h5 file into one array, stored in a new. h5 file.

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

outName = sys.argv[1]
outputPath = "QualInput/" + outName + ".h5"

#collect first input file and get its numpy array
fileName = sys.argv[2]
inputFile = h5py.File(fileName, 'r')
tree = inputFile.keys()[0]
data = np.array(inputFile[tree])
print "appending file " + fileName + " to total array"

#loop over the rest of the input files and append each one to the first array
for entry in sys.argv[3:]:
    loopName = entry
    loopFile = h5py.File(loopName, 'r')
    loopData = np.array(loopFile[tree])
    
    #concatenate the loop array with the total array
    print "appending file " + loopName + " to total array"
    data = np.concatenate((data,loopData),axis=0)
    
        #debugging
    print(len(data))


#write total array to file
print "Writing new array to file %s" %outputPath
f = h5py.File(outputPath, "w")
f.create_dataset("transfer_data", data=data)
f.close()

print "Done!"
