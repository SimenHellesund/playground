################
# Description #
###############

# Author: Simen Hellesund
# Contanct: shellesu@cern.ch / simehe@uio.no
# Usage: python downloading_final.py

"""
The purpose of this script is to download data on Rucio file transfers and grid site metrics using Elasticsearch, and to store this data in the form of numpy arrays. These numpy arrays can then be used to train a scikit-learn multivariate classifier which can (hopefully) learn to determine whether or not a file transfer will fail.

This version: Add end time and interval of search period to enable batching. Running over too large of a period in one single go causes memory issues.

The file CERN-bundle.pem need to be in the same directory as this file. This file contains the certificate to connect to the CERN Kibana.

Need h5py and elasticsearch python packages to run.
"""

################
# Dependencies #
################

#for querying kibana:                                                                                   
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

#for writing .h5 files:
import h5py

#for reading jsons:
import json
import requests

#misc:
import numpy as np
import time
import sys #using input arguments 

#############
# Variables #
#############

#Setting this to True turns on a lot of extra output, for debugging purposes
verbose = False#True

#time the execution of the script
start_time = time.time()

#Input variables
if len(sys.argv) != 3:
    raise Exception('Usage: python downloading_batching.py end_time interval_length_in_hours')

now = int(sys.argv[1])
hours = int(sys.argv[2])

interval = hours*60*60*1000 # converts to milliseconds, which is what the Elasticsearh query uses                                      
past = now - interval

#Rucio transfers are added into Elasticsearch with a delay from when the transfer actually took place. ddm_metric data does not have such a delay. Add an ofset in the ddm Elasticsearch query to take this into account. Query is extended further into the past than the query for transfers.
metrics_offset = 60*60*1000
metrics_past = past - metrics_offset

#These are used to convert time stamps in date and hour form to epoch time.
pattern = '%Y-%m-%d %H:%M:%S'
pattern_ddm = '%Y-%m-%dT%H:%M:%S'

#name of the Elasticsearch indices where the data is pulled from.
transfer_index = "atlas_rucio-events-*"
ddm_index = "atlas_ddm-metrics-20*"
chicago_index = "network_weather-*"

#url to the specific ES instances. 
es = Elasticsearch(['es-atlas.cern.ch:9203'],timeout=60, use_ssl=True, verify_certs=True, ca_certs=\
'CERN-bundle.pem', http_auth='roatlas:la2Dbd5rtn3df!sx') #CERN instance
es_chicago = Elasticsearch(['atlas-kibana.mwt2.org:9200'],timeout=60) # Chicago instance


#mapping of site name convention in from rucio transfers to ddm_metrics
with open('mapping-rse-site.json') as siteNameJSON:    
    siteNameMap = json.load(siteNameJSON)

#make list of site names
siteNames = list(siteNameMap.values())
siteNamesRucio = list(siteNameMap.keys()) #these are sitenames as they show in in rucio-transfers

#name of output file name
outputName = "output"+str(now)+".h5"

# This function is used to find the closest value in a list, BELOW some target value. This is used to match the timestamp of a certain transfer with the closest ddm_metrics variable in time.  
def returnMin(lists,target):
	frontrunner = 0
	frontrunnerIndex = None
	for index, List in enumerate(lists):
		if target - List[0] > 0 and target - List[0] < target - frontrunner:
			frontrunner = List[0]
			frontrunnerIndex = index
	if frontrunnerIndex != None:
		return lists[frontrunnerIndex][1]
	else:
		raise Exception('Lookup Failed!')

#################################
# Define and perform ES queries #
#################################

print "Querying Kibana for transfer and metrics information"

query_transfers = {
        "size": 0,
        "query": {
        "bool": {
        "must": [
        {
            "range": {
            "@timestamp": {
                "gte": past,
                "lte": now,
                "format": "epoch_millis"
                }
            }
        },

            {"terms": {"_type": ["transfer-done","transfer-failed"]}},

        ]
        }
        }
    }

query_metrics = {
    "size": 0,
    "query": {
    "bool": {
    "must": [
    {
        "range": {
        "timestamp": {
            "gte": metrics_past,
            "lte": now,
            "format": "epoch_millis"
            }
        }
    },
        
    ],
    "should": [
			  {"exists" : {"field" : "queued-total"}},
			  {"exists" : {"field" : "done-total-1h"}},
			  {"exists" : {"field" : "done-total-6h"}},
			  {"exists" : {"field" : "mbps-dashb-1h"}},			      
			  ],
    "minimum_should_match": 1
    }
    }
}

query_chicago = {
    "size": 0,
    "query": {
    "bool": {
    "must": [
    {
        "range": {
        "timestamp": {
            "gte": metrics_past,
            "lte": now,
            "format": "epoch_millis"
            }
        }
    },
    {"exists" : {"field" : "srcSite"}},
    {"exists" : {"field" : "destSite"}},
    ],
    
    "should": [
			  {"exists" : {"field" : "delay_mean"}},
			  {"exists" : {"field" : "packet_loss"}},
                          ],
    "minimum_should_match": 1,
    
    }
    }
}

# This is where ES is actually queried. Information stored in generator objects, that can be looped over.
scroll_transfers = scan(es, query=query_transfers, index=transfer_index, scroll='5m', timeout='5m', size=10000)
scroll_metrics = scan(es, query=query_metrics, index=ddm_index, scroll='5m', timeout='5m', size=10000)
scroll_chicago = scan(es_chicago, query=query_chicago, index=chicago_index, scroll='5m', timeout='5m', size=10000)

#########################
# Collect Transfer Data #
#########################

print("Collecting Transfer Variables")

#this list will contain one dictionary for each transfer. Each key in this inner dict will correspond to a variable related to the tansfer
transfers = []

counter = 0
for entry in scroll_transfers:

	if not counter%10000:  print "Processing event number ", counter
	counter += 1
	
	#transfer information stored in this dictionary
	transfer = {}
	source = entry['_source']['payload']['src-rse'] #where did the transfer come from
        destination = entry['_source']['payload']['dst-rse'] #where is the transfer going
        #discard entries with sources or destination not consistent with site naming convention:
        if source not in siteNamesRucio: continue
        if destination not in siteNamesRucio: continue
        #convert source and snakkes name according to mapping to ddm name conventions
        source = siteNameMap[source]
        destination = siteNameMap[destination]
        transfertime = entry['_source']['payload']['transferred_at'] # when was the transfer performed/attempted
	submittime = entry['_source']['payload']['submitted_at'] #when was the transfer submitted
	#convert to uppercase to minimise chance of mishaps when comparing to "closeness" information from JSON later.
#	link = source.rsplit("_",1)[0].upper()+":"+destination.rsplit("_",1)[0].upper() # source_destination 
        link = source + ":" + destination
	size = entry['_source']['payload']['bytes'] #size of file being transfered

        #convert transfertime and submittime to epoch milliseconds. Makes comparisons easier.
	transfertime = int(time.mktime(time.strptime(transfertime,pattern)))*1000
	submittime = int(time.mktime(time.strptime(submittime,pattern)))*1000
	delta = transfertime - submittime # identical to time in queue?

	#Which file transfer protocol was used to transfer the file.
	protonum = -1
	protocol = entry['_source']['payload']['protocol']
	if protocol == "srm":
		protonum = 0
	elif protocol == "gsiftp":
		protonum = 1
	elif protocol == "davs":
		protonum = 2
	elif protocol == "root":
		protonum = 3

	# Is this a retried transfer?
	retried = None
	previous_request_id = entry['_source']['payload']['previous-request-id']
	if previous_request_id == None:
		retried = 0
	else:
		retried = 1
		
	#was the transfer successful or not. This is the target the multivariate classifier will train on later.
	if entry['_source']['type'] == "transfer-done":
		successful = 1
	elif entry['_source']['type'] == "transfer-failed":
		successful = 0

	#add variables to transfer dictionary
	transfer["link"] = link
	transfer["transfertime"] = transfertime
	transfer["submittime"] = submittime
	transfer["delta_time"] = delta
	transfer["size"] = size
	transfer["successful"] = successful 
	transfer["source"] = source
	transfer["destination"] = destination
	transfer["protocol"] = protonum
	transfer["retried"] = retried

	#add transfer dictionary to list of transfers.
	transfers.append(transfer)	
	

#############################
# Make Metrics Lookup Table #
#############################

print "Now Building Metric Variables Lookup Dictionaries"

# Dicts for the variables that are being looked up
throughput_link = {} #mean last hour
queued_link = {} # how many files are currently queued across the link in question
done_link_1h = {} # how many successful file transfered occured across the link in the last hour
done_link_6h = {} # how many successful file transfered occured across the link in the last six hours

# These dicts have the following form:
# variable = {"link":[[timestamp1,value1],[timestamp2,value2],...[timestampN,valueN]]}

counter = 0
for entry in scroll_metrics:

	if not counter%10000:  print "Processing event number ", counter
	counter += 1

	source = entry['_source']['src']
	destination = entry['_source']['dst']
	link = source + ":" + destination
        if source not in siteNames: continue
        if destination not in siteNames: continue
	timestamp = entry['_source']['timestamp']
	timestamp = int(time.mktime(time.strptime(timestamp,pattern_ddm)))*1000

	if "done-total-6h" in entry['_source'].keys():
		try:
			done_link_6h[link].append([timestamp,entry['_source']["done-total-6h"]])
		except:
			done_link_6h[link] = []
			done_link_6h[link].append([timestamp,entry['_source']["done-total-6h"]])

	if "done-total-1h" in entry['_source'].keys():
		try:
			done_link_1h[link].append([timestamp,entry['_source']["done-total-1h"]])
		except:
			done_link_1h[link] = []
			done_link_1h[link].append([timestamp,entry['_source']["done-total-1h"]])

	if "queued-total" in entry['_source'].keys():
		try:
			queued_link[link].append([timestamp,entry['_source']["queued-total"]])
		except:
			queued_link[link] = []
			queued_link[link].append([timestamp,entry['_source']["queued-total"]])

	if 'mbps-dashb-1h' in entry['_source'].keys():
		try:
			throughput_link[link].append([timestamp,entry['_source']['mbps-dashb-1h']])
		except:
			throughput_link[link] = []
			throughput_link[link].append([timestamp,entry['_source']['mbps-dashb-1h']])

####################################
# "Chicago" Variables Lookup Table #
####################################

#these variables are not stored in the same place as the other "metrics" variables. Only present in the "network_weather-*"-index of the chicago instance of kibana.

print "Making latency and packet loss lookup table"

# Dicts for the variables that are being looked up
packetloss = {} #mean last hour
latency = {} #mean last hour

counter = 0
for entry in scroll_chicago:

	if not counter%10000: print "Processing event number %d" % counter		
	counter += 1

	source = entry['_source']['srcSite']
	destination = entry['_source']['destSite']
        #Remove "contamination" from CMS sites 
        if source not in siteNames: continue
        if destination not in siteNames: continue

	link = source + ":" + destination
	timestamp = entry['_source']['timestamp'] - 2*60*60*1000 #time difference  between cern and chicago grrrrr#TODO, make this more sophisticated and safe from daylight savings issues etc.

        if "packet_loss" in entry['_source'].keys():
		try:
			packetloss[link].append([timestamp,entry['_source']["packet_loss"]])
		except:
			packetloss[link] = []
			packetloss[link].append([timestamp,entry['_source']["packet_loss"]])

	if "delay_median" in entry['_source'].keys():
		try:
			latency[link].append([timestamp,entry['_source']["delay_median"]])
		except:
			latency[link] = []
			latency[link].append([timestamp,entry['_source']["delay_median"]])

#################################
# Making Closeness Lookup Table #
#################################

print "Now Constructing Closeness Lookup Table"

closeness = {}

#closeness is a static variable, so this can be read from the JSON directly.
url = "http://atlas-adc-netmetrics-lb.cern.ch/metrics/latest.json"

resp = requests.get(url=url)
metrics = resp.json()
for entry in metrics:
	if "closeness" in metrics[entry].keys():
		closeness[entry] = metrics[entry]["closeness"]["latest"]
	

##########################################
# Adding Metrics and Closeness to Events #
##########################################

print "Adding Metrics and Closeness to Transfer Events"

#DEBUGGING
latencies = 0
latency_lookups = 0

final_transfers = []
counter = 0
successes = 0
failures = 0
counter = 0
for entry in transfers:

	if not counter%10000:  print "Processing event number " , counter , "/" , len(transfers)
	counter += 1

	found_all = True

	## Add closeness info. simpler, because it is static.
	try:
            entry["closeness"] = closeness[entry["link"]]
	except:
            found_all = False
            if verbose: print "Failed to find closeness info for link ", entry["link"], ". Dropping event!"
			
	
	#check to see if variable in keys of dict.	
	if entry["link"] in done_link_1h.keys():# and len(done_link_1h[entry["link"]]) > 0:
		
            try:
                entry["done_link_1h"] = returnMin(done_link_1h[entry["link"]],entry["transfertime"]) 
            except:
                found_all = False
                if verbose: print "failed lookup for done_link_1h"
        else:
            if verbose: print "No entry found in done_link_1h. Dropping Event!"
            found_all = False

	if entry["link"] in done_link_6h.keys():# and len(done_link_6h[entry["link"]]) > 0:

            try:
                entry["done_link_6h"] = returnMin(done_link_6h[entry["link"]],entry["transfertime"])
            except:
                found_all = False
                if verbose: print "failed lookup for done_link_6h"
	else:
            if verbose: print "No entry found in done_link_6h. Dropping Event!"
            found_all = False

	if entry["link"] in queued_link.keys():# and len(queued_link[entry["link"]]) > 0:
            try:
                entry["queued_link"] = returnMin(queued_link[entry["link"]],entry["transfertime"])
            except:
                found_all = False
                if verbose: print "failed lookup for queued_link"
	else:
            if verbose: print "No entry found in queued_link. Dropping Event!"
            found_all = False
	
	if entry["link"] in throughput_link.keys():# and len(throughput_link[entry["link"]]) > 0:
            try:
                entry["throughput_link"] = returnMin(throughput_link[entry["link"]],entry["transfertime"])
            except:
                found_all = False
                if verbose: print "failed lookup for throughput_link"
	else:
            if verbose: print "No entry found in throughput_link. Dropping Event!"
            found_all = False

	if entry["link"] in packetloss.keys():
            try:
                entry["packetloss"] = returnMin(packetloss[entry["link"]],entry["transfertime"])
            except:
                found_all = False
                if verbose: print "Failed lookup for packetloss. Dropping Event"
	else:
            if verbose: print "No entry found in packetloss dict. Dropping Event!"
            found_all = False

	if entry["link"] in latency.keys():
            try:
                entry["latency"] = returnMin(latency[entry["link"]],entry["transfertime"])
            except:
                found_all = False
                if verbose: print "Failed lookup for latency. Dropping Event"
	else:
            if verbose: print "No entry found in latency dict. Dropping Event!"
            found_all = False

	if found_all == True:
            successes += 1
            final_transfers.append(entry)
	else:
            failures += 1

        
print "final transfers gathered"
print "number of transfers in final list: ",successes
print "persentage of transfers found: ", float(successes)/(failures + successes)


####################
# Make Numpy Array #
####################

print "Constructing Numpy Array"

#Make separate arrays for all the variables in the final_transfers list
numpy_closeness = np.array([x['closeness'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_delta_time = np.array([x['delta_time'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_throughput_link = np.array([x['throughput_link'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_done_link_6h = np.array([x['done_link_6h'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_done_link_1h = np.array([x['done_link_1h'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_queued_link = np.array([x['queued_link'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_size = np.array([x['size'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_protocol = np.array([x['protocol'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_retried = np.array([x['retried'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_packetloss = np.array([x['packetloss'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)
numpy_latency = np.array([x['latency'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)

numpy_successful = np.array([x['successful'] for x in final_transfers]).reshape((len(final_transfers), 1)).astype(float)

#Make final output array by concatenating the arrays of the separate variables. When updating, also update "variables"
numpy_total = np.concatenate((numpy_closeness,
                              numpy_throughput_link,
                              numpy_done_link_6h,
                              numpy_done_link_1h,
                              numpy_queued_link,
                              numpy_size,numpy_protocol,
                              numpy_retried,
                              numpy_packetloss,
                              numpy_latency,
                              numpy_successful), axis=1)

#list to keep track of the order of variables. Remember to update this when "numpy_total" changes
variables = ["closeness",
             "throughput",
             "done_6h",
             "done_1h",
             "queued",
             "size",
             "protocol",
             "retried",
             "packetloss",
             "latency"]


#######################
# Write Array to File #
#######################

print "Writing Array to File"

#storing file
f = h5py.File(outputName, "w")
f.create_dataset("transfer_data", data=numpy_total)
f.close()

########################################
# Write List of Variables to Text File #
########################################

print "Writing features list to file"

with open("varList_final.py", 'wb') as fp:
        fp.write("variables = [")

        for line in variables[:-1]:
            fp.write(" \"" + line + "\",\n")

        fp.write(" \"" + variables[-1] + "\"")
        fp.write("]")


print "Execution took %i seconds" %int(time.time() - start_time)
print "Done!"
