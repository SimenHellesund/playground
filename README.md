# Playground
Author: Simen Hellesund
Contact: shellesu@cern.ch

## Introduction
The goal of this project is to make a system that can predict failures in file transfers made using the ATLAS Distributed Data Management (DDM) system on the LHC worldwide computing grid (grid). 

The system is currently split into two main parts: One that collects historical data about failed and successful file transfers as well as the state of the system. The other uses this data to train and test multivariate classfiers.

## Prerequisites
External packages needed:
* h5py
* Elasticsearch
* Scikit-learn
* Keras
* Tensorflow

Transfer data is collected using ElasticSearch and stored as numpy arrays in .h5 files using the h5py package. Scikit-learn, Keras and Tensorflow are all used to implement machine learning methods.

The simplest way I found to set up these tools were to use Anaconda. Once installed a conda environment with a chosen set of packages can be set up using the command:

    conda create -n NAME_OF_ENV python=2.7 numpy scipy matplotlib scikit-learn keras tensorflow h5py

Once set up, activate this environment to work inside it by typing the following command:

    source activate NAME_OF_ENV

And to deactivate:

    source deactivate

## Usage



