# MCLoadFlow
Monte Carlo Load Flow
## Abbreviations
std = standard deviation
PDF = Probability Distribtuion Function

## Introduction
This is a set of four modules for the execution of probabilistic load flow, using the base load flow module of PowerFactory.
The scripts can be adapted to work also with other commercial softare (in example with PSS/E), by simply modifying the module named MCLoadFlow, which is providing the exchange of messages between Python and the simulation software.
The four modules are described in the following:

# 1. ConfigFile.py
Running mode: this is a Python script that shall be run from Python console.

This module is used to set the following parameters:
RESULT_PATH = r"C:\...\path to store result files"

SAMPLE_PATH = r"C:\...\path where pick-up measurements files for Wind and PV"

FOLDER_NAME = '' # this is the folder of PowerFactory, where to find the project - as default is equal to '', unless subfolders are used in PowerFactory data manager. 

PROJECT_NAME = 'name of the project in PowerFactory'

STUDY_CASE_NAME = 'name of the study case to activate in PowerFactory'

N_SAMPLES = 8760 # here you can enter the number of samples you would like to analyse

STD_DEV = 0.09 # if a normal distribution function is used for load stochastic model, here the standard deviation can be entered. The software will consider this in the preapration of the PDF, using the actual load values as set in PowerFactory and the standard deviation entered here.

# 2. SyntheticDataGeneration
This 
