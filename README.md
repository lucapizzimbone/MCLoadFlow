# MCLoadFlow
This file provides a description and the instructions to use the MCLoadFLow package, which is including the following components:
- ConfigFile
- SampleGen
- SimComp
- DataVis

The above components are illustrated in the following UML diagram:

![image](https://user-images.githubusercontent.com/82202509/119944564-b785db00-bf94-11eb-969b-3138f12a8c7f.png)

## Abbreviations
CDF = Cumulative Distribution Function

PDF = Probability Distribtuion Function

std = standard deviation

## Introduction
This is a set of four components for the execution of probabilistic load flow, using the base load flow module of PowerFactory.
The scripts can be adapted to work also with other commercial softare (in example with PSS/E), by simply modifying the component named MCLoadFlow, which is providing the exchange of messages between Python and the simulation software.
The four components are described in the following:

## 1. ConfigFile
Running mode: this is a Python script that shall be run from Python console.

This component is used to set the following parameters:
RESULT_PATH = r"C:\...\path to store result files"

SAMPLE_PATH = r"C:\...\path where pick-up measurements files for Wind and PV"

FOLDER_NAME = '' # this is the folder of PowerFactory, where to find the project - as default is equal to '', unless subfolders are used in PowerFactory data manager. 

PROJECT_NAME = 'name of the project in PowerFactory'

STUDY_CASE_NAME = 'name of the study case to activate in PowerFactory'

N_SAMPLES = 8760 # here you can enter the number of samples you would like to analyse

STD_DEV = 0.09 # if a normal distribution function is used for load stochastic model, here the standard deviation can be entered. The software will consider this in the preapration of the PDF, using the actual load values as set in PowerFactory and the standard deviation entered here.



## 2. SampleGen
This component is constituted by a set of Jupiter Notebook files for the selection of the best fitting PDF, based on input samples (i.e. field measureemnts) and the generation of synthetic data, based on the calcuated PDF. The component is generating the output in *.csv format and in diagrams.

## 3. SimComp
This component is the interface for PowerFactory. 
At first, it is necessary to set the path where PowerFactory can find the PowerFactory.pyd file (usually to be found in a subfolder of the PowerFactory installation directory)

sys.path.append(r"D:\...\path within PowerFactory directory to find \Python\x.x) # here the path for PowerFactory.pyd file shall be given 

Hence, the next step is to set-up the PowerFactory project. These are the set-ups:

- for the bus or terminals to be monitored by the Python scripts, the label 'obs' shall be typed in the description field (Attribute: desc) of the bus or terminal elements in PowerFactory, as per following figure:

![image](https://user-images.githubusercontent.com/82202509/117000552-868def80-ace1-11eb-8b33-f540905d3fda.png)

- For the wind and PV system, the label 'wind' or 'pv' shall be respectively typed in the in the characteristic name field (Attribute: chr_name) of the generator element of PowerFactory, as per following figures:
- 
![image](https://user-images.githubusercontent.com/82202509/117001040-29466e00-ace2-11eb-836c-98c0c2a3b489.png)
![image](https://user-images.githubusercontent.com/82202509/117001104-39f6e400-ace2-11eb-8afb-de9a92f42e85.png)

The MCLoadFLow will return the data to create the *.csv files that will be used by the component DataVisualization.

## 4. DataVis
The component for data visualization, named DataVisualization is generating a set of plots, for the visualization of the output data, generated by the MCLoadFLow component, through the interface ConfigFile. The plots include: (i) box plots for voltage at nodes and loading of lines and transformers, (ii) scatter plots and PDF/CDF plots for voltage at nodes and loading of lines and transformers, inlcuding correlation values, (iii) grid losses diagrams. Plots are saved in pdf files and can also be visualized after the execution of the script. The results are provided for all monitored elements with the automatic calculation of max and min values (absolute and percentile). The DataVisualization component allows the setting of the following parameters:

PATH = r'D:\...\path where to find result files'
MIN_VOLTAGE = 0.95  # filter for minimum voltage plots
MAX_VOLTAGE = 1.05  # filter for maximum voltage plots 
TRAFO_LOADING = 90 # filter for transformer maximum loading plots
LINE_LOADING = 80 # filter for line maximum loading plots
MAX_QUANTILE = 0.95 # filter for selection of max loaded elements or max voltage based on quantile values
MIN_QUANTILE = 0.05 # filter for selection of max loaded elements or max voltage based on quantile values
PLOT_FIGURES = True # if False, no plots are shown and only pdf files will be saved in the same PATH of the csv input files.

## How to use the scripts
1. Copy all the scripts in the same directory and create a subfolder for the measurement data (input) and one for the output results (output). You may use any name. The path of these folders shall be specified in the ConfigFile and DataVisualization, as described in the previous sections.
2. Hence use the Jupyter notebook SyntheticDataGeneration and follow the instructions given in the notebook itself, to load measurement *.csv files, visulaize the PDF diagrams and generate a sufficient number of Synthetic samples.
3. PowerFactory model shall be also ready for the MC simulations, as described in the section 3. 
4. The next step is to setup the ConfigFile, as described in section 1 and run the script. The script will automatically recall the functions in the MCLoadFlow component. PowerFactory shall be closed in this time, as it works in GUI-less mode. A progress bar will indicate the time elapsed from the beginning of the simulation and the remaining time. After the completion of the simulation, the files will be saved in the selected directory.
5. Finally, run the DataVisualization script to generate plots and relevant *.pdf files.

End of the ReadMe File
