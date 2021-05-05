import sys
import os
import csv
import numpy as np
import random
import warnings
from math import sqrt, cos, log, pi
import pandas as pd
import glob as glob
from tqdm import tqdm

sys.path.append(r"D:\Program Files\DIgSILENT\PowerFactory 2021 SP1\Python\3.9")
import powerfactory as pf

class PowerFactorySim(object):

    def __init__(self, folder_name='', project_name='Project', study_case_name='Study Case'):
        # start powerfactory
        self.app = pf.GetApplicationExt()
        # activate project
        self.project = self.app.ActivateProject(os.path.join(folder_name, project_name))
        # activate study case
        study_case_folder = self.app.GetProjectFolder('study')
        print(study_case_folder)
        study_case = study_case_folder.GetContents(
            study_case_name + '.IntCase')[0]
        #self.study_case = study_case
        print(study_case)
        study_case.Activate()
        self.test = 'fine'

    def set_loads(self, mean=100,
                  standard_deviation=20, power_factor=95):

        """
        Function to set all loads in the system, drawn from a distribution.
        Arguments:
            mean - of the distribution to be sampled
            standard_deviation - spread of sampling from mean
            power_factor - if constant power factor required, set argument
        """

        # collect all load elements
        loads = self.app.GetCalcRelevantObjects('*.ElmLod')
        # create keys for loads dictionary form all names of loads
        keys = []
        for key in loads:
            keys.append(key.loc_name)
        # create values for keys drawn from distribution
        values = []
        for _ in range(len(keys)):
            values.append(round(np.random.normal(mean, standard_deviation), 2))
        # create p and q loads dictionary for fixed power factor
        p_loads = {k: v for k, v in zip(keys, values)}
        q_loads = {k: v for k, v in zip(keys, [round(i * (1 - power_factor), 2) for i in values])}
        # set active and reactive loads
        for load in loads:
            load.plini = p_loads[load.loc_name]
            load.qlini = q_loads[load.loc_name]

    def get_loads_from_file(self):

        """
        Function to return loads that were stored in a csv file locally
        """

        path = r"C:\Users\olive\PycharmProjects\power_factory\nominal_loads"

        # read nominal active loads from directory
        active_reader = csv.reader(open(path + r'\active_nominal.csv', 'r'))
        active_loads = {}
        for k, v in active_reader:
            active_loads[k] = float(v)
        # read nominal reactive loads from dictionary
        reactive_reader = csv.reader(open(path + r'\reactive_nominal.csv', 'r'))
        reactive_loads = {}
        for k, v in reactive_reader:
            reactive_loads[k] = float(v)

        return active_loads, reactive_loads

    def set_all_loads_pq(self, p_load, q_load, scale_factor=None):

        """
        Function to set all loads in the system. If loads need to be scaled from
        nominal system values set scale_factor
        """

        loads = self.app.GetCalcRelevantObjects('*.ElmLod')

        if scale_factor is not None:
            for key in p_load:
                p_load[key] *= scale_factor
            for key in q_load:
                q_load[key] *= scale_factor
            for key in p_load:
                p_load[key] = int(p_load[key])
            for key in q_load:
                q_load[key] = int(q_load[key])

        for load in loads:
            load.plini = p_load[load.loc_name]
            load.qlini = q_load[load.loc_name]

    def set_all_vre_op(self, idx):

        """
        Function to set weibull wind speed for all wind generator in the system and power output for PV
        """

        vres = self.app.GetCalcRelevantObjects('*.ElmGenstat')

        for vre in vres:
            if vre.chr_name == 'wind':
                vre.windspeed = inputSample.loc[idx, vre.loc_name]
            if vre.chr_name == 'pv':
                vre.pgini = inputSample.loc[idx, vre.loc_name]

    def get_all_loads_pq(self):

        """
        Function to return all system loads under current state
        """

        loads = self.app.GetCalcRelevantObjects('*.ElmLod')
        p_base = {}
        q_base = {}

        for load in loads:
            p_base[load.loc_name] = load.plini
            q_base[load.loc_name] = load.qlini

        return p_base, q_base

    def get_all_windgenspeed(self):

        """
        Function to return all wind speed reference under current state
        """

        windgens = self.app.GetCalcRelevantObjects('*.ElmGenstat')
        windspeed_scale = {}

        for windgen in windgens:
            windspeed_scale[windgen.loc_name] = windgen.windspeed

        return windspeed_scale

    def prepare_loadflow(self, ldf_mode='balanced'):

        """
        Function to prepare conditions for load flow calculation
        """

        modes = {'balanced': 0,
                 'unbalanced': 1,
                 'dc': 2}
        # get load flow object
        self.ldf = self.app.GetFromStudyCase('ComLdf')
        # set load flow mode
        self.ldf.iopt_net = modes[ldf_mode]

    def run_loadflow(self):
        return bool(self.ldf.Execute())

    def get_bus_voltages(self):

        """
        Function to return bus voltages following load flow calculation
        """

        voltages = {}
        # collect all bus elements
        buses = self.app.GetCalcRelevantObjects('*.ElmTerm', 0) # with the zero value is not considering out of service elements
        # store voltage of each bus in a dictionary
        for bus in buses:
            if bus.desc == ['obs']:
                #removed: bus_name = bus.GetNodeName() # is giving a unique name to the node
                voltages[bus.loc_name] = bus.GetAttribute('m:u')
        return voltages

    def get_line_loading(self):

        """
        Function to return line loading and power following load flow calculation
        """

        line_loading = {}
        line_power = {}
        # collect all line elements
        lines = self.app.GetCalcRelevantObjects('*.ElmLne', 0)
        # store loading of each line in a dictionary
        for line in lines:

                line_loading[line.loc_name] = line.GetAttribute('c:loading')
                line_power[line.loc_name] = line.GetAttribute('m:Psum:bus1')

        return line_loading, line_power

    def get_trafo_loading(self):

        """
        Function to return trafo loading following load flow calculation
        """

        trafo_loading = {}
        # collect all transformers elements
        trafos = self.app.GetCalcRelevantObjects('*.ElmTr2, *.ElmTr3, *.ElmTr4, *.ElmTrb', 0)
        # store loading of each transformer in a dictionary
        for trafo in trafos:
            #if bus.iUsage == 0: # modified it was only 0 - problem to be solved: how to select all terminals
                trafo_loading[trafo.loc_name] = trafo.GetAttribute('c:loading')

        return trafo_loading

    def get_power_losses(self):

        """
        Function to return power losses following load flow calculation
        """

        grid_losses = {}
        # collect all grid elements
        grids = self.app.GetCalcRelevantObjects('*.ElmNet', 0)
        # store losses of each grid in a dictionary
        for grid in grids:
            grid_losses[grid.loc_name] = grid.GetAttribute('c:LossP')
        return grid_losses

    def get_non_dispatchable_power(self):

        """
        Function to return power from VRE, following load flow calculation
        """

        ndp_wind = {}
        ndp_pv = {}

        # collect all vre elements
        vres = self.app.GetCalcRelevantObjects('*.ElmGenstat', 0)
        # store power outout of each vre in a dictionary
        for vre in vres:
            if vre.chr_name == 'wind': # here vre is only wind power
                ndp_wind[vre.loc_name] = vre.GetAttribute('m:P:bus1')

            if vre.chr_name == 'pv':
                ndp_pv[vre.loc_name] = vre.GetAttribute('m:P:bus1')

        return ndp_wind, ndp_pv

class MontecarloLoadFlow(PowerFactorySim):

    """
    Class to run a monte carlo load flow. Credit for original version:
    {Probabilistic Power Flow Module for PowerFactory DIgSILENT,
    Saeed Teimourzadeh, Behnam Mohammadi-Ivatloo}
    """

    def gen_normal_loads_pq(self, p_total, q_total, p_base, q_base, std_dev=0.1):
        #generate two random number from uniform distribution
        rand1 = random.uniform(0,1)
        rand2 = random.uniform(0,1)
        #sample loads from a normal distribution
        p_total_rand = p_total*(1 + std_dev*sqrt(-2*log(rand1))*cos(2*pi*rand2))
        q_total_rand = q_total * (1 + std_dev * sqrt(-2 * log(rand1)) * cos(2 * pi * rand2))

        loads = self.app.GetCalcRelevantObjects('*.ElmLod')
        #store normally distributed load values as dict
        p_normal = {}
        q_normal = {}
        for load in loads:
            p_normal[load.loc_name] = (p_base[load.loc_name]/p_total*p_total_rand)
            q_normal[load.loc_name] = (q_base[load.loc_name]/q_total*q_total_rand)
        return p_normal, q_normal, p_total_rand, q_total_rand

    def gen_vre_dist(self):

        #generate random number from Weibull distribution
        vres = self.app.GetCalcRelevantObjects('*.ElmGenstat')
        #store weibull distributed wind speed values as dict
        ws_weibull = {}
        pv_dist = {}

        for vre in vres:
            if vre.desc !=[] and vre.chr_name == 'vre':
                w = vre.desc
                w = float(w[0])
                ws_weibull[vre.loc_name] = float(np.random.weibull(w, 1)*windspeed_scale[vre.loc_name])

            if vre.chr_name == 'pv':
                row = random.randint(0, len(pvpd.index)-1)
                pv_dist[vre.loc_name] = float(pvpd.loc[row, vre.desc] * vre.Pmax_uc)
        return ws_weibull, pv_dist

    def get_sample_files(self, sample_path):
        all_files = glob.glob(sample_path + '\*.csv')
        inputSample = pd.DataFrame()
        for filename in all_files:
            li = pd.read_csv(filename, skip_blank_lines=True, header=[0])
            inputSample[li.columns.values] = li[li.columns.values]
        return inputSample.abs()

    def monte_carlo_loadflow(self, n_samples, sample_path, std_dev, day=False, max_attempts=10):
        self.prepare_loadflow()

        '''
        get Sample DataFrame for MC
        '''
        #get initial loads
        p_base, q_base = self.get_all_loads_pq()

        #calculate total base system load
        p_total = sum(p_base.values())
        q_total = sum(q_base.values())

        # create dataframe for pv stochastic distribution
        global inputSample # making data frame global, to be used in other functions (i.e.self.gen_vre_dist)
        inputSample = self.get_sample_files(sample_path)
        print(inputSample)

        #running the load flow for the established number of samples
        sample_i = 0
        for sample in tqdm(range(n_samples)):
            #re-attempt load flow in case of non-convergence
            p_total_rand = 0
            q_total_rand = 0
            for attempt in range(max_attempts):
                #generate random normally distributed loads
                p_normal, q_normal, p_total_rand, q_total_rand = self.gen_normal_loads_pq(
                    p_total, q_total, p_base, q_base, std_dev=std_dev)
                #ws_weibull, pv_dist = self.gen_vre_dist() #samples are already available on DataFrame inputSample

                #set system loads and VRE to random loads
                self.set_all_loads_pq(p_normal, q_normal)
                self.set_all_vre_op(sample_i)
                sample_i = sample_i+1
                #run load flow
                failed = self.run_loadflow()
                if failed:
                    warnings.warn(
                        "sample " +str(sample)
                        + " did not converge, re-attempt "
                        + str(attempt + 1) + " out of "
                        + str(max_attempts)
                    )
                else:
                    break
            p = (p_total_rand / p_total)
            q = (q_total_rand / q_total)
            vre_out = self.get_non_dispatchable_power()
            yield self.get_bus_voltages(), self.get_line_loading()[0], self.get_trafo_loading(), p_normal, \
                  self.get_power_losses(), vre_out[0], vre_out[1], self.get_line_loading()[1]

            #restore system to base loads
        self.set_all_loads_pq(p_base, q_base)