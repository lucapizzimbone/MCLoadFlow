import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from matplotlib.collections import PathCollection
from matplotlib.backends.backend_pdf import PdfPages
import colorsys
from scipy import stats

# SET FOLLOWING ELEMENTS
PATH = r'D:\Software_tools\PowerFactory_tests\Results_test'
MIN_VOLTAGE = 0.95  # plot filter for not energized buses
MAX_VOLTAGE = 1.05  # filter box plots for voltage at buses
TRAFO_LOADING = 90 # plot filter percentage of loading
LINE_LOADING = 80 # plot filter percentage of loading
MAX_QUANTILE = 0.95 # filter for selection of max loaded elements or max voltage based on quantile values
MIN_QUANTILE = 0.05 # filter for selection of max loaded elements or max voltage based on quantile values
CLUSTERS = True
PLOT_FIGURES = True # if False, no plots are shown. Files will be saved anyhow

'''
function for plot histogram for cumulative probability
'''
def hist(data, bins, title, labels, range = None):
  fig = plt.figure(figsize=(15, 8))
  ax = plt.axes()
  plt.ylabel("Proportion")
  values, base, _ = plt.hist(data, bins=bins, density=True, alpha=0.5, color="tab:green", range=range, label="Histogram")
  ax_bis = ax.twinx()
  values = np.append(values,0)
  ax_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize = 1, label="Cumulative Histogram" )
  plt.xlabel(labels)
  plt.ylabel("Proportion")
  plt.title(title)
  ax_bis.legend();
  ax.legend();
  return

# following function is necessary to flat dataframe values to numpy list
def flatlist(x):
    flat_x = []
    for sublist in x:
        for item in sublist:
            flat_x.append(item)
    return flat_x

def lin(x):
    return slope*x+intercept

def lin2(x):
    return slope2*x+intercept2

# read csv files generated by the probabilistic analysis
voltage_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_bus.csv')
lne_loading_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_lne_ld.csv')
lne_power_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_lne_pw.csv')
trafo_loading_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_trf.csv')
p_rand_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_p.csv')
grid_losses_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_power_losses.csv')
vrewi_power_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_power_vre_wind.csv')
vrepv_power_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_power_vre_pv.csv')
if CLUSTERS:
    cluster_df = pd.read_csv(PATH + r'\res_prob_lf_1000_std0.1_clusters.csv')

style.use('seaborn-white') # change to whitegrid is grid is needed

#print(voltage_df.describe(include='all'))

#get sum of vre for visualization of scatter diagrams
vre_power_tot = vrewi_power_df.sum(axis=1) + vrepv_power_df.sum(axis=1)
pd.set_option('display.max_rows', 10)

'''
look for buses with max and min voltage
'''
voltage_filtered_boo = voltage_df.max()[voltage_df.max()>0.1] # take out not energised buses
voltage_filtered = voltage_df[voltage_filtered_boo.index]
voltage_sup_boo = voltage_filtered.max()[voltage_filtered.max() >= MAX_VOLTAGE]
voltage_sup = voltage_filtered[voltage_sup_boo.index]
if voltage_sup.empty:
    print('no max voltage violation occurred (max voltage plot will not be printed)')
    VOLTAGE_SUP = False
else:
    VOLTAGE_SUP = True
voltage_inf_boo = voltage_filtered.min()[voltage_filtered.min() <= MIN_VOLTAGE]
voltage_inf = voltage_filtered[voltage_inf_boo.index]
if voltage_inf.empty:
    print('no min voltage violation occurred (min voltage plot will not be printed)')
    VOLTAGE_INF = False
else:
    VOLTAGE_INF = True
max_vol_col = voltage_filtered.max()
min_vol_col = voltage_filtered.min()
bus_max_v = max_vol_col[max_vol_col.values == max_vol_col.max()].index.values
bus_min_v = min_vol_col[min_vol_col.values == min_vol_col.min()].index.values
print('bus with maximum voltage: {} = {} p.u.'.format(bus_max_v[0], max_vol_col[max_vol_col.values == max_vol_col.max()].values[0]))
print('bus with minimum voltage: {} = {} p.u.'.format(bus_min_v[0], min_vol_col[min_vol_col.values == min_vol_col.min()].values[0]))

'''
look for buses with max and min voltage quantile
'''
maxq_vol_col = voltage_filtered.quantile(MAX_QUANTILE)
minq_vol_col = voltage_filtered.quantile(MIN_QUANTILE)
bus_maxq_v = maxq_vol_col[maxq_vol_col.values == maxq_vol_col.max()].index.values
bus_minq_v = minq_vol_col[minq_vol_col.values == minq_vol_col.min()].index.values
print('bus with maximum {} quantile voltage: {} = {} p.u.'.format(MAX_QUANTILE, bus_maxq_v[0],
                                                                  maxq_vol_col[maxq_vol_col.values ==
                                                                               maxq_vol_col.max()].values[0]))
print('bus with minimum {} quantile voltage: {} = {} p.u.'.format(MIN_QUANTILE, bus_minq_v[0],
                                                                  minq_vol_col[minq_vol_col.values ==
                                                                               minq_vol_col.min()].values[0]))

'''
look for line with max loading
'''
max_loadln_col = lne_loading_df.max()
lne_max_load = max_loadln_col[max_loadln_col.values == max_loadln_col.max()].index.values
print('line with maximum loading: {} = {} %'.format(lne_max_load[0],
                                                    max_loadln_col[max_loadln_col.values ==
                                                                   max_loadln_col.max()].values[0]))

'''
look for line with max  quantile loading
'''
maxq_loadln_col = lne_loading_df.quantile(MAX_QUANTILE)
lne_maxq_load = maxq_loadln_col[maxq_loadln_col.values == maxq_loadln_col.max()].index.values
#print(lne_max_load, maxq_loadln_col[maxq_loadln_col.values == maxq_loadln_col.quantile(0.75)])
print('line with maximum {} quantile loading: {} = {} %'.format(MAX_QUANTILE, lne_maxq_load[0],
                                                                maxq_loadln_col[maxq_loadln_col.values ==
                                                                                maxq_loadln_col.max()].values[0]))

'''
look for trf with max loading
'''
max_loadtr_col = trafo_loading_df.max()
trf_max_load = max_loadtr_col[max_loadtr_col.values == max_loadtr_col.max()].index.values
print('transformer with maximum loading: {} = {} %'.format(trf_max_load[0],
                                                           max_loadtr_col[max_loadtr_col.values ==
                                                                          max_loadtr_col.max()].values[0]))

'''
look for trf with max quantile loading
'''
maxq_loadtr_col = trafo_loading_df.quantile(MAX_QUANTILE)
trf_maxq_load = maxq_loadtr_col[maxq_loadtr_col.values == maxq_loadtr_col.max()].index.values
print('transformer with maximum {} quantile loading: {} = {} %'.format(MAX_QUANTILE, trf_maxq_load[0],
                                                                maxq_loadtr_col[maxq_loadtr_col.values ==
                                                                                maxq_loadtr_col.max()].values[0]))


'''
#print subplots for buses with max and min voltage
#scatter and cumulative probability
'''
# create subplots
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(2,2,1)
plt.scatter(vre_power_tot, voltage_filtered[bus_max_v], color='tab:red', s=2)

x = vre_power_tot.values
y = flatlist(voltage_filtered[bus_max_v].values)
slope, intercept, r, p, std_err = stats.linregress(x, y)
#plot linear regression

linmodel = list(map(lin,vre_power_tot))
plt.plot(x, linmodel, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope*vre_power_tot.values.max(), r, p))

ax2 = fig.add_subplot(2,2,2)
values, base, _ = plt.hist(voltage_filtered[bus_max_v], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax2_bis = ax2.twinx()
values = np.append(values,0)
ax2_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )
ax3 = fig.add_subplot(2,2,3)
plt.scatter(vre_power_tot, voltage_filtered[bus_min_v], color='tab:blue', s=2)

x2 = vre_power_tot.values
y2 = flatlist(voltage_filtered[bus_min_v].values)
slope2, intercept2, r2, p2, std_err2 = stats.linregress(x2, y2)
#plot linear regression

linmodel2 = list(map(lin2,vre_power_tot))
plt.plot(x2, linmodel2, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope2*vre_power_tot.values.max(), r2, p2))

ax4 = fig.add_subplot(2,2,4)
values, base, _ = plt.hist(voltage_filtered[bus_min_v], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax4_bis = ax4.twinx()
values = np.append(values,0)
ax4_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )

#set titles and labels
fig.suptitle('Probabilistic Analysis - Max and Min Voltages', fontsize=14)
ax1.set_title('Voltage vs VRE power (Max Voltage) '+str(bus_max_v[0]), fontsize=11)
ax1.legend()
ax1.set(xlabel='Total VRE Power [MW]')
ax1.set(ylabel='Voltage [p.u.]')
ax2.set_title(bus_max_v[0], fontsize=11)
ax2.set(xlabel='Voltage [p.u.]')
ax2.set(ylabel='Proportion')
ax2_bis.set(ylabel='Proportion')
ax2_bis.legend();
ax2.legend();
ax3.set_title('Voltage vs VRE power (Min Voltage) '+str(bus_min_v[0]), fontsize=11)
ax3.legend()
ax3.set(xlabel='Total VRE Power [MW]')
ax3.set(ylabel='Voltage [p.u.]')
ax4.set_title(bus_min_v[0], fontsize=11)
ax4.set(xlabel='Voltage [p.u.]')
ax4.set(ylabel='Proportion')
ax4_bis.legend();
ax4.legend();
fig.tight_layout(h_pad=1)
fig.savefig('Voltage_max_min.png')

'''
#print subplots for buses with 0.75 and 0.25 voltage quantile
#scatter and cumulative probability
'''
# create subplots
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(2,2,1)
plt.scatter(vre_power_tot, voltage_df[bus_maxq_v], color='tab:red', s=2)

x = vre_power_tot.values
y = flatlist(voltage_df[bus_maxq_v].values)
slope, intercept, r, p, std_err = stats.linregress(x, y)

#plot linear regression

linmodel = list(map(lin,vre_power_tot))
plt.plot(x, linmodel, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope*vre_power_tot.values.max(), r, p))

ax2 = fig.add_subplot(2,2,2)
values, base, _ = plt.hist(voltage_filtered[bus_maxq_v], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax2_bis = ax2.twinx()
values = np.append(values,0)
ax2_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )
ax3 = fig.add_subplot(2,2,3)
plt.scatter(vre_power_tot, voltage_df[bus_minq_v], color='tab:blue', s=2)

x2 = vre_power_tot.values
y2 = flatlist(voltage_df[bus_minq_v].values)
slope2, intercept2, r2, p2, std_err2 = stats.linregress(x2, y2)
#plot linear regression

linmodel2 = list(map(lin2,vre_power_tot))
plt.plot(x2, linmodel2, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope2*vre_power_tot.values.max(), r2, p2))

ax4 = fig.add_subplot(2,2,4)
values, base, _ = plt.hist(voltage_filtered[bus_minq_v], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax4_bis = ax4.twinx()
values = np.append(values,0)
ax4_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )

#set titles and labels
fig.suptitle('Probabilistic Analysis - Max and Min Voltages ({} and {} quantile)'.format(MAX_QUANTILE, MIN_QUANTILE), fontsize=14)
ax1.set_title('Voltage vs VRE power (Max {} Quantile Voltage) '.format(MAX_QUANTILE)+str(bus_maxq_v[0]), fontsize=11)
ax1.legend()
ax1.set(xlabel='Total VRE Power [MW]')
ax1.set(ylabel='Voltage [p.u.]')
ax2.set_title(bus_maxq_v[0], fontsize=11)
ax2.set(xlabel='Voltage [p.u.]')
ax2.set(ylabel='Proportion')
ax2_bis.set(ylabel='Proportion')
ax2_bis.legend();
ax2.legend();
ax3.set_title('Voltage vs VRE Power (Min {} Quantile Voltage) '.format(MIN_QUANTILE)+str(bus_minq_v[0]), fontsize=11)
ax3.legend()
ax3.set(xlabel='Total VRE Power [MW]')
ax3.set(ylabel='Voltage [p.u.]')
ax4.set_title(bus_minq_v[0], fontsize=11)
ax4.set(xlabel='Voltage [p.u.]')
ax4.set(ylabel='Proportion')
ax4_bis.legend();
ax4.legend();
fig.tight_layout(h_pad=1)
fig.savefig('Voltage_max_min_quantile.png')

'''
#print subplots for lines and transformers with max loading
#scatter and cumulative probability - calcuate correlation by linear regression
'''
# create subplots
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(2,2,1)
plt.scatter(vre_power_tot, lne_loading_df[lne_max_load], color='tab:red', s=2)

x = vre_power_tot.values
y = flatlist(lne_loading_df[lne_max_load].values)
slope, intercept, r, p, std_err = stats.linregress(x, y)

#plot linear regression

linmodel = list(map(lin,vre_power_tot))
plt.plot(x, linmodel, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope*vre_power_tot.values.max()/100, r, p))
ax2 = fig.add_subplot(2,2,2)
values, base, _ = plt.hist(lne_loading_df[lne_max_load], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax2_bis = ax2.twinx()
values = np.append(values,0)
ax2_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )
ax3 = fig.add_subplot(2,2,3)
plt.scatter(vre_power_tot, trafo_loading_df[trf_max_load], color='tab:red', s=2)

x2 = vre_power_tot.values
y2 = flatlist(trafo_loading_df[trf_max_load].values)
slope2, intercept2, r2, p2, std_err2 = stats.linregress(x2, y2)
#plot linear regression

linmodel2 = list(map(lin2,vre_power_tot))
plt.plot(x2, linmodel2, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope2*vre_power_tot.values.max()/100, r2, p2))

#m2, b2 = np.polyfit(vre_power_tot, trafo_loading_df[trf_max_load], 1)
#plt.plot(vre_power_tot, m2*vre_power_tot+b2, color='darkslategray', label="Slope = {}".format(m2[0]))
ax4 = fig.add_subplot(2,2,4)
values, base, _ = plt.hist(trafo_loading_df[trf_max_load], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax4_bis = ax4.twinx()
values = np.append(values,0)
ax4_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )

#set titles and labels
fig.suptitle('Probabilistic Analysis - Max Loading at Lines and Transformers', fontsize=14)
ax1.set_title('Line loading vs VRE power (Max Loading) '+str(lne_max_load[0]), fontsize=11)
ax1.set(xlabel='Total VRE Power [MW]')
ax1.set(ylabel='Loading [%]')
ax1.legend()
ax2.set_title(lne_max_load[0], fontsize=11)
ax2.set(xlabel='Loading [%]')
ax2.set(ylabel='Proportion')
ax2_bis.set(ylabel='Proportion')
ax2_bis.legend();
ax2.legend();
ax3.set_title('Transformer Loading vs VRE Power (Max Loading) '+str(trf_max_load[0]), fontsize=11)
ax3.legend()
ax3.set(xlabel='Total VRE Power [MW]')
ax3.set(ylabel='Loading [%]')
ax4.set_title(trf_max_load[0], fontsize=11)
ax4.set(xlabel='Loading [%]')
ax4.set(ylabel='Proportion')
ax4_bis.legend();
ax4.legend();
fig.tight_layout(h_pad=1)
fig.savefig('Loading_max.png')

'''
#print subplots for lines and transformers with max 0.75 quantile loading
#scatter and cumulative probability
'''
# create subplots
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(2,2,1)
plt.scatter(vre_power_tot, lne_loading_df[lne_maxq_load], color='tab:red', s=2)

x = vre_power_tot.values
y = flatlist(lne_loading_df[lne_maxq_load].values)
slope, intercept, r, p, std_err = stats.linregress(x, y)

#plot linear regression

linmodel = list(map(lin,vre_power_tot))
plt.plot(x, linmodel, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope*vre_power_tot.values.max()/100, r, p))
ax2 = fig.add_subplot(2,2,2)
values, base, _ = plt.hist(lne_loading_df[lne_maxq_load], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax2_bis = ax2.twinx()
values = np.append(values,0)
ax2_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )
ax3 = fig.add_subplot(2,2,3)
plt.scatter(vre_power_tot, trafo_loading_df[trf_maxq_load], color='tab:red', s=2)

x2 = vre_power_tot.values
y2 = flatlist(trafo_loading_df[trf_maxq_load].values)
slope2, intercept2, r2, p2, std_err2 = stats.linregress(x2, y2)
#plot linear regression

linmodel2 = list(map(lin2,vre_power_tot))
plt.plot(x2, linmodel2, color='darkslategray', label="Slope = {}\n"
                                        "Correlation = {}\n"
                                        "p-value = {}".format(slope2*vre_power_tot.values.max()/100, r2, p2))



#m2, b2 = np.polyfit(vre_power_tot, trafo_loading_df[trf_maxq_load], 1)
#plt.plot(vre_power_tot, m2*vre_power_tot+b2, color='darkslategray', label="Slope = {}".format(m2[0]))
ax4 = fig.add_subplot(2,2,4)
values, base, _ = plt.hist(trafo_loading_df[trf_maxq_load], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
ax4_bis = ax4.twinx()
values = np.append(values,0)
ax4_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )

#set titles and labels
fig.suptitle('Probabilistic Analysis - Max {} Quantile Loading at Lines and Transformers'.format(MAX_QUANTILE), fontsize=14)
ax1.set_title('Line loading vs VRE power (Max Loading) '+str(lne_maxq_load[0]), fontsize=11)
ax1.legend()
ax1.set(xlabel='Total VRE Power [MW]')
ax1.set(ylabel='Loading [%]')
ax2.set_title(lne_maxq_load, fontsize=11)
ax2.set(xlabel='Loading [%]')
ax2.set(ylabel='Proportion')
ax2_bis.set(ylabel='Proportion')
ax2_bis.legend();
ax2.legend();
ax3.set_title('Transformer loading vs VRE power (Max Loading) '+str(trf_maxq_load[0]), fontsize=11)
ax3.legend()
ax3.set(xlabel='Total VRE Power [MW]')
ax3.set(ylabel='Loading [%]')
ax4.set_title(trf_maxq_load, fontsize=11)
ax4.set(xlabel='Loading [%]')
ax4.set(ylabel='Proportion')
ax4_bis.legend();
ax4.legend();
fig.tight_layout(h_pad=1)
fig.savefig('Loading_max.png')

'''
#  plot box for bus
'''
if VOLTAGE_SUP:
    fig = plt.figure(figsize=(12, 7))
    #flierprops = dict(marker='o', markerfacecolor='tab:blue', markersize=5,
                      #linestyle='none')
    ax1 = fig.add_subplot(1,1,1)
    r='red'
    b='royalblue'
    g='gray'
    b = 'blue'
    k='black'
    plt.boxplot(voltage_sup, patch_artist=True, boxprops=dict(facecolor=r, color=g), capprops=dict(color=r), whiskerprops=dict(color=r),
                flierprops=dict(color=r, markeredgecolor=g), medianprops=dict(color=k))
    #voltage_sup.plot.box(flierprops=flierprops, ax=ax1)
    #voltage_inf.plot.box(flierprops=flierprops, ax=ax2)
    fig.suptitle('Probabilistic Analysis', fontsize=14)
    ax1.set_title('Bus Voltage for Superior Limits (above {} p.u.)'.format(MAX_VOLTAGE), fontsize=11)
    ax1.set(xlabel='Nodes')
    ax1.set(ylabel='Voltage [p.u.]')
    ax1.set_xticklabels(voltage_sup.columns)
    ax1.tick_params(labelrotation=45)
    fig.tight_layout(h_pad=1)
    fig.savefig('PlotBoxBus_max.png')
if VOLTAGE_INF:
    fig = plt.figure(figsize=(12, 7))
    ax2 = fig.add_subplot(1,1,1)
    plt.boxplot(voltage_inf, patch_artist = True, boxprops = dict(facecolor=b, color=g), capprops = dict(color=b), whiskerprops = dict(color=b),
    flierprops = dict(color=b, markeredgecolor=g), medianprops = dict(color=k))
    fig.suptitle('Probabilistic Analysis', fontsize=14)
    ax2.set_title('Bus Voltage for Inferior Limits (below {} p.u.)'.format(MIN_VOLTAGE), fontsize=11)
    ax2.set(xlabel='Nodes')
    ax2.set(ylabel='Voltage [p.u.]')
    ax2.set_xticklabels(voltage_inf.columns)
    ax2.tick_params(labelrotation=45)
    fig.tight_layout(h_pad=1)
    fig.savefig('PlotBoxBus_min.png')

'''
#  plot box for lines
'''
fig = plt.figure(figsize=(12, 7))
ax2 = fig.add_subplot(1,1,1)
lne80 = lne_loading_df.max()[lne_loading_df.max()>LINE_LOADING]
lne_filtered = lne_loading_df[lne80.index]
plt.boxplot(lne_filtered, patch_artist=True, boxprops=dict(facecolor=r, color=g), capprops=dict(color=r), whiskerprops=dict(color=r),
                flierprops=dict(color=r, markeredgecolor=g), medianprops=dict(color=k))

# set titles
fig.suptitle('Probabilistic Analysis', fontsize=14)
ax2.set_title(' Statistic Data for Line Loading (filter for line loading > {}%)'.format(LINE_LOADING), fontsize=11)
ax2.set(xlabel='Lines')
ax2.set(ylabel='Loading [%]')
ax2.set_xticklabels(lne_filtered.columns)
ax2.tick_params(labelrotation=45)
fig.tight_layout(h_pad=1)
fig.savefig('PlotBoxLne.png')

'''
#  plot box for transformers
'''

fig = plt.figure(figsize=(12, 7))
ax3 = fig.add_subplot(1,1,1)
trafo80 = trafo_loading_df.max()[trafo_loading_df.max()>TRAFO_LOADING]
trafo_filtered = trafo_loading_df[trafo80.index]
plt.boxplot(trafo_filtered, patch_artist=True, boxprops=dict(facecolor=r, color=g), capprops=dict(color=r), whiskerprops=dict(color=r),
                flierprops=dict(color=r, markeredgecolor=g), medianprops=dict(color=k))
# set titles
fig.suptitle('Probabilistic Analysis', fontsize=14)
ax3.set_title('Statistic Data for Transformer Loading (filter for transformer loading > {}%)'.format(TRAFO_LOADING), fontsize=11)
ax3.set(xlabel='Transformers')
ax3.set(ylabel='Loading [%]')
ax3.set_xticklabels(trafo_filtered.columns)
ax3.tick_params(labelrotation=45)
fig.tight_layout(h_pad=1)
fig.savefig('PlotBoxTrf.png')

'''
#plot box for grid losses
'''
fig = plt.figure(figsize=(12, 7))
#flierprops = dict(marker='o', markerfacecolor='tab:blue', markersize=5,
                  #linestyle='none')
ax1 = fig.add_subplot(1,1,1)
gl_filtered = grid_losses_df.drop(['Summary Grid'], axis=1)
#gl_filtered.plot.box(flierprops=flierprops, ax=ax1)
c = 'blue'
plt.boxplot(gl_filtered, patch_artist=True, boxprops=dict(facecolor=b, color=g), capprops=dict(color=b), whiskerprops=dict(color=b),
                flierprops=dict(color=b, markeredgecolor=g), medianprops=dict(color=k))
fig.suptitle('Probabilistic Analysis', fontsize=14)
ax1.set_title('Power Losses', fontsize=11)
ax1.set(xlabel='Summary Grid')
ax1.set(ylabel='Power Losses [MW]')
ax1.set_xticklabels(gl_filtered.columns)
ax1.tick_params(labelrotation=45)
fig.tight_layout(h_pad=1)
fig.savefig('PlotBoxLosses.png')

#cumulative and probabilistic diagrams
pp = PdfPages('Line_loading_dist.pdf')
for line in list(lne_loading_df):
    plot1 = hist(lne_loading_df[line], 20, line, 'Loading of Overehad Line')
    pp.savefig(plot1)
    plt.close()
pp.close()

pp = PdfPages('Voltage_dist.pdf')
for bus in list(voltage_filtered):
    plot1 = hist(voltage_filtered[bus], 20, bus, 'Voltage at Grid Node')
    pp.savefig(plot1)
    plt.close()
pp.close()

pp = PdfPages('Transformer_loading_dist.pdf')
for trafo in list(trafo_loading_df):
    plot1 = hist(trafo_loading_df[trafo], 20, trafo, 'Loading of Transformer')
    pp.savefig(plot1)
    plt.close()
pp.close()

pp = PdfPages('pv power output.pdf')
for vre in list(vrepv_power_df):
    plot1 = hist(vrepv_power_df[vre], 50, vre, 'pv power')
    pp.savefig(plot1)
    plt.close()
pp.close()

pp = PdfPages('wind power output.pdf')
for vre in list(vrewi_power_df):
    plot1 = hist(vrewi_power_df[vre], 50, vre, 'wind power')
    pp.savefig(plot1)
    plt.close()
pp.close()

pp = PdfPages('Grid losses_dist.pdf')
for grid in list(grid_losses_df):
    plot1 = hist(grid_losses_df[grid], 50, grid, 'grid losses [MW]')
    pp.savefig(plot1)
    plt.close()
pp.close()

'''
#save pdf file for subplots at all buses
#scatter and cumulative probability
'''
# create subplots
corr_bv_rp = pd.DataFrame()
pp = PdfPages('Bus_Voltage_subplots.pdf')
for i in range (0, len(voltage_df.columns), 2):
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2,2,1)
    plt.scatter(vre_power_tot, voltage_filtered.iloc[:, i], color='tab:blue', s=2)

    x = vre_power_tot.values
    y = voltage_filtered.iloc[:, i].values
    slope, intercept, r, p, std_err = stats.linregress(x, y)

    # plot linear regression

    linmodel = list(map(lin, vre_power_tot))
    plt.plot(x, linmodel, color='darkslategray', label="Slope = {}\n"
                                                       "Correlation = {}\n"
                                                       "p-value = {}".format(
        slope * vre_power_tot.values.max(), r, p))


    corr_bv_rp = corr_bv_rp.append({'bus': voltage_filtered.columns[i], 'm': slope * vre_power_tot.values.max(), 'r': r, 'p': p}, ignore_index=True)
    ax2 = fig.add_subplot(2,2,2)
    values, base, _ = plt.hist(voltage_filtered.iloc[:, i], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
    ax2_bis = ax2.twinx()
    values = np.append(values,0)
    ax2_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )
    if i != len(voltage_df.columns)-1:
        ax3 = fig.add_subplot(2,2,3)
        plt.scatter(vre_power_tot, voltage_filtered.iloc[:, i+1], color='tab:blue', s=2)
        x2 = vre_power_tot.values
        y2 = voltage_filtered.iloc[:, i+1].values
        slope2, intercept2, r2, p2, std_err2 = stats.linregress(x2, y2)

        # plot linear regression
        linmodel2 = list(map(lin2, vre_power_tot))
        plt.plot(x2, linmodel2, color='darkslategray', label="Slope = {}\n"
                                                           "Correlation = {}\n"
                                                           "p-value = {}".format(
            slope2 * vre_power_tot.values.max(), r2, p2))

        corr_bv_rp = corr_bv_rp.append({'bus': voltage_filtered.columns[i+1], 'm': slope2 * vre_power_tot.values.max(), 'r': r2, 'p': p2}, ignore_index=True)
        ax4 = fig.add_subplot(2,2,4)
        values, base, _ = plt.hist(voltage_filtered.iloc[:, i+1], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
        ax4_bis = ax4.twinx()
        values = np.append(values,0)
        ax4_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )

    #set titles and labels
    fig.suptitle('Probabilistic Analysis - Bus Voltages', fontsize=14)
    ax1.set_title('Voltage at {} node vs VRE power'.format(str(voltage_filtered.columns[i])), fontsize=11)
    ax1.legend()
    ax1.set(xlabel='Total VRE Power [MW]')
    ax1.set(ylabel='Voltage [p.u.]')
    ax2.set_title(str(voltage_filtered.columns[i]), fontsize=11)
    ax2.set(xlabel='Voltage [p.u.]')
    ax2.set(ylabel='Proportion')
    ax2_bis.set(ylabel='Proportion')
    ax2_bis.legend();
    ax2.legend();
    if i != len(voltage_df.columns)-1:
        ax3.set_title('Voltage at {} node vs VRE power'.format(str(voltage_filtered.columns[i+1])), fontsize=11)
        ax3.legend()
        ax3.set(xlabel='Total VRE Power [MW]')
        ax3.set(ylabel='Voltage [p.u.]')
        ax4.set_title(str(voltage_filtered.columns[i+1]), fontsize=11)
        ax4.set(xlabel='Voltage [p.u.]')
        ax4.set(ylabel='Proportion')
        ax4_bis.legend();
        ax4.legend();
    fig.tight_layout(h_pad=1)
    pp.savefig(fig)
    plt.close()
pp.close()

'''
# plot correlation bars for bus voltages divided by clusters'''


#plot correlation bars for buses

fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(1,1,1)
colors = {0: '#38ad94', 1: '#ff493a', 2: '#4eafc9', 3: '#ffe4a5', 4: '#6495ed', 5: '#ff68be', 6: '#bada55', 7: '#f7af00', 8: '#decaf1', 9: '#bdc6c7'}
corr_bv_rp = corr_bv_rp.set_index(keys='bus')
print(corr_bv_rp)
if CLUSTERS:
    cluster_sorted = cluster_df.sort_values(by=0, axis=1)
    corr_bv_rp_sorted = corr_bv_rp.reindex(cluster_sorted.columns)
    corr_bv_rp_sri = corr_bv_rp_sorted.reset_index()
    cluster_sorted_t = cluster_sorted.T
    corr_bv_rp_sri = corr_bv_rp_sri.join(cluster_sorted_t.iloc[:,0], on='index')
    corr_bv_rp_sri.plot(x='index', y=['m'], kind='bar',
                        color=[colors.get(x, '#333333') for x in corr_bv_rp_sri.iloc[:,4].values], legend=None, ax=ax1) # this line plot the colors for clusters
else:
    corr_bv_rp = corr_bv_rp.reset_index()
    print(corr_bv_rp)
    corr_bv_rp.plot(x='bus', y=['m','r'], kind='bar', ax=ax1)

fig.suptitle('Probabilistic Analysis', fontsize=14)
ax1.set_title('Correlations Slopes of Bus Voltage vs VRE power', fontsize=11)

if CLUSTERS:
    ax1.set(xlabel='Nodes (colors by clusters)')
else:
    ax1.set(xlabel='Nodes')
ax1.set(ylabel='p.u.')
fig.tight_layout(h_pad=1)
fig.savefig('Bus_voltage_correlations.png')

'''
#save pdf file of subplots for all lines
#scatter and cumulative probability
'''
# create subplots
corr_ll_rp = pd.DataFrame()
pp = PdfPages('Lne_Loading_subplots.pdf')
for i in range (0, len(lne_loading_df.columns), 2):
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2,2,1)
    plt.scatter(vre_power_tot, lne_loading_df.iloc[:, i], color='tab:red', s=2)

    x = vre_power_tot.values
    y = lne_loading_df.iloc[:, i].values
    slope, intercept, r, p, std_err = stats.linregress(x, y)

    # plot linear regression

    linmodel = list(map(lin, vre_power_tot))
    plt.plot(x, linmodel, color='darkslategray', label="Slope = {}\n"
                                                       "Correlation = {}\n"
                                                       "p-value = {}".format(
        slope * vre_power_tot.values.max()/100, r, p))

    corr_ll_rp = corr_ll_rp.append({'line': lne_loading_df.columns[i], 'm': slope * vre_power_tot.values.max()/100, 'r': r, 'p': p}, ignore_index=True)
    ax2 = fig.add_subplot(2,2,2)
    values, base, _ = plt.hist(lne_loading_df.iloc[:, i], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
    ax2_bis = ax2.twinx()
    values = np.append(values,0)
    ax2_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )
    if i != len(lne_loading_df.columns)-1:
        ax3 = fig.add_subplot(2,2,3)
        plt.scatter(vre_power_tot, lne_loading_df.iloc[:, i+1], color='tab:red', s=2)
        x2 = vre_power_tot.values
        y2 = lne_loading_df.iloc[:, i+1].values
        slope2, intercept2, r2, p2, std_err2 = stats.linregress(x2, y2)

        # plot linear regression
        linmodel2 = list(map(lin2, vre_power_tot))
        plt.plot(x2, linmodel2, color='darkslategray', label="Slope = {}\n"
                                                           "Correlation = {}\n"
                                                           "p-value = {}".format(
            slope2 * vre_power_tot.values.max()/100, r2, p2))

        corr_ll_rp = corr_ll_rp.append({'line': lne_loading_df.columns[i+1], 'm': slope2 * vre_power_tot.values.max()/100, 'r': r2, 'p': p2}, ignore_index=True)
        ax4 = fig.add_subplot(2,2,4)
        values, base, _ = plt.hist(lne_loading_df.iloc[:, i+1], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
        ax4_bis = ax4.twinx()
        values = np.append(values,0)
        ax4_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )

    #set titles and labels
    fig.suptitle('Probabilistic Analysis - Loading of Lines', fontsize=14)
    ax1.set_title('{} Line Loading vs VRE Power'.format(str(lne_loading_df.columns[i])), fontsize=11)
    ax1.legend()
    ax1.set(xlabel='Total VRE Power [MW]')
    ax1.set(ylabel='Loading [%]')
    ax2.set_title(str(lne_loading_df.columns[i]), fontsize=11)
    ax2.set(xlabel='Loading [%]')
    ax2.set(ylabel='Proportion')
    ax2_bis.set(ylabel='Proportion')
    ax2_bis.legend();
    ax2.legend();
    if i != len(lne_loading_df.columns)-1:
        ax3.set_title('{} Line Loading vs VRE Power'.format(str(lne_loading_df.columns[i+1])), fontsize=11)
        ax3.legend()
        ax3.set(xlabel='Total VRE Power [MW]')
        ax3.set(ylabel='Loading [%]')
        ax4.set_title(str(lne_loading_df.columns[i+1]), fontsize=11)
        ax4.set(xlabel='Loading [%]')
        ax4.set(ylabel='Proportion')
        ax4_bis.legend();
        ax4.legend();
    fig.tight_layout(h_pad=1)
    pp.savefig(fig)
    plt.close()

#plot correlation bars for line loading
corr_ll_rp_sorted = corr_ll_rp.sort_values(by='m')
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
corr_ll_rp_sorted.head(10).plot(x='line', y=['m','r'], kind='bar', ax=ax1)
corr_ll_rp_sorted.tail(10).plot(x='line', y=['m', 'r'], kind='bar', ax=ax2)
fig.suptitle('Probabilistic Analysis', fontsize=14)
ax1.set_title('Correlations and Slopes of Line Loading vs VRE power', fontsize=11)
ax1.legend()
ax1.set(xlabel='Lines (first ten elements with negative highest correlation slopes)')
ax1.set(ylabel='p.u.')
ax2.set_title('Correlations and Slopes of Line Loading vs VRE power', fontsize=11)
ax2.legend()
ax2.set(xlabel='Lines (first ten elements with positive highest correlation slopes)')
ax2.set(ylabel='p.u.')
fig.tight_layout(h_pad=1)
print(corr_ll_rp_sorted)
fig.tight_layout(h_pad=1)
fig.savefig('Line_loading_correlations.png')
pp.savefig(fig)
pp.close()

'''
#save pdf file of subplots for all transformers
#scatter and cumulative probability
'''
# create subplots
corr_lt_rp = pd.DataFrame()
pp = PdfPages('Trf_Loading_subplots.pdf')
for i in range (0, len(trafo_loading_df.columns), 2):
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2,2,1)
    plt.scatter(vre_power_tot, trafo_loading_df.iloc[:, i], color='tab:red', s=2)

    x = vre_power_tot.values
    y = trafo_loading_df.iloc[:, i].values
    slope, intercept, r, p, std_err = stats.linregress(x, y)

    # plot linear regression
    linmodel = list(map(lin, vre_power_tot))
    plt.plot(x, linmodel, color='darkslategray', label="Slope = {}\n"
                                                       "Correlation = {}\n"
                                                       "p-value = {}".format(
        slope * vre_power_tot.values.max()/100, r, p))

    corr_lt_rp = corr_lt_rp.append({'trf': trafo_loading_df.columns[i], 'm': slope * vre_power_tot.values.max()/100, 'r': r, 'p': p}, ignore_index=True)
    ax2 = fig.add_subplot(2,2,2)
    values, base, _ = plt.hist(trafo_loading_df.iloc[:, i], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
    ax2_bis = ax2.twinx()
    values = np.append(values,0)
    ax2_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )
    if i != len(trafo_loading_df.columns)-1:
        ax3 = fig.add_subplot(2,2,3)
        plt.scatter(vre_power_tot, trafo_loading_df.iloc[:, i+1], color='tab:red', s=2)

        x2 = vre_power_tot.values
        y2 = trafo_loading_df.iloc[:, i+1].values
        slope2, intercept2, r2, p2, std_err2 = stats.linregress(x2, y2)

        # plot linear regression

        linmodel2 = list(map(lin2, vre_power_tot))
        plt.plot(x2, linmodel2, color='darkslategray', label="Slope = {}\n"
                                                           "Correlation = {}\n"
                                                           "p-value = {}".format(
            slope2 * vre_power_tot.values.max()/100, r2, p2))

        corr_lt_rp = corr_lt_rp.append({'trf': trafo_loading_df.columns[i+1], 'm': slope2 * vre_power_tot.values.max()/100, 'r':r2, 'p': p2}, ignore_index=True)
        ax4 = fig.add_subplot(2,2,4)
        values, base, _ = plt.hist(trafo_loading_df.iloc[:, i+1], bins=20, density=True, alpha=0.5, color="tab:green", range=None, label="Histogram")
        ax4_bis = ax4.twinx()
        values = np.append(values,0)
        ax4_bis.plot(base, np.cumsum(values)/ np.cumsum(values)[-1], color='tab:orange', marker='o', linestyle='-', markersize=1, label="Cumulative Histogram" )

    #set titles and labels
    fig.suptitle('Probabilistic Analysis - Loading of Transformers', fontsize=14)
    ax1.set_title('{} Transformer Loading vs VRE Power'.format(str(trafo_loading_df.columns[i])), fontsize=11)
    ax1.legend()
    ax1.set(xlabel='Total VRE Power [MW]')
    ax1.set(ylabel='Loading [%]')
    ax2.set_title(str(trafo_loading_df.columns[i]), fontsize=11)
    ax2.set(xlabel='Loading [%]')
    ax2.set(ylabel='Proportion')
    ax2_bis.set(ylabel='Proportion')
    ax2_bis.legend();
    ax2.legend();
    if i != len(trafo_loading_df.columns)-1:
        ax3.set_title('{} Transformer Loading vs VRE Power'.format(str(trafo_loading_df.columns[i+1])), fontsize=11)
        ax3.legend()
        ax3.set(xlabel='Total VRE Power [MW]')
        ax3.set(ylabel='Loading [%]')
        ax4.set_title(str(trafo_loading_df.columns[i+1]), fontsize=11)
        ax4.set(xlabel='Loading [%]')
        ax4.set(ylabel='Proportion')
        ax4_bis.legend();
        ax4.legend();
        fig.tight_layout(h_pad=1)
    pp.savefig(fig)
    plt.close()

#plot correlation bars
corr_lt_rp_sorted = corr_lt_rp.sort_values(by='m')
fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
corr_lt_rp_sorted.head(10).plot(x='trf', y=['m', 'r'], kind='bar', ax=ax1)
corr_lt_rp_sorted.tail(10).plot(x='trf', y=['m', 'r'], kind='bar', ax=ax2)
fig.suptitle('Probabilistic Analysis', fontsize=14)
ax1.set_title('Correlations and Slopes of Transformer Loading vs VRE Power', fontsize=11)
ax1.legend()
ax1.set(xlabel='Transformers (first ten elements with negative highest correlation slopes)')
ax1.set(ylabel='p.u.')
ax2.set_title('Correlations and Slopes of Transformer Loading vs VRE Power', fontsize=11)
ax2.legend()
ax2.set(xlabel='Transformers (first ten elements with positive highest correlation slopes)')
ax2.set(ylabel='p.u.')
print(corr_lt_rp_sorted)
fig.tight_layout(h_pad=1)
fig.savefig('Trf_loading_correlations.png')
pp.savefig(fig)
pp.close()

if PLOT_FIGURES:
  plt.show()