import csv
from SimComp import MontecarloLoadFlow
import pandas as pd

RESULT_PATH = r"D:\temp1\OneDrive - ENGIE\Python scripts\ProbAnalysis\VersionCopula\Example\Results"
SAMPLE_PATH = r"D:\temp1\OneDrive - ENGIE\Python scripts\ProbAnalysis\VersionCopula\Example\SampleData"
FOLDER_NAME = ''
PROJECT_NAME = 'IEEE_PDO_MyArticle'
STUDY_CASE_NAME = '2024'
N_SAMPLES = 8760
STD_DEV = 0.09
DAY = False # if True, it considers only day time with production for PV

# activate both project and study case
sim = MontecarloLoadFlow(
    folder_name=FOLDER_NAME,
    project_name=PROJECT_NAME,
    study_case_name=STUDY_CASE_NAME)

#create montecarlo load flow iterable object
mcldf = pd.DataFrame(sim.monte_carlo_loadflow(N_SAMPLES, SAMPLE_PATH, STD_DEV, DAY))
# #create csv file to store voltages

with open(RESULT_PATH + r'\res_prob_lf_bus.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get voltages

 for row_index, voltages in enumerate(mcldf[0]):

     #write file header (bus names)
     if row_index == 0:
         csvwriter = csv.DictWriter(csvfile, voltages.keys())
         csvwriter.writeheader()
     #write file rows (voltages)
     csvwriter.writerow(voltages)

with open(RESULT_PATH + r'\res_prob_lf_lne.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get line loading

 for row_index, lines in enumerate(mcldf[1]):

     #write file header (line names)
     if row_index == 0:
         csvwriter = csv.DictWriter(csvfile, lines.keys())
         csvwriter.writeheader()
     #write file rows (lines)
     csvwriter.writerow(lines)

with open(RESULT_PATH + r'\res_prob_trf.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get transformer loading

 for row_index, trafo in enumerate(mcldf[2]):

     #write file header (transformer names)
     if row_index == 0:
         csvwriter = csv.DictWriter(csvfile, trafo.keys())
         csvwriter.writeheader()
     #write file rows (transformers)
     csvwriter.writerow(trafo)

with open(RESULT_PATH + r'\res_prob_lf_1000_std0.1_p.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get pq random values
 for row_index, p in enumerate(mcldf[3]):
     if row_index == 0:
        csvwriter = csv.DictWriter(csvfile, p.keys())
        csvwriter.writeheader()
     # write file rows (active power random numbers)
     csvwriter.writerow(p)

# power losses (total of the grid)
with open(RESULT_PATH + r'\res_prob_lf_1000_std0.1_power_losses.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get pq random values

 for row_index, pl in enumerate(mcldf[4]):
     if row_index == 0:
        csvwriter = csv.DictWriter(csvfile, pl.keys())
        csvwriter.writeheader()
     # write file rows (reactive power random numbers)
     csvwriter.writerow(pl)

# power vre (each)
with open(RESULT_PATH + r'\res_prob_lf_1000_std0.1_power_vre_wind.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get wind power output

 for row_index, rewind in enumerate(mcldf[5]):
     if row_index == 0:
        csvwriter = csv.DictWriter(csvfile, rewind.keys())
        csvwriter.writeheader()
     # write file rows (reactive power random numbers)
     csvwriter.writerow(rewind)

with open(RESULT_PATH + r'\res_prob_lf_1000_std0.1_power_vre_pv.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get pv power output

 for row_index, repv in enumerate(mcldf[6]):
     if row_index == 0:
        csvwriter = csv.DictWriter(csvfile, repv.keys())
        csvwriter.writeheader()
     # write file rows (reactive power random numbers)
     csvwriter.writerow(repv)

with open(RESULT_PATH + r'\res_prob_lf_1000_std0.1_lne_pw.csv', 'w', newline='') as csvfile:
#iterate over mcldf object to get line loading

 for row_index, lines in enumerate(mcldf[7]):


     #write file header (line names)
     if row_index == 0:
         csvwriter = csv.DictWriter(csvfile, lines.keys())
         csvwriter.writeheader()
     #write file rows (lines)
     csvwriter.writerow(lines)