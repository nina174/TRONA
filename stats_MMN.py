# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:08:13 2023

@author: ehrhardtn
"""

#from os import chdir, listdir
from os.path import join, isfile
import numpy as np
import mne
import random
from mne.stats import f_mway_rm, f_threshold_mway_rm
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from math import sqrt

outdir = ("Y:\\01_Studien\\29_TRONA\\Analysen_und_Ergebnisse\\")

## first prepare data

std_all_na = np.array([])
dev_all_na = np.array([])
std_all_tr = np.array([])
dev_all_tr = np.array([])

#participants = [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,31,33,35,37,39] #all participants except 1 (no EOGs) and 11 (epoching doesn't work)
participants = [4,5,6,7,8,9,12,13,14,15,17,18,19,20,21,22,23,24,25,29,31,33,37,39] # only participants with >70% trials retained

for sub in participants:
    if isfile(join(outdir, "Nass",f"{sub}_NA_TOENE_clean_150_epo.fif")) and isfile(join(outdir, "Trocken Artefact Corrected",f"{sub}_TR_TOENE_clean_150_epo.fif")):
        
        # load epochs from normal EEG
        epochs = mne.read_epochs(join(outdir, "Nass", f"{sub}_NA_TOENE_clean_150_epo.fif"))       
        
        # average trials with standard tone and apply baseline correction
        stand = epochs['standard2'].average(picks='eeg').apply_baseline(baseline=(-0.1,0))
        
        stand = np.mean((stand.data), axis=0) # average over channels
        stand = np.expand_dims(stand, axis = 0) # add second dimension to stack participants
        
        # stack averages of standard tones in one array for all participants
        if len(std_all_na) == 0:
            std_all_na = stand
        else:
            std_all_na = np.row_stack((std_all_na, stand))
        
        # average trials with deviant tones and apply baseline correction
        dev = epochs['deviant1', 'deviant2'].average(picks='eeg').apply_baseline(baseline=(-0.1,0))
        
        dev = np.mean((dev.data), axis=0) # average over channels
        dev = np.expand_dims(dev, axis = 0) # add second dimension to stack participants
        
        # stack averages of deviant tones from one channel in one array for all participants
        if len(dev_all_na) == 0:
            dev_all_na = dev
        else:
            dev_all_na = np.row_stack((dev_all_na, dev))
            
        # load epochs from dry EEG            
        epochs = mne.read_epochs(join(outdir, "Trocken Artefact Corrected", f"{sub}_TR_TOENE_clean_150_epo.fif"))       
        
        # average trials with standard tone and apply baseline correction
        stand = epochs['standard2'].average(picks='eeg').apply_baseline(baseline=(-0.1,0))
        
        stand = np.mean((stand.data), axis=0) # average over channels
        stand = np.expand_dims(stand, axis = 0) # add second dimension to stack participants
        
        # stack averages of standard tones from one channel in one array for all participants
        if len(std_all_tr) == 0:
            std_all_tr = stand
        else:
            std_all_tr = np.row_stack((std_all_tr, stand))
        
        # average trials with deviant tones and apply baseline correction
        dev = epochs['deviant1', 'deviant2'].average(picks='eeg').apply_baseline(baseline=(-0.1,0))
        
        dev = np.mean((dev.data), axis=0) # average over channels
        dev = np.expand_dims(dev, axis = 0) # add second dimension to stack participants
        
        # stack averages of deviant tones from one channel in one array for all participants
        if len(dev_all_tr) == 0:
            dev_all_tr = dev
        else:
            dev_all_tr = np.row_stack((dev_all_tr, dev))
            
## actual statistics

# ANOVA for interaction

project_seed = 999 # RNG seed
random.seed(project_seed) # set seed to ensure computational reproducibility


factor_levels = [2, 2]
n_replications = len(participants)
effects = "A:B" #A*B for all effects, A:B for interaction effect

data = [std_all_na, std_all_tr, dev_all_na, dev_all_tr]

def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]

# The ANOVA returns a tuple f-values and p-values, we will pick the former.
pthresh = 0.05  
f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effects,
                               pthresh)
tail = 1  # f-test, so tail > 0
n_permutations = 1000
F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
    data, stat_fun=stat_fun, threshold=f_thresh, tail=tail,
    n_jobs=None, n_permutations=n_permutations, buffer_size=None,
    out_type='mask',
    seed = project_seed)

# plot interaction

times = epochs.times
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax.set_title("Contrast (Deviant tones - Standard tones) all channels")

ax.plot(times, dev_all_na.mean(axis=0) - std_all_na.mean(axis=0),
        label="Wet", color = "palevioletred")
ax.plot(times, dev_all_tr.mean(axis=0) - std_all_tr.mean(axis=0),
        label="Dry", color = "lightsteelblue")
ax.set_ylabel("Amplitude")
ax.legend()

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = ax2.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)

hf = plt.plot(times, F_obs, 'g')
ax2.legend((h, ), ('cluster p-value < 0.05', ))
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("f-values")

# extract values

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        print(f"cluster no.{i_c}")
        print(f"timepoints: {times[c.start]}-{times[c.stop-1]} ms")
        print(f"clustersize: {c.stop-c.start}")
        print(f"mean F-value: {F_obs[c.start:c.stop].mean()}")
        print(f"p-value: {cluster_p_values[i_c]}")
   
        
# plot ERPs with CI

ci_na = 1.96 * np.std(dev_all_na.mean(axis=0) - std_all_na.mean(axis=0))/sqrt(len(dev_all_na))
ci_tr = 1.96 * np.std(dev_all_tr.mean(axis=0) - std_all_tr.mean(axis=0))/sqrt(len(dev_all_tr))

times = epochs.times
fig, ax = plt.subplots()
ax.set_title("Contrast (Deviant tones - Standard tones) all channels")

ax.plot(times, dev_all_na.mean(axis=0) - std_all_na.mean(axis=0),
        label="Wet", color = "palevioletred")
ax.fill_between(times, ((dev_all_na.mean(axis=0) - std_all_na.mean(axis=0))-ci_na),
                ((dev_all_na.mean(axis=0) - std_all_na.mean(axis=0))+ci_na), 
                color = "palevioletred", alpha = .1)  

ax.plot(times, dev_all_tr.mean(axis=0) - std_all_tr.mean(axis=0),
        label="Dry", color = "lightsteelblue")
ax.fill_between(times, ((dev_all_tr.mean(axis=0) - std_all_tr.mean(axis=0))-ci_tr),
                ((dev_all_tr.mean(axis=0) - std_all_tr.mean(axis=0))+ci_tr),
                color = "lightsteelblue", alpha = .1)  
ax.set_ylabel("Amplitude")
ax.legend()
plt.ylim(ymin = -0.000003, ymax = 0.000003)
plt.gca().invert_yaxis()

# follow-up tests

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        dev_na = dev_all_na[:,c.start:c.stop].mean(axis=1)
        std_na = std_all_na[:,c.start:c.stop].mean(axis=1)
        result = ttest_rel(dev_na, std_na)
        print(f"cluster no.{i_c}")
        print(f"timepoints: {times[c.start]}-{times[c.stop-1]} ms")
        print(f"results wet EEG: {result.statistic}, corrected p-value{result.pvalue*(len(clusters)*2)}")
        
        dev_tr = dev_all_tr[:,c.start:c.stop].mean(axis=1)
        std_tr = std_all_tr[:,c.start:c.stop].mean(axis=1)
        result = ttest_rel(dev_tr, std_tr)
        print(f"cluster no.{i_c}")
        print(f"timepoints: {times[c.start]}-{times[c.stop-1]} ms")
        print(f"results dry EEG: t-value: {result.statistic}, corrected p-value: {result.pvalue*(len(clusters)*2)}")
        

# plot interaction with bar plots
tps = np.where(np.logical_and(epochs.times>=0.09, epochs.times<=0.11))

standard_means, standard_std = (std_all_na[:,tps].mean(), std_all_tr[:,tps].mean()), (std_all_na[:,tps].std()/sqrt(24), std_all_tr[:,tps].std()/sqrt(24))
deviant_means, deviant_std = (dev_all_na[:,tps].mean(), dev_all_tr[:,tps].mean()), (dev_all_na[:,tps].std()/sqrt(24), dev_all_tr[:,tps].std()/sqrt(24))

ind = np.arange(len(standard_means))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, standard_means, width, yerr=standard_std,
                label='standard')
rects2 = ax.bar(ind + width/2, deviant_means, width, yerr=deviant_std,
                label='deviant')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Amplitude')
ax.set_title('')
ax.set_xticks(ind)
ax.set_xticklabels(('Nass', 'Trocken'))
ax.legend()
ax.invert_yaxis()

plt.show()
