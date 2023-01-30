# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:25:20 2023

@author: ehrhardtn
"""

from os import chdir, listdir
from os.path import join, isfile
import numpy as np
import mne

## Import data ##############################################################

data_dir = ("Y:\\01_Studien\\29_TRONA\\Daten\\")
out_dir = ("Y:\\01_Studien\\29_TRONA\\Analysen_und_Ergebnisse\\")

#system = ["Nass", "Trocken Artefact Corrected"]
#condition = ["TOENE", "REEG1", "REEG2"]

system = ["Nass"]
condition = ["TOENE"]

for sys in system:
    
    chdir(join(out_dir, f"{sys}"))
    
    filenames = listdir()
    
    for cond in condition:
        
        if isfile(f"{cond}_preprocessing_150.npy"):
            table = np.load(f"{cond}_preprocessing_150.npy")
        else: 
            table = np.array([['ID', 'no. rejected channel', '% rejected trials']])
    
        for filename in filenames:
                
            if f"{cond}_ica_epo.fif" not in filename:
                continue
        
            epochs = mne.read_epochs(filename)
            
            reject_criteria = dict(eeg=150e-6)       # 100 µV
            clean_epochs = epochs.copy()
            clean_epochs.drop_bad(reject=reject_criteria)
            #clean_epochs.plot_drop_log()
            # print(clean_epochs.drop_log)
            
            rej_trls = [n for n, dl in enumerate(clean_epochs.drop_log) if len(dl)] # find indices of rejected trials
            if rej_trls:
                rej_trls = np.array(rej_trls)
                rej_trls = np.c_[rej_trls, epochs.events[rej_trls,2]] # add event marker of rejected trials
            
            #mne.Epochs.plot(clean_epochs, picks='all', n_channels=len(epochs.ch_names), scalings=25e-6, events=events)
            
            sub = filename.partition("_")
            sub = sub[0]
                                  
            table = np.append(table,[[sub, len(epochs.info['bads']), 100*(len(rej_trls)/len(epochs.events))]], axis = 0)
            
            file = filename.partition("ica_epo.fif")    
                          
            clean_epochs.save(f"{file[0]}clean_150_epo.fif", overwrite=True)
            np.savetxt(f"{file[0]}rejtrls_150.txt", rej_trls, fmt='%d')
            
        np.save(f"{cond}_preprocessing_150.npy", table)
            
        #np.save('preprocessing_150.npy', table)
        #np.savetxt('preprocessing_150.npy', table, fmt='%d')