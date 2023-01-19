# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 09:11:38 2022

@author: ehrhardtn
"""

import os
from os.path import isfile
import numpy as np
import mne
import glob
from mne.preprocessing import (ICA)

## Import data ##############################################################

raw_folder = ('Y:\\01_Studien\\29_TRONA\\Daten\\Nass\\')
results_folder = ('Y:\\01_Studien\\29_TRONA\\Analysen_und_Ergebnisse\\Nass\\')
os.chdir(results_folder)
#condition = ["TOENE"]

if isfile('preprocessing_150.npy'):
    table = np.load('preprocessing_150.npy')
else:
    table = np.array([['ID', 'no. rejected channel', 'no. rejected components', '% rejected trials']])

for sub in range(29,41): # (1,41) to loop through participants 1 to 40


    if isfile(glob.glob(os.path.join(raw_folder, f'{sub}_NA_TOENE*.vhdr'))[0]):
        raw_file = glob.glob(os.path.join(raw_folder, f'{sub}_NA_TOENE*.vhdr'))[0]
        eeg_data = mne.io.read_raw_brainvision(raw_file, preload=True)
        eeg_data.set_channel_types(mapping = {'vEOG': 'eog', 'HEOG': 'eog', 'VEOG2': 'eog'}) # für nass-EEG
        #print(raw.info)
        
        ## Reference data to average of mastoids
        eeg_data.set_eeg_reference(ref_channels=['M1', 'M2'], ch_type='eeg')
        
        if isfile(f'{sub}_NA_badch.txt'):
            with open(f'{sub}_NA_badch.txt', 'r') as f:
                bad_ch = [line.rstrip('\n') for line in f]
        
            eeg_data.info['bads'] = bad_ch
            #print(bad_ch)
            
        else: 
            eeg_data.plot(duration=10, n_channels=len(eeg_data.ch_names), scalings=25e-6) #scale=50uV, all channels, block=True pauses script while plot is open
               
            bad_ch = eeg_data.info['bads']
            
            with open(f'{sub}_NA_badch.txt', 'w') as f:
                for c in bad_ch:
                    f.write(str(c) + '\n')
        
        
        
        
    
    
