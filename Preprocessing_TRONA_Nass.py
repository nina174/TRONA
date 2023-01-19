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

        ## Set the montage
        eeg_data.set_montage('standard_1005')
        #eeg_data.plot_sensors(show_names=True, sphere='eeglab') # sieht verschoben aus?!! TO DO 
        
        ## Filter the data 
        
        eeg_data.filter(l_freq=0.1, h_freq = 25, picks=['eog', 'eeg'])
        
        ## Trial definition
        
        events, event_id = mne.events_from_annotations(eeg_data, regexp='Stimulus')
        
        event_dict = {'standard1': 10001, 'standard2': 10002,
                      'deviant1': 10003, 'deviant2': 10004}
        
        decim = np.round(eeg_data.info['sfreq']/200) #actual sampling frequency divided by desired sampling frequency to get decim
        
        epochs = mne.Epochs(eeg_data, events, event_id=event_dict, 
                            tmin=-0.1, tmax=0.5, preload=True, baseline=None,
                            decim = decim) #automatic baseline correction from tmin to 0, but can be customized with the baseline parameter)
        
        ## Plot epochs and manually reject noisy trials
        
        #mne.Epochs.plot(epochs, picks='all', n_channels=len(epochs.ch_names), scalings=25e-6, events=events)
        #epochs.drop_bad()
        
        epochs.info['bads'] = bad_ch
        
        ## Save preprocessed data
        
        #epochs.save(f'{sub}_NA_epo.fif', overwrite=True)
        
        # epochs=mne.read_epochs(f'{sub}_NA_epo.fif')
        
        # # rereferencing EOG
        
        if any(ele == 'vEOG' for ele in epochs.info['bads']):
            epochs_bip_ref = epochs
            epochs_bip_ref = mne.set_bipolar_reference(epochs_bip_ref, anode=['HEOG'], cathode=['F7'], drop_refs=False) # ???
            epochs_bip_ref.drop_channels('HEOG')
        else:
            epochs_bip_ref = mne.set_bipolar_reference(epochs, anode=['vEOG'], cathode=['VEOG2']) # vertical EOG
            epochs_bip_ref = mne.set_bipolar_reference(epochs_bip_ref, anode=['HEOG'], cathode=['F7'], drop_refs=False) # ???
            epochs_bip_ref.drop_channels('HEOG')
        # ICA, correlation and rejection, check if it worked
        
        if isfile(f'{sub}_NA_ica.fif'):
            ica=mne.preprocessing.read_ica(f'{sub}_NA_ica.fif')
        else:
            ica = ICA(n_components=15, max_iter='auto', random_state=97)
            ica.fit(epochs_bip_ref)
            ica.save(f'{sub}_NA_ica.fif', overwrite=True)
        
        #ica.plot_sources(epochs_bip_ref, show_scrollbars=False) #right clicking on the name of the component will bring up a plot of its properties
        #ica.plot_components(inst=epochs_bip_ref) #clicking on component will open properties window
        
        eog_indices, eog_scores = ica.find_bads_eog(epochs_bip_ref, measure = 'correlation', threshold = 0.2)
        #ica.plot_scores(eog_scores) # barplot of ICA component "EOG match" scores
        ica.exclude = eog_indices
        
        reconst_epochs = epochs_bip_ref.copy()
        ica.apply(reconst_epochs)
        
        ## Interpolation of bad channels
        reconst_epochs.interpolate_bads(reset_bads=True, method={"eeg":"spline"}, verbose="INFO")
        
        #mne.Epochs.plot(reconst_epochs, picks='all', n_channels=len(epochs.ch_names), scalings=25e-6, events=events)
        
        # reconst_epochs.plot(picks='eog')
        # epochs_bip_ref.plot(picks='eog')
        
        # automatic trial rejection
        
        reject_criteria = dict(eeg=150e-6)       # 100 µV
        clean_epochs = reconst_epochs.copy()
        clean_epochs.drop_bad(reject=reject_criteria)
        #clean_epochs.plot_drop_log()
        # print(clean_epochs.drop_log)
        
        rej_trls = [n for n, dl in enumerate(clean_epochs.drop_log) if len(dl)] # find indices of rejected trials
        if rej_trls:
            rej_trls = np.array(rej_trls)
            rej_trls = np.c_[rej_trls, events[rej_trls,2]] # add event marker of rejected trials
        
        #mne.Epochs.plot(clean_epochs, picks='all', n_channels=len(epochs.ch_names), scalings=25e-6, events=events)
        
        table = np.append(table,[[sub, len(bad_ch), len(eog_indices), 100*(len(rej_trls)/len(epochs.events))]], axis = 0)
        
        clean_epochs.save(f'{sub}_NA_clean_150_epo.fif', overwrite=True)
        np.savetxt(f'{sub}_NA_rejtrls_150.txt', rej_trls, fmt='%d')
        
    else:
        continue

np.save('preprocessing_150.npy', table)
np.savetxt('preprocessing_150.npy', table, fmt='%d')
        
        
        
        
    
    
