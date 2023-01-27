# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:50:09 2023

@author: ehrhardtn
"""

from os import chdir, listdir
from os.path import join, isfile
import mne
from mne.preprocessing import (ICA)

## Import data ##############################################################

data_dir = ("Y:\\01_Studien\\29_TRONA\\Daten\\")
out_dir = ("Y:\\01_Studien\\29_TRONA\\Analysen_und_Ergebnisse\\")

system = ["Nass", "Trocken Artefact Corrected"]
condition = ["TOENE", "REEG"]

for sys in system:
    
    chdir(join(out_dir, f"{sys}"))
    
    filenames = listdir()
        
    for filename in filenames:
        if filename[-7:] != "epo.fif":
            continue
        
        epochs = mne.read_epochs(filename)
        
        if sys == "Nass": 
            
            # # rereferencing EOG
            if any(ele == 'vEOG' for ele in epochs.info['bads']):
                epochs_bip_ref = epochs
                epochs_bip_ref = mne.set_bipolar_reference(epochs_bip_ref, anode=['HEOG'], cathode=['F7'], drop_refs=False) # ???
                epochs_bip_ref.drop_channels('HEOG')
            else:
                epochs_bip_ref = mne.set_bipolar_reference(epochs, anode=['vEOG'], cathode=['VEOG2']) # vertical EOG
                epochs_bip_ref = mne.set_bipolar_reference(epochs_bip_ref, anode=['HEOG'], cathode=['F7'], drop_refs=False) # ???
                epochs_bip_ref.drop_channels('HEOG')
                                
        else:
            
            if any(ele == 'BIP1' for ele in epochs.info['bads']):
                epochs_bip_ref = epochs
                epochs_bip_ref = mne.set_bipolar_reference(epochs_bip_ref, anode=['BIP2'], cathode=['1L'], drop_refs=False) # ???
                epochs_bip_ref.drop_channels('BIP2')
            else:
                epochs_bip_ref = mne.set_bipolar_reference(epochs, anode=['BIP2'], cathode=['1L'], drop_refs=False) # vertical EOG (anode-cathode)
                epochs_bip_ref = mne.set_bipolar_reference(epochs_bip_ref, anode=['BIP1'], cathode=['1LD'], drop_refs=False) # ???
                epochs_bip_ref.drop_channels(['BIP1', 'BIP2'])
                
 # ICA, correlation and rejection, check if it worked
         
        file = filename.partition("epo.fif")    
 
        if isfile(f"{file[0]}ica.fif"):
            ica=mne.preprocessing.read_ica(f"{file[0]}ica.fif")
        else:
            ica = ICA(n_components=15, max_iter='auto', random_state=97)
            ica.fit(epochs_bip_ref)
            ica.save(f"{file[0]}ica.fif", overwrite=True)
 
        #ica.plot_sources(epochs_bip_ref, show_scrollbars=False) #right clicking on the name of the component will bring up a plot of its properties
        #ica.plot_components(inst=epochs_bip_ref) #clicking on component will open properties window
 
        eog_indices, eog_scores = ica.find_bads_eog(epochs_bip_ref, measure = 'correlation', threshold = 0.2)
        #ica.plot_scores(eog_scores) # barplot of ICA component "EOG match" scores
        ica.exclude = eog_indices
 
        reconst_epochs = epochs_bip_ref.copy()
        ica.apply(reconst_epochs)
 
        ## Interpolation of bad channels
        reconst_epochs.interpolate_bads(reset_bads=True, method={"eeg":"spline"}, verbose="INFO")
    
        epochs.save(f"{file[0]}epo_ica.fif", overwrite=True)