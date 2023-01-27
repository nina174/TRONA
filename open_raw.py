# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:41:08 2023

@author: ehrhardtn
"""

from os import chdir, listdir
from os.path import isfile, join
import mne

## Import data ##############################################################

data_dir = ("Y:\\01_Studien\\29_TRONA\\Daten\\")
out_dir = ("Y:\\01_Studien\\29_TRONA\\Analysen_und_Ergebnisse\\")

system = ["Nass"]
condition = ["TOENE"]
preposts = ["1", "2"]

for sys in system:
    
    sys_dir = join(data_dir, f"{sys}")
    chdir(sys_dir)
        
    for cond in condition:
             
        filenames = listdir(sys_dir)
            
        for filename in filenames:
            if filename[-5:] != ".vhdr" or "_A_" in filename or "_B_" in filename:
                continue
            
            eeg_data = mne.io.read_raw_brainvision(filename, preload=True)
            
            if "Nass" in sys:
                # Set channel types
                eeg_data.set_channel_types(mapping = {'vEOG': 'eog', 'HEOG': 'eog', 'VEOG2': 'eog'})
                
                ## Reference data to average of mastoids
                eeg_data.set_eeg_reference(ref_channels=['M1', 'M2'], ch_type='eeg')
                
                ## Set the montage
                eeg_data.set_montage('standard_1005')
            
            else: 
                # Set channel types
                eeg_data.set_channel_types(mapping = {'BIP1': 'eog', 'BIP2': 'eog'}) # BIP1 = horizontal, BIP2 = vertical (left) 
                
                # Reference data to average of mastoids
                eeg_data.set_eeg_reference(ref_channels=['3LD', '3RD'], ch_type='eeg')
                
                ## Set the montage
                digmon = mne.channels.read_dig_fif("Y:\\01_Studien\\29_TRONA\\Allgemeines\\Info_Trocken_EEG\\montage_ANTWaveguard.fif")  
                eeg_data.set_montage(digmon)
                
            sub = filename.partition("_")
            sub = sub[0]
                            
            if isfile(join(out_dir, f"{sys}", f'{sub}_NA_badch.txt')):
                with open(join(out_dir, f"{sys}", f'{sub}_NA_badch.txt'), 'r') as f:
                    bad_ch = [line.rstrip('\n') for line in f]
    
                eeg_data.info['bads'] = bad_ch
                
            else:
                eeg_data.plot(duration=10, n_channels=len(eeg_data.ch_names), scalings=25e-6) #scale=50uV, all channels, block=True pauses script while plot is open
           
                bad_ch = eeg_data.info['bads']
        
                with open(join(out_dir, f"{sys}", f'{sub}_NA_badch.txt'), 'w') as f:
                    for c in bad_ch:
                        f.write(str(c) + '\n')

        