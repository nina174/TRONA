# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:41:08 2023

@author: ehrhardtn
"""

from os import chdir, listdir
from os.path import join
import mne
from glob import glob

## Import data ##############################################################

data_dir = ("Y:\\01_Studien\\29_TRONA\\Daten\\")
out_dir = ("Y:\\01_Studien\\29_TRONA\\Analysen_und_Ergebnisse\\")

#system = ["Nass", "Trocken Artefact Corrected"]
#condition = ["TOENE", "REEG1", "REEG2"]

system = ["Trocken Artefact Corrected"]
condition = ["TOENE"]

for sys in system:
    
    sys_dir = join(data_dir, f"{sys}")
    chdir(sys_dir)
        
    for cond in condition:
             
        filenames = listdir(sys_dir)
            
        for filename in filenames:
            if filename[-5:] != ".vhdr" or f"{cond}" not in filename:
                continue
            
            eeg_data = mne.io.read_raw_brainvision(filename, preload=True)
            
            if "Nass" in sys:
                
                if "vEOG" in eeg_data.info["ch_names"]: 
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
               
                with open("Y:\\01_Studien\\29_TRONA\\Allgemeines\\Info_Trocken_EEG\\ANTWaveguard_pos_chnames.csv", "r") as fp:
                    ch_names = [line.rstrip('\n') for line in fp]
                    
                digmon.ch_names = ch_names                    
                
                eeg_data.set_montage(digmon)
                
            sub = filename.partition("_")
            sub = sub[0]
            
            if glob(join(out_dir, f"{sys}", f"{sub}*badch.txt")) != []:
                ch_file = glob(join(out_dir, f"{sys}", f"{sub}*badch.txt"))[0]
                with open(ch_file, 'r') as f:
                    bad_ch = [line.rstrip('\n') for line in f]
        
                eeg_data.info['bads'] = bad_ch
                    
            else:
                eeg_data.plot(duration=10, n_channels=len(eeg_data.ch_names), scalings=25e-6, block = True) #scale=50uV, all channels, block=True pauses script while plot is open
       
                bad_ch = eeg_data.info['bads']
    
                with open(join(out_dir, f"{sys}", f'{sub}_badch.txt'), 'w') as f:
                    for c in bad_ch:
                        f.write(str(c) + '\n')
                        
            file = filename.partition("202")    
                        
            eeg_data.save(join(out_dir, f"{sys}", f"{file[0]}raw.fif"), overwrite=True)

        