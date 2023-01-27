# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:15:42 2023

@author: ehrhardtn
"""

from os import chdir, listdir
from os.path import join
import mne

## Import data ##############################################################

data_dir = ("Y:\\01_Studien\\29_TRONA\\Daten\\")
out_dir = ("Y:\\01_Studien\\29_TRONA\\Analysen_und_Ergebnisse\\")

system = ["Nass", "Trocken Artefact Corrected"]
condition = ["TOENE", "REEG1", "REEG2"]

for sys in system:
    
    sys_dir = join(data_dir, f"{sys}")
    chdir(sys_dir)
    
    for cond in condition:
             
        filenames = listdir(sys_dir)
        
        for filename in filenames:
            if filename[-5:] != ".vhdr" or f"{cond}" not in filename:
                continue
            
            file = filename.partition("202")    
                        
            eeg_data = mne.io.read_raw_fif(join(out_dir, f"{file[0]}raw.fif"))
            
            ## Filter the data            
            eeg_data.filter(l_freq=0.1, h_freq = 25, picks=['eog', 'eeg'])
            
            ## Trial definition
            if cond=="TOENE":
                events, event_id = mne.events_from_annotations(eeg_data, regexp='Stimulus')
            
                event_dict = {'standard1': 10001, 'standard2': 10002,
                          'deviant1': 10003, 'deviant2': 10004}
            
                #decim = np.round(eeg_data.info['sfreq']/512) #actual sampling frequency divided by desired sampling frequency to get decim
            
                epochs = mne.Epochs(eeg_data, events, event_id=event_dict, 
                                    tmin=-0.1, tmax=0.5, preload=True, baseline=None) #automatic baseline correction from tmin to 0, but can be customized with the baseline parameter)
                
            else:
                rs_data = eeg_data.crop(tmin=eeg_data.annotations.onset[2], tmax=eeg_data.annotations.onset[3], include_tmax=True)
                epochs = mne.make_fixed_length_epochs(rs_data, duration=8, preload=True)
            
            epochs.resample(512, npad='auto', window='boxcar')
                
            ## Plot epochs and manually reject noisy trials
            
            #mne.Epochs.plot(epochs, picks='all', n_channels=len(epochs.ch_names), scalings=25e-6, events=events)
            #epochs.drop_bad()
            
            sub = filename.partition("_")
            sub = sub[0]
                                   
            with open(join(out_dir, f"{sys}", f'{sub}_NA_badch.txt'), 'r') as f:
                bad_ch = [line.rstrip('\n') for line in f]
            
            epochs.info['bads'] = bad_ch
            
            ## Save preprocessed data
            
            epochs.save(join(out_dir, f"{sys}", f"{file[0]}epo.fif"), overwrite=True)
                