# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:50:40 2023

@author: ehrhardtn
"""

from os import chdir
from os.path import join
import numpy as np
import pandas as pd
import mne

montage_dir = ("Y:\\01_Studien\\29_TRONA\\Allgemeines\\Info_Trocken_EEG\\")
chdir(montage_dir)

elec_pos = join(montage_dir, "ANTWaveguard_pos.csv")

elec_df = pd.read_csv(elec_pos,delimiter=";") # x- und y-Koordinaten vertauscht
elec_df["xpos"] = (elec_df)["xpos"]*(-1) # rechts und links sind vertauscht -> invert x
elec_df = elec_df[128:198]
elec_dict = {}
for idx, row in elec_df.iterrows():
    elec_dict[row["Label"]] = np.array([row["xpos"],
                                        row["ypos"],
                                        row["zpos"]])
    elec_dict[row["Label"]] *= 1e-3

digmon = mne.channels.make_dig_montage(ch_pos=elec_dict)

ch_names = digmon.ch_names

# open file in write mode
with open(join(montage_dir, "ANTWaveguard_pos_chnames.csv"), "w") as fp:
    for ch_name in ch_names:
        # write each item on a new line
        fp.write("%s\n" % ch_name)

digmon.save("montage_ANTWaveguard.fif", overwrite=True)
