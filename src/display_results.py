import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import time
import pyVHR as vhr

from pyVHR.utils.errors import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


results_path = 'results/2023-09-28/K1_raw_patch_CHROM.json'
df = pd.read_json(results_path)
groundtruth_path = 'results/groundtruth.json'
gt_df = pd.read_json(groundtruth_path)

#Pick a specific instance
instance = 0

times = np.array(df['TIMES'][instance])
bpms = np.array(df['BPMS'][instance])
bvps = np.array(df['BVPS'][instance])

window = 0
window_bpms = np.array(bpms[window])
window_bvps = np.array(bvps[window])

# Process the groundtruth data now
gt_bpms = gt_df['ECG_BPM'][instance]
gt_times = gt_df['ECG_TIMES'][instance]
f = np.interp(gt_times, gt_times, gt_bpms)
ref_hrs = np.interp(times, gt_times, f)

########## Calculating the metrics

bpms.shape
ref_hrs.shape

bpms[0]-ref_hrs[0]

errors = []
for idx, ref_hr in enumerate(ref_hrs):
    error = bpms[idx] - ref_hr
    errors.append(error)

errors = np.array(errors)
mean_errors = np.mean(errors,axis=0).reshape(50,50)
median_errors = np.median(errors,axis=0).reshape(50,50)
sns.heatmap(mean_errors, norm=LogNorm())
plt.show()

def get_region_snr(bvp,fps,ref_hr):
    # Set up some variables
    interv1 = 0.2*60
    interv2 = 0.2*60
    NyquistF = fps/2.;
    FResBPM = 0.5
    nfft = np.ceil((60*2*NyquistF)/FResBPM)
    bvp = np.expand_dims(bvp,axis=0) # Expand the dimensions of the bvp so we can calculate the periodogram
    pfreqs, power = Welch(bvp,fps,nfft=nfft)
    power = np.squeeze(power,axis=0) # change the shape of the power so we can apply the boolean mask

    # Define the masks that is within 12bpm of the actual hr
    GTMask1 = np.logical_and(pfreqs>=ref_hr-interv1, pfreqs<=ref_hr+interv1)
    GTMask2 = np.logical_and(pfreqs>=(ref_hr*2)-interv2, pfreqs<=(ref_hr*2)+interv2)
    GTMask = np.logical_or(GTMask1, GTMask2)
    FMask = np.logical_not(GTMask)
    SPower = np.sum(power[GTMask])
    allPower = np.sum(power[FMask])
    snr = 10*np.log10(SPower/allPower)
    return snr

def get_window_snrs(window_bvps,fps,ref_hr):
    window_snrs = []
    for bvp in window_bvps:
        snr = get_region_snr(bvp,fps,ref_hr)
        window_snrs.append(snr)
    window_snrs = np.array(window_snrs)
    return window_snrs

def get_instance_snrs(bvps,fps,ref_hrs):
    instance_snrs = []
    for idx, window_bvps in enumerate(bvps):
        window_snrs = get_window_snrs(window_bvps,fps,ref_hrs[idx])
        instance_snrs.append(window_snrs)
    instance_snrs = np.array(instance_snrs)
    return instance_snrs

sig_extractor = vhr.extraction.SignalProcessing()
sig_extractor.landmarks = np.array(range(2500))
sig_extractor.landmarks.shape[0]
wsize = 8
fps = 30
ma = vhr.extraction.MotionAnalysis(sig_extractor, wsize, fps,np.array(range(2500))
psd_bpms = vhr.BPM.BPM_clustering(ma, bvps, fps, wsize, movement_thrs=None, opt_factor=0.5)





instance_snrs = get_instance_snrs(bvps,30,ref_hrs)
patch_means = np.mean(instance_snrs, axis = 0)
patch_medians = np.median(instance_snrs, axis = 0)
patch_stds = np.std(instance_snrs,axis = 0)

median_hm = sns.heatmap(patch_medians.reshape(50,50))
mean_hm = sns.heatmap(patch_means.reshape(50,50))
std_hm = sns.heatmap(patch_stds.reshape(50,50))
plt.show()

# plot a heatmap of the window_bpms