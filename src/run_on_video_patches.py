import pyVHR as vhr
from pyVHR.BVP import *
from pyVHR.utils.errors import *
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm


# Want to use the raw frames first
## For specific instance
seg_frames = False
camera = 'K1'
 
# Define our processing parameters class

class ProcessingParameters:
    
    def __init__(self):
        self.wsize = 8
        self.stride = 1
        self.low_thr = 0
        self.high_thr = 255
        self.low_hz = 0.5
        self.high_hz = 4
        self.seconds = 0
        self.patch_size = 10.
        self.patch_overlap = 0
        self.skin_threshold = 40
        vhr.extraction.SkinProcessingParams.RGB_LOW_TH =  0
        vhr.extraction.SkinProcessingParams.RGB_HIGH_TH = 255
        vhr.extraction.SignalProcessingParams.RGB_LOW_TH = 0
        vhr.extraction.SignalProcessingParams.RGB_HIGH_TH = 255

processing_params = ProcessingParameters()
if seg_frames:
    segmented = 'segmented'
else:
    segmented = 'raw'

# set up the function for defining our signal extractor
def initialize_sig_extractor(videoFilename, mean_rgb):
    sig_extractor = vhr.extraction.SignalProcessing()
    sig_extractor.set_visualize_skin_and_landmarks( visualize_skin=True,
                                                    visualize_landmarks=True, 
                                                    visualize_landmarks_number=False, 
                                                    visualize_patch=True)
    sig_extractor.choose_cuda_device(0)
    sig_extractor.set_skin_extractor(vhr.extraction.SkinExtractionRectangle('GPU')) # Set the skin extractor
    sig_extractor.set_square_patches_side(processing_params.patch_size)
    sig_extractor.set_overlap(processing_params.patch_overlap)
    sig_extractor.thickness = 1
    sig_extractor.font_size = 0.3
    fps = vhr.extraction.get_fps(videoFilename)
    sig_extractor.set_total_frames(fps*processing_params.seconds)
    sig_extractor.skin_extractor.mean_rgb = mean_rgb
    return sig_extractor


def window_deep_bvps(full_bvps):
    windowed_bvps = []
    for patch_bvp in tqdm(full_bvps):
        HR_CNN_bvps, _ = BVP_windowing(patch_bvp, processing_params.wsize, fps, stride=processing_params.stride)
        windowed_bvps.append(HR_CNN_bvps)

    windowed_bvps = np.array(windowed_bvps)
    windowed_bvps = windowed_bvps.reshape(windowed_bvps.shape[1],windowed_bvps.shape[0],windowed_bvps.shape[3])
    windowed_bvps = [window for window in windowed_bvps]

    return windowed_bvps

## Set up where we want to store the dataframes once they're processed
date = '2023-09-28'
results_path = 'results'
df_dir = os.path.join(results_path, date)
os.makedirs(df_dir, exist_ok=True)

# Set up the dataframe filenames
# Make the dataframe name
#Done CHROM_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_CHROM.json')
#Done LGI_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_LGI.json')
#Done POS_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_POS.json')
#PBV_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_PBV.json')
#PCA_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_PCA.json')
#Done GREEN_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_GREEN.json')
#Done OMIT_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_OMIT.json')
#ICA_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_ICA.json')
#HR_CNN_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_HR_CNN.json')
MTTS_CAN_df_fname = os.path.join(df_dir, f'{camera}_{segmented}_patch_MTTS_CAN.json')

# Make the dataframe for the methods
#Done CHROM_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#Done LGI_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#Done POS_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#PBV_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#PCA_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#Done GREEN_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#Done OMIT_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#ICA_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
#Done HR_CNN_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})
MTTS_CAN_results_df = pd.DataFrame(columns = {'INSTANCE','TIMES', 'BPMS', 'BVPS'})

# Let's get the instance dataframes and get ready to process
instance_df_path = os.path.join(results_path, 'instance_info.json')
instance_df = pd.read_json(instance_df_path)

for row in tqdm(range(len(instance_df))):
    row = 0
    instance = instance_df.iloc[row]['INSTANCE']
    videopath = instance_df.iloc[row][f'{camera}_PATH']
    if videopath == None:
        print(f'\nInstance {instance} has no video')
        continue
    mean_rgb = instance_df.iloc[row][f'{camera}_SKIN_RECTANGLE_MEAN_RGB']

    # Initialize the signal extractor
    sig_extractor = initialize_sig_extractor(videopath, mean_rgb)
    fps = vhr.extraction.get_fps(videopath)

    # Set the fixed patches
    hol_sig = sig_extractor.extract_holistic_rectangle(videopath,processing_params.skin_threshold,segmented_frames=seg_frames)
    sig_extractor.set_fixed_patches(videopath,region_type="squares",overlap=processing_params.patch_overlap)
    sig_extraction_method = "mean"

    # Get the patch_signal
    patch_sig = sig_extractor.extract_fixed_patches(sig_extraction_method,segmented_frames = seg_frames)

    # Get the frames for the deep learning methods
    if seg_frames:
        frames = np.array(sig_extractor.visualize_skin_collection)
    else:
        frames = sig_extractor.extract_raw(videopath)

    # Apply the filters
    windowed_patch_sig, timesES = vhr.extraction.sig_windowing(patch_sig, processing_params.wsize, processing_params.stride, fps)
    filtered_windowed_patch_sig = vhr.BVP.apply_filter(windowed_patch_sig, vhr.BVP.BPfilter, params={'order':6,'minHz':processing_params.low_hz,'maxHz':processing_params.high_hz,'fps':fps})
    
    # Apply the specific methods to get the BVPs
    #Done CHROM_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cuda', method=cupy_CHROM)
    #Done LGI_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cpu', method=cpu_LGI)
    #Done POS_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cuda', method=cupy_POS, params={'fps':fps})
    #PBV_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cpu', method=cpu_PBV)
    #PCA_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cpu', method=cpu_PCA, params={'component':'all_comp'})
    #Done GREEN_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cpu', method=cpu_GREEN)
    #Done OMIT_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cpu', method=cpu_OMIT)
    #ICA_bvps = RGB_sig_to_BVP(filtered_windowed_patch_sig, fps, device_type='cpu', method=cpu_ICA, params={'component':'all_comp'})

    # Get the BVPs for the deep learning methods
    # get the frames for the patch
    #HR_CNN_full_bvps = []
    MTTS_CAN_full_bvps = []
    for fixed_patch in tqdm(sig_extractor.fixed_patches):
        frames = fixed_patch.raw_frames
    #    HR_CNN_bvp = vhr.deepRPPG.HR_CNN_bvp_pred(frames,verb=0,filter_pred=False)
        MTTS_CAN_bvp = vhr.deepRPPG.MTTS_CAN_deep(frames, fps, verb=0,filter_pred=False)
    #    HR_CNN_full_bvps.append(HR_CNN_bvp)
        MTTS_CAN_full_bvps.append(MTTS_CAN_bvp)
    #HR_CNN_full_bvps = np.array(HR_CNN_full_bvps)
    #HR_CNN_bvps = window_deep_bvps(HR_CNN_full_bvps)
    MTTS_CAN_full_bvps = np.array(MTTS_CAN_full_bvps)
    MTTS_CAN_bvps = window_deep_bvps(MTTS_CAN_full_bvps)

    #HR_CNN_bvps = vhr.BVP.apply_filter(HR_CNN_bvps, vhr.BVP.BPfilter, params={'order':6,'minHz':processing_params.low_hz,'maxHz':processing_params.high_hz,'fps':fps})
    MTTS_CAN_bvps = vhr.BVP.apply_filter(MTTS_CAN_bvps, vhr.BVP.BPfilter, params={'order':6,'minHz':processing_params.low_hz,'maxHz':processing_params.high_hz,'fps':fps})

    # Get the BPM estimates
    #Done CHROM_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(CHROM_bvps, fps)).tolist()
    #Done LGI_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(LGI_bvps, fps)).tolist()
    #Done POS_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(POS_bvps, fps)).tolist()
    #PBV_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(PBV_bvps, fps)).tolist()
    #PCA_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(PCA_bvps, fps)).tolist()
    #Done GREEN_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(GREEN_bvps, fps)).tolist()
    #Done OMIT_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(OMIT_bvps, fps)).tolist()
    #ICA_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(ICA_bvps, fps)).tolist()
    #Done HR_CNN_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(HR_CNN_bvps, fps)).tolist()
    MTTS_CAN_bpmES = np.array(vhr.BPM.BVP_to_BPM_cuda(MTTS_CAN_bvps, fps)).tolist()

    #Done CHROM_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [CHROM_bpmES], 'BVPS':[CHROM_bvps]})
    #Done LGI_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [LGI_bpmES], 'BVPS':[LGI_bvps]})
    #Done POS_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [POS_bpmES], 'BVPS':[POS_bvps]})
    #PBV_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [PBV_bpmES], 'BVPS':[PBV_bvps]})
    #PCA_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [PCA_bpmES], 'BVPS':[PCA_bvps]})
    #Done GREEN_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [GREEN_bpmES], 'BVPS':[GREEN_bvps]})
    #Done OMIT_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [OMIT_bpmES], 'BVPS':[OMIT_bvps]})
    #ICA_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [ICA_bpmES], 'BVPS':[ICA_bvps]})
    #Done HR_CNN_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [HR_CNN_bpmES], 'BVPS':[HR_CNN_bvps]})
    MTTS_CAN_row_results = pd.DataFrame({'INSTANCE':instance,'TIMES': [timesES], 'BPMS': [MTTS_CAN_bpmES], 'BVPS':[MTTS_CAN_bvps]})

    #Done CHROM_results_df = pd.concat([CHROM_results_df, CHROM_row_results], ignore_index=True)
    #Done LGI_results_df = pd.concat([LGI_results_df, LGI_row_results], ignore_index=True)
    #Done POS_results_df = pd.concat([POS_results_df, POS_row_results], ignore_index=True)
    #PBV_results_df = pd.concat([PBV_results_df, PBV_row_results], ignore_index=True)
    #PCA_results_df = pd.concat([PCA_results_df, PCA_row_results], ignore_index=True)
    #Done GREEN_results_df = pd.concat([GREEN_results_df, GREEN_row_results], ignore_index=True)
    #Done OMIT_results_df = pd.concat([OMIT_results_df, OMIT_row_results], ignore_index=True)
    #ICA_results_df = pd.concat([ICA_results_df, ICA_row_results], ignore_index=True)
    #DoneHR_CNN_results_df = pd.concat([HR_CNN_results_df, HR_CNN_row_results], ignore_index=True)
    MTTS_CAN_results_df = pd.concat([MTTS_CAN_results_df, MTTS_CAN_row_results], ignore_index=True)
    
# Save the df
#Done CHROM_results_df.to_json(CHROM_df_fname)
#Done LGI_results_df.to_json(LGI_df_fname)
#Done POS_results_df.to_json(POS_df_fname)
#PBV_results_df.to_json(PBV_df_fname)
#PCA_results_df.to_json(PCA_df_fname)
#Done GREEN_results_df.to_json(GREEN_df_fname)
#Done OMIT_results_df.to_json(OMIT_df_fname)
#ICA_results_df.to_json(ICA_df_fname)
#Done HR_CNN_results_df.to_json(HR_CNN_df_fname)
MTTS_CAN_results_df.to_json(MTTS_CAN_df_fname)
