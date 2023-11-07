import pyVHR as vhr
from pyVHR.BVP import *
from pyVHR.utils.errors import *
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm

patch_size = 10
overlap = 0

results_path = 'results'
instance_df_path = os.path.join(results_path, 'instance_info.json')

instance_df = pd.read_json(instance_df_path)

instances = instance_df['INSTANCE'].values
k1_paths = instance_df['K1_PATH'].values
k2_paths = instance_df['K2_PATH'].values

### Initialise the sig extractor
sig_extractor = vhr.extraction.SignalProcessing()
sig_extractor.set_visualize_skin_and_landmarks( visualize_skin=True,
                                                    visualize_landmarks=True, 
                                                    visualize_landmarks_number=False, 
                                                    visualize_patch=True)
sig_extractor.choose_cuda_device(0)
sig_extractor.set_skin_extractor(vhr.extraction.SkinExtractionRectangle('GPU')) # Set the skin extractor
sig_extractor.set_square_patches_side(patch_size)
sig_extractor.set_overlap(overlap)
sig_extractor.thickness = 1
sig_extractor.font_size = 0.3

### Set RGB thresholding to ignore everything
