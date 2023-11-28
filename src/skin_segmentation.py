import sys
sys.path.append('../')
import pandas as pd
import os
import pyVHR as vhr
import cv2
from tqdm import tqdm


instance_path = "C:/Users/20759193/source/repos/pyVHR/results/instance_info.json"

df = pd.read_json(instance_path)

def extract_and_save_frames(row,cam):
    vid_path = row[cam +"_PATH"]
    if vid_path is not None:
        sig_extractor = vhr.extraction.SignalProcessing()
        sig_extractor.choose_cuda_device(0)
        sig_extractor.set_skin_extractor(vhr.extraction.SkinExtractionRectangle('GPU'))
        sig_extractor.set_visualize_skin_and_landmarks( visualize_skin=True,
                                                        visualize_landmarks=True, 
                                                        visualize_landmarks_number=False, 
                                                        visualize_patch=True)

        sig_extractor.skin_extractor.rect = row[cam+"_SKIN_RECTANGLE_COORDS"]
        x, y, w, h = sig_extractor.skin_extractor.rect
        bottom_right_x = x + w
        bottom_right_y = y + h

        sig_extractor.skin_extractor.mean_rgb = row[cam+"_SKIN_RECTANGLE_MEAN_RGB"]
        skin_threshold = row[cam+"_PIXEL_THRESHOLD"]
        hol_sig = sig_extractor.extract_holistic_rectangle(vid_path,skin_threshold)


        raw_frame = cv2.cvtColor(sig_extractor.display_frame, cv2.COLOR_BGR2RGB)
        segmented_frame = cv2.cvtColor(sig_extractor.display_skin_frame, cv2.COLOR_BGR2RGB)

        raw_frame_path = os.path.join(os.path.split(vid_path)[0],"raw_frame.png")
        cv2.imwrite(raw_frame_path, raw_frame)
        segmented_frame_path = os.path.join(os.path.split(vid_path)[0],"segmented_frame.png")
        cv2.imwrite(segmented_frame_path, segmented_frame)
        
        raw_frame_overlay = cv2.rectangle(raw_frame, (x, y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        segmented_frame_overlay = cv2.rectangle(segmented_frame, (x, y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        raw_frame_overlay_path = os.path.join(os.path.split(vid_path)[0],"raw_frame_overlay.png")
        cv2.imwrite(raw_frame_overlay_path, raw_frame_overlay)
        segmented_frame_overlay_path = os.path.join(os.path.split(vid_path)[0],"segmented_frame_overlay.png")
        cv2.imwrite(segmented_frame_overlay_path, segmented_frame_overlay)


for index, row in tqdm(df.iterrows()):

    extract_and_save_frames(row,"K1")
    extract_and_save_frames(row,"K2")