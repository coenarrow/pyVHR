import numpy as np
import tensorflow as tf
from .MTTS_CAN.model import Attention_mask, MTTS_CAN
from scipy.signal import butter
import cv2
from skimage.util import img_as_float
import scipy.io
from scipy.sparse import spdiags
import h5py
import os
import requests

def preprocess_raw_video(frames, fs=30, dim=36):
  """A slightly different version from the original: 
    takes frames as input instead of video path """

  totalFrames = frames.shape[0]
  Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
  i = 0
  t = []
  width = frames.shape[2]
  height = frames.shape[1]
  # Crop each frame size into dim x dim
  for img in frames:
    t.append(1/fs*i)       # current timestamp in milisecond
    img = img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]
    vidLxL = cv2.resize(img_as_float(img), (dim, dim), interpolation = cv2.INTER_AREA)
    vidLxL[vidLxL > 1] = 1
    vidLxL[vidLxL < (1/255)] = 1/255
    Xsub[i, :, :, :] = vidLxL
    i = i + 1

  # Normalized Frames in the motion branch
  normalized_len = len(t) - 1
  dXsub = np.zeros((normalized_len, dim, dim, 3), dtype = np.float32)
  for j in range(normalized_len - 1):
    dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])
  dXsub = dXsub / np.std(dXsub)
  
  # Normalize raw frames in the apperance branch
  Xsub = Xsub - np.mean(Xsub)
  Xsub = Xsub  / np.std(Xsub)
  Xsub = Xsub[:totalFrames-1, :, :, :]
  
  # Plot an example of data after preprocess
  dXsub = np.concatenate((dXsub, Xsub), axis=3);
  return dXsub

def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def MTTS_CAN_deep(frames:list, 
                  fs:float, 
                  model_checkpoint:str=None, 
                  batch_size:int=100, 
                  dim:int=36, 
                  img_rows:int=36, 
                  img_cols:int=36, 
                  frame_depth:int=10, 
                  verb:int=0):
  """
  This function applies the MTTS-CAN model to a video sequence to estimate the heart rate.
  
  Parameters
  ----------
  frames : list
      The list of frames of the video sequence.
  fs : float
      The frame rate of the video sequence.
  model_checkpoint : str, optional
      The path to the pretrained model checkpoint. If None, the model will be downloaded from the repository.
  batch_size : int, optional
      The batch size for the model.
  dim : int, optional
      The dimension of the frames.
  img_rows : int, optional
      The number of rows of the frames.
  img_cols : int, optional
      The number of columns of the frames.
  frame_depth : int, optional
      How many frames to use for each prediction.
  verb : int, optional
      The verbosity of the model.
  """
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
     for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

  if model_checkpoint is None:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    model_checkpoint = os.path.join(module_dir, 'mtts_can_model.hdf5')
    if not os.path.isfile(model_checkpoint):
      url = "https://github.com/phuselab/pyVHR/raw/master/resources/deepRPPG/mtts_can_model.hdf5"
      print(f'Downloading MTTS_CAN model to {model_checkpoint}...')
      r = requests.get(url, allow_redirects=True)
      open(model_checkpoint, 'wb').write(r.content)   

  # frame preprocessing
  dXsub = preprocess_raw_video(frames, fs=fs, dim=dim)
  dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
  dXsub = dXsub[:dXsub_len, :, :, :]

  # load pretrained model
  model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
  model.load_weights(model_checkpoint)

  # apply pretrained model
  yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=verb)

  # filtering
  pulse_pred = yptest[0]
  pulse_pred = detrend(np.cumsum(pulse_pred), 100)
  return pulse_pred

def filter_MTTS_CAN_outputs(raw_bvp:np.ndarray,
                            fps:float,
                            min_bpm:int,
                            max_bpm:int,
                            order:int=6) -> np.ndarray:
    """
    This function filters the outputs of the MTTS-CAN model.
    
    Parameters
    ----------
    bvp_outputs : np.ndarray
        The raw BVP outputs of the model.
    fps : float
        The frame rate of the video sequence.
    min_bpm : int
        The minimum BPM for the filter.
    max_bpm : int
        The maximum BPM for the filter.
    order : int, optional
        The order of the butterworth bandpass filter.
    """
    low_cutoff = min_bpm / 60
    high_cutoff = max_bpm / 60

    # filter the outputs
    [b_pulse, a_pulse] = butter(order, [low_cutoff / fps * 2, high_cutoff / fps * 2], btype='bandpass')
    filtered_bvp = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(raw_bvp))
    return filtered_bvp