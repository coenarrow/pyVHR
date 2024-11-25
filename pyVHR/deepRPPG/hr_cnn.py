import numpy as np
import torch
import torchvision.transforms as transforms
import time
from collections import OrderedDict
from torch.utils.data import DataLoader
from .HR_CNN.utils import butter_bandpass_filter
from .HR_CNN.PulseDataset import PulseDataset
from .HR_CNN.FaceHRNet09V4ELU import FaceHRNet09V4ELU
import os
import requests

def HR_CNN_bvp_pred(frames:list,model_path:str=None) -> np.ndarray:
    print("initialize model...")

    # Get the directory of the current module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the model file relative to the module directory
    if model_path is None:
        model_path = os.path.join(module_dir, 'hr_cnn_model.pth')
    if not os.path.isfile(model_path):
      url = "https://github.com/phuselab/pyVHR/raw/master/resources/deepRPPG/hr_cnn_model.pth"
      print(f'Downloading HR_CNN model to {model_path}...')
      r = requests.get(url, allow_redirects=True)
      open(model_path, 'wb').write(r.content)   

    model = FaceHRNet09V4ELU(rgb=True)

    model = torch.nn.DataParallel(model)

    model.cuda()

    ss = sum(p.numel() for p in model.parameters())
    print('num params: ', ss)

    state_dict = torch.load(model_path)

    new_state_dict = OrderedDict()
    # original saved file with DataParallel
    for k, v in state_dict.items():
        new_state_dict['module.' + k] = v

    model.load_state_dict(new_state_dict)

    pulse_test = PulseDataset(frames, transform=transforms.ToTensor())

    val_loader = DataLoader(
        pulse_test,
        batch_size=128, shuffle=False, pin_memory=True, drop_last=True)

    model.eval()

    outputs = []

    start = time.time()
    for i, net_input in enumerate(val_loader):
        net_input = net_input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(net_input)
            outputs.append(output.squeeze())

    end = time.time()
    print("processing time: ", end - start)

    outputs = torch.cat(outputs)

    outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)

    outputs = np.array(outputs.tolist())

    return outputs

def filter_HR_CNN_outputs(bvp_outputs:np.ndarray,
                          fps:float,
                          min_bpm:float,
                          max_bpm:float,
                          order:int=4) -> np.ndarray:
    """
    Filter the HR_CNN outputs using a bandpass filter
    
    Parameters
    ----------
    bvp_outputs : np.ndarray
        The HR_CNN outputs to filter
    fps : float
        The frames per second of the video
    min_bpm : float
        The minimum BPM to filter
    max_bpm : float
        The maximum BPM to filter
    order : int
        The order of the Butterworth filter

    Returns
    -------
    np.ndarray
        The filtered outputs
    """
    lowcut = min_bpm/60
    highcut = max_bpm/60

    filtered_outputs = butter_bandpass_filter(bvp_outputs, lowcut, highcut, fps, order=order)
    normalized_outputs = np.array(filtered_outputs - np.mean(filtered_outputs)) / np.std(filtered_outputs)

    return np.array(normalized_outputs)