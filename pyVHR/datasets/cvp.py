import pandas as pd
from biosppy.signals import ecg
from pyVHR.datasets.dataset import Dataset
from pyVHR.utils.ecg import ECGsignal
from pyVHR.BPM.BPM import BVPsignal

class CVP(Dataset):
    """
    CVP Dataset

    .. CVP dataset structure:
    .. -----------------
    ..     datasetDIR/
    ..     |
    ..     ||-- 1/
    ..     |   |-- K1_Cropped_Color.mkv
           |   |-- K2_Cropped_Color.mkv
    ..     |   |-- data.csv
    ..     |...
    ..     |...
    """
    name = 'CVP'
    #signalGT = 'ABP'        # GT signal type
    numLevels = 1           # depth of the filesystem collecting video and ABP files
    numSubjects = 2         # number of subjects
    video_EXT = 'mkv'       # extension of the video files
    frameRate = 30          # video frame rate
    VIDEO_SUBSTRING = ''    # substring contained in the filename
    SIG_EXT = 'csv'         # extension of the ABP files
    SIG_SUBSTRING = 'data'  # substring contained in the filename
    SIG_SampleRate = 20000  # sample rate will be calculated dynamically
    show_ECG = False

    def readSigfile(self, filename, signalGT):
        """ Load ground truth signal.

        Returns:
        a pyVHR.utils.ecg.ECGsignal or pyVHR.BPM.BPM.BVPsignal object with the appropriate signal
        """

        # Read ECG or ABP data from CSV file
        data_df = pd.read_csv(filename)

        if signalGT == 'ECG':
            data = data_df[signalGT].values
            # Compute sample rate dynamically
            self.SIG_SampleRate = len(data) / 30
            return ECGsignal(data, self.SIG_SampleRate)
        elif signalGT == 'ABP':
            data = data_df[signalGT].values
            # Compute sample rate dynamically
            self.SIG_SampleRate = len(data) / 30
            return BVPsignal(data, self.SIG_SampleRate)
        elif signalGT == 'CVP':
            data = data_df[signalGT].values
            # Compute sample rate dynamically
            self.SIG_SampleRate = len(data) / 30
            return BVPsignal(data, self.SIG_SampleRate)