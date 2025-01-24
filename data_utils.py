import mne
import numpy as np
import xml.etree.ElementTree as ET


def read_shh1_data(data_path, label_path) -> tuple:

    """
    Load .edf data and arousal locations

    Parameters
    -----------
        data_path : path to edf data
        label_path: path to locations

    Returns
    ----------
        a tuple containing a ndarray of edf data and 
        an ndarray of arousals containing zeros and ones
    """

    # read raw data
    raw_data = mne.io.read_raw_edf(data_path, verbose=0,
                                  exclude=['SaO2', 'H.R.', 'ECG', 'SOUND', 
                                           'AIRFLOW', 'RES', 'THOR RES', 'ABDO RES', 'POSITION', 
                                           'LIGHT', 'NEW AIR', 'OX stat', 'stat', 'EPMS', 'AIR', 'AUX'],
                                  infer_types=False)
    
    # get ch names   
    raw_data.pick(['EEG'])

    # sampling frequency
    fs = raw_data.info["sfreq"]
    if fs > 125:
        raw_data.resample(sfreq=125)

    # read data
    data = raw_data.get_data().T
    
    # standardize
    mu = data.mean(axis=0, keepdims=True)
    sigma = data.std(axis=0, keepdims=True)
    data = (data - mu) / sigma

    # read stage labels
    tree = ET.parse(label_path)
    root = tree.getroot()
    stages = np.array([stage.text for stage in root[4].findall("SleepStage")], dtype=np.int8)
    
    # merge stage 3 and 4, change 5 to 4
    stages[stages == 4] = 3
    stages[stages == 5] = 4
    stages[stages > 5] = 5

   
   
    # return data and stages
    # return data[:int(stages.shape[0] * 30 * fs)], stages
    return data[:int(stages.shape[0] * 30 * fs)], stages