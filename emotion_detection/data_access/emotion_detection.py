from emotion_detection.constants import RAW_FILE_PATH
from emotion_detection.exception import Emotion_detection_Exception
from emotion_detection.utils.main_utils import process_ecg_data_and_create_dataframe
import pandas as pd
import scipy.io as sio
import sys
from typing import Optional
import numpy as np



class EmotionDetection:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        
        try:
            self.file_path = RAW_FILE_PATH
        except Exception as e:
            raise Emotion_detection_Exception(e,sys)
    def export_mat_file_as_dataframe(self)->pd.DataFrame:
        try:
            data = sio.loadmat(self.file_path)
            df = process_ecg_data_and_create_dataframe(self, data)
            return df

        except Exception as e:
            raise Emotion_detection_Exception(e,sys)