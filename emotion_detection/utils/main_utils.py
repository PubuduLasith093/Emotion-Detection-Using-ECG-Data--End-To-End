import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame
import pandas as pd
import neurokit2 as nk

from emotion_detection.exception import Emotion_detection_Exception
from emotion_detection.logger import logging
from emotion_detection.constants import *


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e


def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e
    
def process_ecg_data_and_create_dataframe(self, data) -> DataFrame:
            ecg_data = []
            a = np.zeros((NO_OF_PARTICIPANTS, NO_OF_VIDEOS, NO_OF_CLASSES))  # For binary scores
            try:
                # Loop through participants and conditions
                for k in range(23):  # Participants
                    for j in range(18):  # Conditions
                        # Extract ECG data
                        basl_l = data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['baseline'][0,0][j,0][:,0]
                        stim_l = data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['stimuli'][0,0][j,0][:,0]
                        basl_r = data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['baseline'][0,0][j,0][:,1]
                        stim_r = data['DREAMER'][0,0]['Data'][0,k]['ECG'][0,0]['stimuli'][0,0][j,0][:,1]
                        
                        # Append data to the list
                        ecg_data.append({
                            "Participant": k,
                            "Condition": j,
                            "Baseline_Left": basl_l,
                            "Stimuli_Left": stim_l,
                            "Baseline_Right": basl_r,
                            "Stimuli_Right": stim_r
                        })

                        # Score processing for binary classifications
                        a[k, j, 0] = 0 if data['DREAMER'][0,0]['Data'][0,k]['ScoreValence'][0,0][j,0] < 4 else 1
                        a[k, j, 1] = 0 if data['DREAMER'][0,0]['Data'][0,k]['ScoreArousal'][0,0][j,0] < 4 else 1
                        a[k, j, 2] = 0 if data['DREAMER'][0,0]['Data'][0,k]['ScoreDominance'][0,0][j,0] < 4 else 1

                # Convert list of dictionaries to DataFrame
                ecg_df = pd.DataFrame(ecg_data)

                # Reshape the score array to match the DataFrame's expected row order and add as new columns
                flattened_scores = a.reshape(-1, 3)
                ecg_df['Valence_Binary'] = flattened_scores[:, 0]
                ecg_df['Arousal_Binary'] = flattened_scores[:, 1]
                ecg_df['Dominance_Binary'] = flattened_scores[:, 2]
                return ecg_df 
            except Exception as e:
                raise Emotion_detection_Exception(e, sys) from e
            
def extract_features(self, df:DataFrame)-> DataFrame:
    df['Baseline_Left'] = df['Baseline_Left'].apply(lambda x: np.array(x.split(','), dtype=int))
    df['Stimuli_Left'] = df['Stimuli_Left'].apply(lambda x: np.array(x.split(','), dtype=int))
    df['Baseline_Right'] = df['Baseline_Right'].apply(lambda x: np.array(x.split(','), dtype=int))
    df['Stimuli_Right'] = df['Stimuli_Right'].apply(lambda x: np.array(x.split(','), dtype=int))
    EXTRACTED_ECG_DF={}
    try:
        for i in range(df.shape[0]):
            ecg_signals_b_l,info_b_l=nk.ecg_process(df.iloc[i]['Baseline_Left'],sampling_rate=256)
            ecg_signals_s_l,info_s_l=nk.ecg_process(df.iloc[i]['Stimuli_Left'],sampling_rate=256)
            ecg_signals_b_r,info_b_r=nk.ecg_process(df.iloc[i]['Baseline_Right'],sampling_rate=256)
            ecg_signals_s_r,info_s_r=nk.ecg_process(df.iloc[i]['Stimuli_Right'],sampling_rate=256)
            processed_ecg_l=nk.ecg_intervalrelated(ecg_signals_s_l)/nk.ecg_intervalrelated(ecg_signals_b_l)
            processed_ecg_r=nk.ecg_intervalrelated(ecg_signals_s_r)/nk.ecg_intervalrelated(ecg_signals_b_r)
            processed_ecg=(processed_ecg_l+processed_ecg_r)/2
            processed_ecg['participent'] = df['Participant'][i]
            processed_ecg['Video'] = df['Condition'][i]
            processed_ecg['Valence'] = df['Valence_Binary'][i]
            processed_ecg['Arousal'] = df['Arousal_Binary'][i]
            processed_ecg['Dominance'] = df['Dominance_Binary'][i]
            if not len(EXTRACTED_ECG_DF):
                EXTRACTED_ECG_DF=processed_ecg
            else:
                EXTRACTED_ECG_DF=pd.concat([EXTRACTED_ECG_DF,processed_ecg],ignore_index=True)

        cols = ['participent', 'Video'] + [col for col in EXTRACTED_ECG_DF.columns if col not in ['participent', 'Video']]
        EXTRACTED_ECG_DF = EXTRACTED_ECG_DF[cols]
        return EXTRACTED_ECG_DF
    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e
    
def select_features(self, df:DataFrame, selected_columns)-> DataFrame:
    try:
        selected_df = df[selected_columns]
        return selected_df
    except Exception as e:
        raise Emotion_detection_Exception(e, sys) from e