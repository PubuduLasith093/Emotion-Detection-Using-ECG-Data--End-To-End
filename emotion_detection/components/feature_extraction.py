import json
import sys

import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from pandas import DataFrame
import os

from emotion_detection.exception import Emotion_detection_Exception
from emotion_detection.logger import logging
from emotion_detection.utils.main_utils import read_yaml_file, write_yaml_file, extract_features
from emotion_detection.entity.artifact_entity import DataIngestionArtifact, FeatureExtractionArtifact
from emotion_detection.entity.config_entity import FeatureExtrcationConfig
from emotion_detection.constants import SCHEMA_FILE_PATH

class FeatureExtraction:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, feature_extraction_config: FeatureExtrcationConfig):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.feature_extraction_config = feature_extraction_config
        except Exception as e:
            raise Emotion_detection_Exception(e,sys)
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Emotion_detection_Exception(e, sys)
        
    def initiate_feature_extraction(self) -> FeatureExtractionArtifact:
        """
        Method Name :   initiate_feature_extraction
        Description :   This method initiates the feature extraction component for the pipeline
        
        Output      :   Returns train and test dataframes with extracted featuers
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Starting feature extraction")
            train_df, test_df = (FeatureExtraction.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                    FeatureExtraction.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            

            extracted_train_df = extract_features(self,train_df)
            extracted_test_df = extract_features(self,test_df)

            dir_path = os.path.dirname(self.feature_extraction_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            extracted_train_df.to_csv(self.feature_extraction_config.training_file_path,index=False,header=True)
            extracted_test_df.to_csv(self.feature_extraction_config.testing_file_path,index=False,header=True)

            feature_extraction_artifact = FeatureExtractionArtifact(
                trained_file_path=self.feature_extraction_config.training_file_path,
                test_file_path=self.feature_extraction_config.testing_file_path)
                

           
            return feature_extraction_artifact
        except Exception as e:
            raise Emotion_detection_Exception(e, sys) from e

