import json
import sys

import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from pandas import DataFrame
import os

from emotion_detection.exception import Emotion_detection_Exception
from emotion_detection.logger import logging
from emotion_detection.utils.main_utils import read_yaml_file, write_yaml_file, extract_features, select_features
from emotion_detection.entity.artifact_entity import FeatureExtractionArtifact, FeatureSelectionArtifact
from emotion_detection.entity.config_entity import FeatureSelectionConfig
from emotion_detection.constants import SCHEMA_FILE_PATH

class FeatureSelection:
    def __init__(self, feature_extraction_artifact: FeatureExtractionArtifact, feature_selection_config: FeatureSelectionConfig):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.feature_extraction_artifact = feature_extraction_artifact
            self.feature_selection_config = feature_selection_config
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise Emotion_detection_Exception(e,sys)
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Emotion_detection_Exception(e, sys)

    def initiate_feature_selection(self) -> FeatureSelectionArtifact:
        """
        Method Name :   initiate_feature_selection
        Description :   This method initiates the feature selection component for the pipeline
        
        Output      :   Returns train and test dataframes with selected featuers
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Starting feature selection")
            train_df, test_df = (FeatureSelection.read_data(file_path=self.feature_extraction_artifact.trained_file_path),
                                    FeatureSelection.read_data(file_path=self.feature_extraction_artifact.test_file_path))
            

            selected_train_df = select_features(self, train_df, self._schema_config["selected_columns"])
            selected_test_df = select_features(self, test_df, self._schema_config["selected_columns"])

            dir_path = os.path.dirname(self.feature_selection_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            selected_train_df.to_csv(self.feature_selection_config.training_file_path,index=False,header=True)
            selected_test_df.to_csv(self.feature_selection_config.testing_file_path,index=False,header=True)

            feature_selection_artifact = FeatureSelectionArtifact(
                trained_file_path=self.feature_selection_config.training_file_path,
                test_file_path=self.feature_selection_config.testing_file_path)
                

           
            return feature_selection_artifact
        except Exception as e:
            raise Emotion_detection_Exception(e, sys) from e
