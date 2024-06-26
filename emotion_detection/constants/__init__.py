import os
from datetime import date

RAW_FILE_PATH ="D:/BK/DREAMER.mat"
PIPELINE_NAME: str = "bank_churn"
ARTIFACT_DIR: str = "artifact"

NO_OF_PARTICIPANTS = 23
NO_OF_VIDEOS = 18
# NO_OF_PARTICIPANTS = 6
# NO_OF_VIDEOS = 18
NO_OF_CLASSES = 3

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "churn"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "emotion_detection.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")



"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "BANK_CHURN_NEW"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
feature extraction related constant start with FEATURE_EXTRACTION VAR NAME
"""
FEATURE_EXTRACTION_DIR_NAME: str = "feature_extraction"
FEATURE_EXTRACTION_EXTRACTED_DIR: str = "extracted"


"""
feature selection related constant start with FEATURE_SELECTION VAR NAME
"""
FEATURE_SELECTION_DIR_NAME: str = "feature_selection"
FEATURE_SELECTION_EXTRACTED_DIR: str = "selected"

