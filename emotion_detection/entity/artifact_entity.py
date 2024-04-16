from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str 

@dataclass
class FeatureExtractionArtifact:
    trained_file_path:str 
    test_file_path:str 