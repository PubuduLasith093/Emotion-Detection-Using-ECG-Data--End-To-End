from emotion_detection.pipline.training_pipeline import TrainPipeline
import warnings
warnings.filterwarnings('ignore')

pipeline = TrainPipeline()
pipeline.run_pipeline()