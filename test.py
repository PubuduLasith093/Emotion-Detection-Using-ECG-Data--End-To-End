import json
import sys
import pandas as pd
from emotion_detection.constants import SCHEMA_FILE_PATH
from emotion_detection.utils.main_utils import read_yaml_file, write_yaml_file

_x=read_yaml_file(file_path=SCHEMA_FILE_PATH)
df = pd.read_csv('train.csv')
print(df)
x = _x["selected_columns"]
print(x)
# for column in _x["selected_columns"]:
#     print(column)
# print(type(x["columns"]))
# print(x["columns"].keys())