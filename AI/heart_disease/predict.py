import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
import joblib

def joblibModel():
    


def predict(FILE_PATH):
    file_list = os.listdir(FILE_PATH)
    for list in file_list:
        filename, ext = os.path.splitext
        if ext == '.joblib':
            