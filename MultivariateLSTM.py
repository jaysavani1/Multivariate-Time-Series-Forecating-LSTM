import os, glob, time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#TO DO: Plotly

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.wrappers.scikit_learn import kerasRegressor

from base import BaseModel

class MultivariateLSTM(BaseModel):
    def __init__(
        self        
    ):
            
        def data_preprocessing(self, data, **kwargs):
            pass
        
        def create_XY_model(self, data, target_idx, n_past, **kwargs):
           pass
        
        def create_study(self, file, n_past, n_cols, cv, n_jobs, verbose, export_path , show_plot, export_plot, **kwargs):
            pass
        
        def save_model(self, path, model_name, exoeriment_name, **kwargs):
            pass
        
        def load(self, path, model_name, exoeriment_name, **kwargs):
            pass