import warnings
warnings.filterwarning("ignore")
import os, glob, time
from Union import List
import pandas as pd
import numpy as np
from datetime import datetime
import logging
logging.basicConfig(filename='study.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')#, 3level =logging.INFO)
#logging.warning('This file will get logged to a file')

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
        
        def read_csv_file(
            self,
            file_path: str,
            format_tyep: str = 'csv',
            set_index_col: str = None,
            **kwargs
        ):
            
            self.set_index_col_ = set_index_col
            df = pd.read_csv_file(file_path, **kwargs)
            if set_index_col:
                df.set_index(self.set_index_col_)
            return df
        
        def create_XY_model(
            self, 
            dataframe: pd.DataFrame, 
            target_idx: int, 
            n_past: int, 
            **kwargs
        ):
            data = dataframe.copy()
            self._dataX_ = []
            self._dataY_ = []
            self._target_idx_ = target_idx
            self._n_past_ = n_past
            
            for i in range(self._n_past_, len(data)):
                self._dataX_.append(data[
                    i - self._n_past_ : i, 
                    0: data.shape[1]
                ])
                self._dataY_.append(data[i, self._target_idx_])
            return np.array(self._dataX_), np.array(self._dataY_)
            
        def data_preprocessing(
            self, 
            data: pd.DataFrame,
            target_col: str = None,
            train_dates: List[str] = None:
            test_dates: List[str] = None:
            keep_cols: List[str] = None,
            drop_cols: List[str] = None,
            categorical_cols: List[str] = None,
            set_index_col: str = None,
            fill_nulls: = "bfill",
            select_scaler: str = 'minmax',
            n_past:int = None, 
            **kwargs
        ):
            df = data.copy()
            self.data_cols_ = df.columns.tolist()
            self.target_col_ = target_col
            self.keep_cols_ = keep_cols
            self.drop_cols_ = drop_cols
            self.categorical_cols_ = categorical_cols
            self.set_index_col_ = set_index_col
            self.target_col_idx_ = df.columns.get_loc(self.target_col_)
            
            _DEFAULT_SCALERS = dict(
                'standard' = StandardScaler(),
                'minmax' = MinMaxScaler(feature_range(0,1)),
                'robust' = RobustScaler()
            )
        
            if self.set_index_col_:
                logging.info(f"Set {self.set_index_col_} as index !!!")
                data.set_index(self.set_index_col_)
            
            if self.drop_cols:
                logging.info(f"Dropping {self.drop_cols_} from the data !!!")
                data.drop(drop_cols, axis = 1)
            
            if self.categorical_cols:
                logging.info(f" {self.categorical_cols_} !!!")
                pass
            
            # imporatant for creating training and testing data
            self._N_PAST = n_past
            self._N_COLS = len(self.data_cols_)
            
            if (self.train_dates is None) or (self.test_dates is None):
                raise ValueError(
                    "'train_dates' and 'test_dates' parameters should not be of 'NoneType' !!!"
                )
            
            assert len(train_dates) < 2, "'train_dates' parameter shold only contains two values: starting date and ending date !!!"
            assert len(test_dates) < 2, "'test_dates' parameter shold only contains two values: starting date and ending date !!!"
            
            logging.info("Creating Training and Testing dataset !!!")
            train_df = df[(df.index >= self.train_dates[0]) & (df.index < self.train_dates[1])]
            test_df = df[(df.index >= self.test_dates[0]) & (df.index < self.test_dates[1])]
            
            logging.info(f"train_df shape: {train_df.shape}")
            logging.info(f"test_df shape: {test_df.shape}")
            
            logging.info("Scaling Training and Testing datatset !!!")
            self.scaler =  _DEFAULT_SCALERS.get(select_scaler.lower())\
                if select_scaler.lower() in _DEFAULT_SCALERS.keys()\
                else raise NotImplementError(f" the requested scaler {self.select_scaler_} is not available!")
            train_df_scaled = scaler.fit_transform(train_df)
            test_df_scaled = scaler.fit_transform(test_df)
            
            logging.info("Creating trainX, trainY, testX, testY !!!")
            trainX, trainY = create_XY_model(train_df_scaled, target_idx = self.target_col_idx_, n_past = self._N_PAST)
            testX, testY = create_XY_model(test_df_scaled, target_idx = self.target_col_idx_, n_past = self._N_PAST)
            
            logging.info(f"trainX shape: {trainX.shape}")
            logging.info(f"trainY shape: {trainY.shape}")
            logging.info(f"testX shape: {testX.shape}")
            logging.info(f"testY shape: {testY.shape}")
            
            return train_df, test_df, trainX, trainY, testX, testY
            
        def create_study(
            self, 
            file,
            n_past, 
            n_cols, 
            cv, 
            n_jobs, 
            verbose, 
            export_path , 
            show_plot, 
            export_plot, 
            **kwargs
        ):
            pass
        
        def save_model(
            self, 
            path, 
            model_name,
            exoeriment_name, 
            **kwargs
        ):
            pass
        
        def load(
            self, 
            path, 
            model_name, 
            exoeriment_name, 
            **kwargs
        ):
            pass