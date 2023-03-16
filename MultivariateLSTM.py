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
        n_past:int = 14, 
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
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        trainX: np.array,
        trainY: np.array,
        testX: np.array,
        testY: np.array,
        file_path: str = None,
        n_past: int = 14, 
        n_cols: int = None, 
        cv = 3, 
        n_jobs = -1, 
        verbose = 2, 
        export_path: str = None, 
        show_plot: bool = False, 
        export_plot: bool = False, 
        **kwargs
    ):
        if file_path not None:
            df = read_csv_file(file_path = file_path):
            self.train_df, self.test_df, self.trainX, self.trainY, self.testX, self.testY = data_preprocessing(
                                                                    data = df,
                                                                    target_col= None,
                                                                    train_dates = None:
                                                                    test_dates = None:
                                                                    keep_cols = None,
                                                                    drop_cols = None,
                                                                    categorical_cols = None,
                                                                    set_index_col = None,
                                                                    fill_nulls = "bfill",
                                                                    select_scaler = 'minmax',
                                                                    n_past = n_past, 
                                                                    **kwargs
                                                            )
        else:
            self.train_df = train_df.copy(), 
            self.test_df = test_df.copy(), 
            trainX = trainX.copy(), 
            trainY = trainY.copy(), 
            testX = testX.copy(), 
            testY = testY.copy()

        def _build_model(optimizer):
            grid_model = Sequential()
            grid_model.add(LSTM(128, return_sequence = True, input_shape = (self._N_PAST, self._N_COLS)))
            grid_model.add(LSTM(128)
            grid_model.add(Dropout(0.2))
            grid_model.add(Dense(1))
            grid_model.compile(loss = 'mse', optimizer = optimizer)
            return grid_model
        
        grid_search = GridSearchCV(
            estimator = kerasRegressor(
                            build_fn = _build_model,
                            verbose = verbose,
                            validation_data = (testX, testY)
                        ),
            param_grid = {
                    'batch_size' : [32, 64, 128],
                    'epochs' : [6, 8, 10],
                    'optimizers' : ['adam', 'Adamdelta']
                },
            n_jobs = n_jobs,
            verbose = verbose    
        )        
        grid_search = grid_search.fit(trainX, trainY)
        
        logging.info(f"Best Parameters : {grid_search.best_params)} \n")
        self.best_model_ = grid_search.best_estimator_.model
        
        return self.best_model_
        
    def predict(self, testX: np.array, testY: np.array):
        
        logging.info("Start prediction on testing data !!!")
        prediction = self.best_model_.predict(self.testX)
        logging.info(f"Prediction shape: {prediction.shape}")
        
        copy_pred_array = np.repeat(prediction, self._N_COLS, axis = 1)
        pred = self.scaler.inverse_transform(np.reshape(copy_pred_array, (len(prediction), self._N_COLS)))[:self.target_col_idx_]
        copy_original_array = np.repeat(testY, self._N_COLS, axis = 1)
        logger.info(f"Actual Target shape: {copy_original_array.shape}")
        original = self.scaler.inverse_transform(np.reshape(copy_original_array, (len(testY), self._N_COLS)))[:self.target_col_idx_]
        
        logging.info(f"Preparing prediction data !!!")
        
        temp_df = self.test_df.merge(
            pd.DataFrame({
                f"{self.target_}_predicted" : pred.tolist()}, 
                index = self.test_df.iloc[:-self._N_PAST, :].index),
            how = 'outer',
            left_index = True,
            right_index = True
        )
        
        df_past = temp_df.iloc[-self._N_PAST*2 : self._N_PAST, :-1].copy(deep = True)
        df_future = temp_df.iloc[-self._N_PAST: , :-1].copy(deep = True)
        df_past_idx = df_past.index
        df_future_idx = df_future.index
        full_idx = df_past_idx.append(df_future_idx)
        
        old_scaled_array = self.scaler.transform(df_past)
        new_scaled_array = self.scaler.transform(df_future)
        full_df = pd.concat([pd.DataFrame(old_scaled_array), new_scaled_array]).set_index(full_idx)
        full_df_scaled_array = full_df.values
        all_data = []
        for i in range(self._N_PAST, len(full_df_scaled_array)):
            data_x = []
            data_x.append(full_df_scaled_array[i - self._N_PAST : i, 0:full_df_scaled_array.shape[1]])
            data_x = np.array(data_x)
            predictions = self.best_model_.predict(data_x)
            all_data.append(predictions)
        new_array = np.array(all_data).reshape(-1,1)
        copy_pred_array = np.repeat(new_array, self._N_COLS, axis = 1)
        y_pred_future = self.scaler.inverse_transform(np.reshape(copy_pred_array,(len(new_array), self._N_COLS)))[:,0]
        
        temp_future_df = pd.DataFrame(temp_df.loc[temp_df[f"{self.target_}_predicted"].isna()][f"{self.target_}_predicted"])
        temp_future_df[f"{self.target_}_predicted"] = y_pred_future
        res_df = temp_df.merge(temp_future_df, how = 'left', left_index = True, right_index = True).fillna(0)
        res_df[f"{self.target_}_predicted"] = res_df[f"{self.target_}_predicted_x"] + res_df[f"{self.target_}_predicted_y"]
        
        self.mean_absolute_error_ = mean_absolute_error(res_df[f"{self.target_}"], res_df[f"{self.target_}_predicted"])
        self.root_mean_square_error_ = mean_square_error(res_df[f"{self.target_}"], res_df[f"{self.target_}_predicted"], squared = False)
        self.mean_absolute_percentage_error_ = mean_absolute_percentage_error(res_df[f])
        self.r2_score_ = r2_score(res_df[f"{self.target_}"], res_df[f"{self.target_}_predicted"])
        
        logging.info(f"Mean Absolute Error : {self.mean_absolute_error_}")
        logging.info(f"Root Mean Squre Error : {self.root_mean_square_error_}")
        logging.info(f"Mean Absolute Percentage Error : {self.mean_absolute_percentage_error_}")
        logging.info(f"R2 Score : {self.r2_score_}")
        
        return res_df
        
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