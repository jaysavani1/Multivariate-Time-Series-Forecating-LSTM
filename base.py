from abc import abstractmethod
import pickle

class BaseModel:
    
    def __init__(self):
        pass
    
    @abstractmethod
    def data_preprocessing(self, data, **kwargs):
        pass
    
    @abstractmethod
    def create_XY_model(self, data, target_idx, n_past, **kwargs):
       pass
    
    @abstractmethod
    def create_study(self, file, n_past, n_cols, cv, n_jobs, verbose, export_path , show_plot, export_plot, **kwargs):
        pass
    
    @abstractmethod
    def save_model(self, path, model_name, exoeriment_name, **kwargs):
        pass
    
    @classmethod
    def load(self, path, model_name, exoeriment_name, **kwargs):
        pass