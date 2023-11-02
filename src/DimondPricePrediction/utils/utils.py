import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_obj(file_path, obj):
    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok= True)

        with open (file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise customexception(e, sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models:dict):
    try:
        report= {}
        for i in range(len(models)):
            model= list(models.values())[i]

            # Transforming data
            model.fit(x_train,y_train)

            # Predict the model
            y_pred= model.predict(x_test)

            # Get r2 scores of the model
            test_model_score= r2_score(y_test,y_pred)

            report[list(models.keys())[i]]= test_model_score
        
        return report
        
    except Exception as e:
        raise customexception(e,sys)

def load_obj(file_path):
    try:
        with open (file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Error occured while load_obj")
        raise customexception(e,sys)