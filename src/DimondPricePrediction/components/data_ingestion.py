import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging

from sklearn.model_selection import train_test_split
from src.DimondPricePrediction.exception import customexception
from dataclasses import dataclass
from pathlib import Path
import os
import sys

class DataIngestionConfig:
    raw_data_path:str= os.path.join('artifacts', 'raw.csv')
    train_data_path:str= os.path.join('artifacts', 'train.csv')
    test_data_path: str= os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Data ingestion started')
        try:
            data= pd.read_csv(Path(os.path.join("notebooks/data", "gemstone.csv")))
            logging.info('I have read the dataset as df')
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok= True)
            data.to_csv(self.ingestion_config.raw_data_path,index= False)
            logging.info("Raw data has been saved to the artifacts folder")

            logging.info("Here i perform train test split")
            train_data,test_data= train_test_split(data, test_size=0.25, random_state= 3)

            logging.info("Train test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path,index= False)
            test_data.to_csv(self.ingestion_config.test_data_path, index= False)
            logging.info("Data ingestion part completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('exception occured at data ingestion stage')
            raise customexception(e,sys)
    